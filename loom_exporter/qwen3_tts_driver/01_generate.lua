    -- The voice. Either the caller's own x-vector -- 1024 floats it got from a previous call, or
    -- from anywhere else -- or one extracted here from a reference clip at 24 kHz. A topology input
    -- takes an array or a reference indifferently, so the two paths differ only in this assignment.
    local _spk
    if inputs.x_vector then
        _spk = inputs.x_vector
    else
        loom.run_subgraph_and_retain('speaker_encoder',
            {n_samples = #inputs.waveform, n_past = 0}, {waveform = inputs.waveform})
        _spk = {from = 'speaker_encoder'}
    end

    -- The prompt and the TEXT SCHEDULE, in one call. This model is streaming: only the first text
    -- token is in the prompt, and every later one is added to a generated frame's embedding, one per
    -- step. `schedule` is that sequence with `tts_pad` appended, so "use trailing[step] while there
    -- is one, else pad" becomes an index rather than a branch.
    local _language = inputs.language_id or DEFAULT_LANGUAGE_ID
    loom.run_subgraph_and_retain('prefill_embed', {n_tokens = #inputs.tokens, n_past = 0},
        {text_ids = inputs.tokens, language_id = {_language}, x_vector = _spk})

    -- The template contributes three tokens before the text and five after, and the schedule drops
    -- the first text token (it is in the prompt) and appends `tts_eos` and `tts_pad`. So its row
    -- count is `#tokens - 9 + 2`, and the driver needs it only to clamp the index at the end.
    local _rows = #inputs.tokens - 7

    loom.run_subgraph_and_retain('talker', {n_tokens = PREFILL_LEN, n_past = 0},
        {inputs_embeds = {from = 'prefill_embed', index = 1},
         position_ids = loom.range(0, PREFILL_LEN),
         attention_mask = loom.causal_mask(PREFILL_LEN, 0)})

    if inputs.seed then loom.seed_rng(inputs.seed) end
    local _temperature = inputs.temperature or TEMPERATURE
    local _top_k = inputs.top_k or TOP_K
    local _top_p = inputs.top_p or TOP_P
    local _penalty = inputs.repetition_penalty or REPETITION_PENALTY
    local _sub_temperature = inputs.subtalker_temperature or SUB_TEMPERATURE
    local _sub_top_k = inputs.subtalker_top_k or SUB_TOP_K
    local _sub_top_p = inputs.subtalker_top_p or SUB_TOP_P
    local _max_new = inputs.max_new_tokens or MAX_NEW_TOKENS

    local _codes, _generated = {}, {}
    for _step = 0, _max_new - 1 do
        -- **`hi` is how `min_new_tokens` is expressed**, and it costs nothing because the export
        -- already trimmed the head to the drawable ids: EOS is its last row, so excluding it is a
        -- one-shorter window rather than a mask.
        local _hi = EOS_INDEX
        if _step >= MIN_NEW_TOKENS then _hi = EOS_INDEX + 1 end

        -- **The repetition penalty is not optional here, and greedy is not an exception.**
        -- `transformers` applies it as a processor rather than a warper, so it moves an argmax too;
        -- a greedy decode without it never emits EOS and runs to `max_new_tokens`. `_generated` is
        -- the driver's own history, which is the half of the knob the checkpoint cannot state.
        local _first = loom.sample_row('talker', 0, {
            temperature = _temperature, top_k = _top_k, top_p = _top_p,
            lo = 0, hi = _hi,
            repetition_penalty = _penalty, penalized = _generated})
        if _first == EOS_INDEX then break end
        _generated[#_generated + 1] = _first

        -- The code predictor, conditioned on the talker's hidden state and on codebook 0.
        --
        -- **It is not KV-cached, and it re-reads its own prefix every step.** One KvCache serves a
        -- whole model with one per-layer width, and these two phases' K/V geometry does not agree --
        -- so rather than force it, the smaller model stops needing a cache. It never exceeds sixteen
        -- positions, so the whole frame costs 135 row-forwards of a five-layer stack, against one
        -- 28-layer talker step beside it. What it buys is that the driver still passes only
        -- integers: the prefix is rebuilt in-graph from `_prefix`, one longer each step.
        local _hidden = {from = 'talker', index = 2}
        loom.run_subgraph_and_retain('predictor_prefill', {n_tokens = 2, n_past = 0},
            {first_id = {_first}, talker_hidden = _hidden,
             position_ids = loom.range(0, 2),
             attention_mask = loom.causal_mask(2, 0)})

        -- **Group `g` is the window `[g*V, (g+1)*V)` of one 15-way-concatenated head**, which is what
        -- keeps fifteen draws on one head. A windowed draw returns the ABSOLUTE index, so the number
        -- that comes back is already the merged embedding table's row -- `_prefix` holds those
        -- unchanged, and the real codes are recovered by subtracting the offset once, at the end.
        local _frame, _prefix = {_first}, {}
        local _module = 'predictor_prefill'
        for _g = 0, N_GROUPS - 2 do
            local _row = loom.sample_row(_module, 0, {
                temperature = _sub_temperature, top_k = _sub_top_k, top_p = _sub_top_p,
                lo = _g * CODEBOOK_SIZE, hi = (_g + 1) * CODEBOOK_SIZE})
            _frame[#_frame + 1] = _row
            if _g < N_GROUPS - 2 then
                _prefix[#_prefix + 1] = _row
                local _n = #_prefix + 2
                loom.run_subgraph_and_retain('predictor_steps', {n_tokens = _n, n_past = 0},
                    {first_id = {_first}, talker_hidden = _hidden, rows = _prefix,
                     position_ids = loom.range(0, _n),
                     attention_mask = loom.causal_mask(_n, 0)})
                _module = 'predictor_steps'
            end
        end

        _codes[#_codes + 1] = _first
        for _g = 0, N_GROUPS - 2 do
            _codes[#_codes + 1] = _frame[_g + 2] - _g * CODEBOOK_SIZE
        end

        -- This frame's sixteen embeddings summed, plus one row of the text schedule, is the talker's
        -- next input. Both operands stay engine-side: the codes are integers and the schedule is
        -- `prefill_embed`'s own retained output, indexed by a row number.
        local _schedule_row = _step
        if _schedule_row > _rows - 1 then _schedule_row = _rows - 1 end
        loom.run_subgraph_and_retain('frame_embed', {n_tokens = _rows, n_past = 0},
            {codes = _frame, schedule_row = {_schedule_row},
             schedule = {from = 'prefill_embed', index = 2}})
        loom.run_subgraph_and_retain('talker', {n_tokens = 1, n_past = PREFILL_LEN + _step},
            {inputs_embeds = {from = 'frame_embed'},
             position_ids = loom.range(PREFILL_LEN + _step, 1),
             attention_mask = loom.causal_mask(1, PREFILL_LEN + _step)})
    end
