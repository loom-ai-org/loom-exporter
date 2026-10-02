    -- ===== Soprano: one generation per sentence, its hidden rows decoded, the audio concatenated. =====
    --
    -- Inputs: `tokens` (required), and the sampler's knobs, each defaulting to the reference's own:
    -- `temperature` (0.001), `top_k` (50), `top_p` (0.95), `repetition_penalty` (1.2, over the prompt
    -- and every id drawn so far), `max_new_tokens` (512 per sentence) and `seed`. Returns the waveform
    -- at 32 kHz.
    --
    -- `inputs.tokens` is what the vocabulary returns: every sentence already in the reference's prompt
    -- form, `[STOP][TEXT]{sentence}[START]`, one after another. `[STOP]` is the model's EOS and the
    -- prompt's first id, so it is also where one sentence's ids end and the next begin -- the cleaned
    -- text has no brackets left to spell it with. Ids that do not open with it are a caller's own bare
    -- sentence and are wrapped here, as one.
    if inputs.seed then loom.seed_rng(inputs.seed) end
    local _temperature = inputs.temperature or DEFAULT_TEMPERATURE
    local _top_k = inputs.top_k or DEFAULT_TOP_K
    local _top_p = inputs.top_p or DEFAULT_TOP_P
    local _penalty = inputs.repetition_penalty or DEFAULT_REPETITION_PENALTY
    local _max_new = inputs.max_new_tokens or MAX_NEW_TOKENS

    local _chunks = {}
    if inputs.tokens[1] ~= EOS_ID then
        local _c = {EOS_ID, TEXT_ID}
        for _i = 1, #inputs.tokens do _c[#_c + 1] = inputs.tokens[_i] end
        _c[#_c + 1] = START_ID
        _chunks[1] = _c
    else
        for _i = 1, #inputs.tokens do
            local _id = inputs.tokens[_i]
            if _id == EOS_ID then _chunks[#_chunks + 1] = {} end
            local _c = _chunks[#_chunks]
            _c[#_c + 1] = _id
        end
    end
    if #_chunks == 0 then error('soprano: no text ids') end

    wave = {}
    for _n = 1, #_chunks do
        local _prompt = _chunks[_n]
        -- `tokenizer(..., truncation=True, max_length=512)`: the reference keeps the first 512.
        if #_prompt > MAX_PROMPT_TOKENS then
            local _cut = {}
            for _i = 1, MAX_PROMPT_TOKENS do _cut[_i] = _prompt[_i] end
            _prompt = _cut
        end
        local _n_prompt = #_prompt
        loom.run_subgraph_and_retain('lm', {n_tokens = _n_prompt, n_past = 0},
            {input_ids = _prompt, position_ids = loom.range(0, _n_prompt),
             attention_mask = loom.causal_mask(_n_prompt, 0)})

        -- `RepetitionPenaltyLogitsProcessor` reads every id of `input_ids`, which in `generate` is the
        -- prompt AND what has been drawn so far -- so the history starts as the prompt.
        local _history = {}
        for _i = 1, _n_prompt do _history[_i] = _prompt[_i] end
        local _rows, _n_rows = {}, 0
        local _n_past = _n_prompt
        for _step = 1, math.min(_max_new, LM_MAX_POSITIONS - _n_prompt) do
            local _id = loom.sample_row('lm', 0, {
                temperature = _temperature, top_k = _top_k, top_p = _top_p,
                repetition_penalty = _penalty, penalized = _history})
            if _id == EOS_ID then break end
            -- The row that PRODUCED this id is the one the decoder reads, so it is kept before the
            -- id is fed back; the row that produces `[STOP]` is never kept.
            local _h = loom.get_output('lm', 2)
            local _base = _n_rows * HIDDEN_SIZE
            for _i = 1, HIDDEN_SIZE do _rows[_base + _i] = _h[_i] end
            _n_rows = _n_rows + 1
            _history[#_history + 1] = _id
            loom.run_subgraph_and_retain('lm', {n_tokens = 1, n_past = _n_past},
                {input_ids = {_id}, position_ids = {_n_past},
                 attention_mask = loom.causal_mask(1, _n_past)})
            _n_past = _n_past + 1
        end

        -- L rows decode to `2048 * (L - 1)` samples, so a sentence needs two rows to make a sound.
        if _n_rows >= 2 then
            local _wave = loom.run_subgraph('decoder', {n_codes = _n_rows, n_past = 0}, {hidden = _rows})
            local _base = #wave
            for _i = 1, #_wave do wave[_base + _i] = _wave[_i] end
        end
    end
    if #wave == 0 then error('soprano: the LM produced no audio rows') end
