    -- ===== The flow LM: text -> latents, one continuous 32-d frame per step, chunk by chunk. =====
    --
    -- `inputs.tokens` is what `loom::PocketTtsVocab::encode` returns: each sentence chunk's prepared
    -- ids, opened by a header saying which tail the reference guessed for it. The header is not fed to
    -- the model: the reference feeds the tokenizer's output straight to the conditioner.
    if inputs.seed then loom.seed_rng(inputs.seed) end
    local _starts = loom.get_weight('step_embed', 'word_start')
    local _chunks = pocket_split_chunks(inputs.tokens, CHUNK_HEADER_SHORT, CHUNK_HEADER_LONG, _starts,
                                         SHORT_CHUNK_MAX_WORDS)
    if #_chunks == 0 then error('pocket-tts: no text ids') end
    local _std = math.sqrt(inputs.temperature or DEFAULT_TEMPERATURE)
    local _threshold = inputs.eos_threshold or DEFAULT_EOS_THRESHOLD
    local _n_steps = inputs.n_steps or DEFAULT_DECODE_STEPS
    -- Run-wide counters, so a caller's pinned draws (`inputs.noise`, one `[32]` row per step TAKEN,
    -- row-major) and teacher latents (`inputs.teacher_latents`, one row per frame KEPT) run on across
    -- chunks in the order the reference consumes them.
    local _draw = 0
    local _kept = 0
    wave = {}

    for _c = 1, #_chunks do
        local _text = _chunks[_c].ids
        local _n_text = #_text
        -- The voice is the flow LM's KV cache after its own prefill, so it is WRITTEN into the cache
        -- rather than run: rows [0, n_voice) are the voice, and the text continues at position n_voice.
        -- Every chunk starts from it afresh (`copy_state=True`). A caller's own saved state
        -- (`inputs.voice_kv`, per layer K then V, `[n, 1024]` each) replaces the built-in one.
        local _n_voice = loom.seed_kv('lm', inputs.voice_kv or 'voice.kv')
        if _n_voice + _n_text >= LM_MAX_POSITIONS then
            error(string.format('pocket-tts: a %d-row voice and %d text tokens leave no room in a '
                                .. '%d-position cache', _n_voice, _n_text, LM_MAX_POSITIONS))
        end
        loom.run_subgraph_and_retain('text_embed', {n_tokens = _n_text, n_past = 0}, {text_ids = _text})
        loom.run_subgraph_and_retain('lm', {n_tokens = _n_text, n_past = _n_voice},
            {inputs_embeds = {from = 'text_embed'}, position_ids = loom.range(_n_voice, _n_text),
             attention_mask = loom.causal_mask(_n_text, _n_voice)})
        local _n_past = _n_voice + _n_text

        -- `TTSModel._estimate_max_gen_len`, and the tail the reference keeps after the EOS head
        -- fires: `prepare_text_prompt`'s guess for the chunk, plus the 2 `generate_audio_stream` adds.
        local _budget = inputs.max_frames or
            math.ceil((_n_text / TOKENS_PER_SECOND_ESTIMATE + GEN_SECONDS_PADDING) * FRAME_RATE)
        _budget = math.min(_budget, LM_MAX_POSITIONS - _n_past)
        local _after = inputs.frames_after_eos or
            ((_chunks[_c].short and SHORT_CHUNK_FRAMES_AFTER_EOS or LONG_CHUNK_FRAMES_AFTER_EOS)
             + FRAMES_AFTER_EOS_PADDING)

        -- Step 0's input is the checkpoint's `bos_emb` (the reference feeds NaN and swaps it in).
        local _latent = loom.get_weight('step_embed', 'bos_emb')
        local _latents = {}
        local _n_frames = 0
        local _eos_step = nil
        for _step = 0, _budget - 1 do
            loom.run_subgraph_and_retain('step_embed', {n_tokens = 1, n_past = 0}, {latent = _latent})
            loom.run_subgraph_and_retain('lm', {n_tokens = 1, n_past = _n_past},
                {inputs_embeds = {from = 'step_embed'}, position_ids = {_n_past},
                 attention_mask = loom.causal_mask(1, _n_past)})
            _n_past = _n_past + 1

            -- One unit draw per step, EVERY step including the one EOS ends on, as the reference draws.
            local _x = {}
            if inputs.noise ~= nil then
                for _i = 1, LATENT_DIM do _x[_i] = inputs.noise[_draw * LATENT_DIM + _i] end
            else
                _x = loom.gaussian_array(LATENT_DIM)
            end
            _draw = _draw + 1
            -- `lsd_decode`: `n_steps` updates from s = i/n to t = (i+1)/n. The first scales the unit
            -- draw by sqrt(temperature); the graph does both products, in f32, as the reference does.
            local _scale = _std
            for _i = 0, _n_steps - 1 do
                loom.run_subgraph_and_retain('flow_head', {n_tokens = 1, n_past = 0},
                    {c = {from = 'lm', index = 1}, s = {_i / _n_steps}, t = {(_i + 1) / _n_steps},
                     x = _x, x_scale = {_scale}, n_steps = {_n_steps}})
                _x = loom.get_output('flow_head', 1)
                _scale = 1.0
            end

            -- `_autoregressive_generation`: the EOS head is ignored for the first frames, then the loop
            -- runs `_after` more steps and ends WITHOUT keeping the latent of the step it ends on.
            local _eos = loom.get_output('lm', 2)[1]
            if _eos_step == nil and _eos > _threshold and _step >= MIN_FRAMES_BEFORE_EOS then
                _eos_step = _step
            end
            if _eos_step ~= nil and _step >= _eos_step + _after then break end
            for _i = 1, LATENT_DIM do _latents[_n_frames * LATENT_DIM + _i] = _x[_i] end
            _n_frames = _n_frames + 1
            _kept = _kept + 1
            _latent = _x
            -- TEACHER FORCING: a caller's latents are fed back in place of the ones this loop draws,
            -- while every step still computes, emits and EOS-tests its own. The loop is a feedback
            -- system, so f32 rounding compounds along it and two correct implementations drift apart
            -- (3.8e-03 on the waveform by the end of a 6 s clip); fed the same history, each step is
            -- comparable on its own. It is also how a continuation would be prompted.
            local _forced = inputs.teacher_latents
            if _forced ~= nil and _forced[_kept * LATENT_DIM] ~= nil then
                _latent = {}
                for _i = 1, LATENT_DIM do _latent[_i] = _forced[(_kept - 1) * LATENT_DIM + _i] end
            end
        end
        if _n_frames == 0 then error('pocket-tts: the flow LM produced no frames') end

        -- Mimi, every frame of the chunk in one call from a fresh state: the reference's decoder
        -- thread starts afresh per chunk too, and its streamed calls compute what one call does.
        local _wave = loom.run_subgraph('mimi_decoder', {n_codes = _n_frames, n_past = 0},
            {latents = _latents, positions = loom.range(0, _n_frames * MIMI_UPSAMPLE)})
        local _base = #wave
        for _i = 1, #_wave do wave[_base + _i] = _wave[_i] end
    end
