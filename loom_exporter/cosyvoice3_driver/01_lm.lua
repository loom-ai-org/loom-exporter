    -- ===== The LM: text -> FSQ speech tokens, under `ras_sampling`. =====
    --
    -- `inputs.tokens` is the text's Qwen2 BPE ids, no framing. The prefill is `Qwen2LM.inference`'s
    -- `[sos, prompt text + text, task_id, prompt speech tokens]`, spelled as three parallel id arrays:
    -- a row is a SPEECH row (sos, task_id, the prompt's tokens -- `speech_embedding`) where
    -- `speech_mask` is 1 and a TEXT row (`embed_tokens`) where it is 0. The unused id is 0.
    --
    -- `inputs.speech_tokens` skips the LM: the flow and vocoder then run on the CALLER's tokens, which
    -- is how the gate checks those two phases on the reference's own tokens (VoxCPM2's
    -- `teacher_patches`). `inputs.return_tokens` returns the LM's raw draws, silence and all, instead of
    -- a waveform -- the exact arm of the sampled decode.
    local speech_tokens
    if inputs.speech_tokens then
        speech_tokens = inputs.speech_tokens
    else
        if inputs.seed then loom.seed_rng(inputs.seed) end
        local _prompt_text = cosyvoice3_voice(inputs.prompt_text, 'voice.prompt_text')
        local _prompt_speech = cosyvoice3_voice(inputs.prompt_speech_tokens, 'voice.prompt_speech_tokens')
        local _text_ids, _speech_ids, _mask = {0}, {SOS}, {1.0}
        local _has_eop = false
        local function _text_row(_id)
            _text_ids[#_text_ids + 1] = _id
            _speech_ids[#_speech_ids + 1] = 0
            _mask[#_mask + 1] = 0.0
            if _id == END_OF_PROMPT then _has_eop = true end
        end
        local function _speech_row(_id)
            _text_ids[#_text_ids + 1] = 0
            _speech_ids[#_speech_ids + 1] = _id
            _mask[#_mask + 1] = 1.0
        end
        for _i = 1, #_prompt_text do _text_row(_prompt_text[_i]) end
        for _i = 1, #inputs.tokens do _text_row(inputs.tokens[_i]) end
        if not _has_eop then
            -- `Qwen2LM.inference` asserts it; without it the LM has no boundary between instruction and text.
            error('cosyvoice3: no <|endofprompt|> (151646) in prompt text + text -- a voice\'s prompt text carries it')
        end
        _speech_row(TASK_ID)
        for _i = 1, #_prompt_speech do _speech_row(_prompt_speech[_i]) end
        local _prefill = #_text_ids

        -- The token budget is the TEXT's (not the prompt's): `min_len`/`max_len` in `Qwen2LM.inference`.
        local _n_text = #inputs.tokens
        local _min_len = math.floor(_n_text * MIN_TOKEN_TEXT_RATIO)
        local _max_len = math.min(math.floor(_n_text * (inputs.max_token_text_ratio or MAX_TOKEN_TEXT_RATIO)),
                                  LM_MAX_POSITIONS - _prefill)
        if _max_len <= 0 then
            error('cosyvoice3: the prefill (' .. _prefill .. ' rows) leaves no room in the ' ..
                  LM_MAX_POSITIONS .. '-position cache')
        end
        loom.run_subgraph_and_retain('lm', {n_tokens = _prefill, n_past = 0},
            {text_ids = _text_ids, speech_ids = _speech_ids, speech_mask = _mask,
             position_ids = loom.range(0, _prefill), attention_mask = loom.causal_mask(_prefill, 0)})

        -- `ras_sampling`: a nucleus draw (top-k 25, top-p 0.8 over the WHOLE softmax); if that id is
        -- among the last RAS_WIN drawn at least RAS_WIN * RAS_TAU times, a second draw from the full
        -- distribution with it banned. Before `min_len` tokens the id `speech_token_size` (6561) is banned
        -- from both draws -- `sampling_ids`' `ignore_eos`, which bans that one id and not the other 199
        -- stop ids. `inputs.draws` pins both uniforms per step (two per step, used or not), which is what
        -- makes the loop reproducible against a reference; absent, the engine's stream draws.
        local _draws = inputs.draws
        local _ras_threshold = RAS_WIN * RAS_TAU
        local _history = {}
        speech_tokens = {}
        local _silent_run = 0
        local _n_past = _prefill
        for _step = 0, _max_len - 1 do
            local _banned = {}
            if _step < _min_len then _banned[1] = SOS end
            local _u1, _u2
            if _draws then _u1, _u2 = _draws[2 * _step + 1], _draws[2 * _step + 2] end
            local _token = loom.sample_row('lm', -1, {temperature = 1.0, top_k = RAS_TOP_K, top_p = RAS_TOP_P,
                                                      top_p_mass = 'row', banned = _banned, uniform = _u1})
            local _rep = 0
            for _j = math.max(1, #_history - RAS_WIN + 1), #_history do
                if _history[_j] == _token then _rep = _rep + 1 end
            end
            if _rep >= _ras_threshold then
                _banned[#_banned + 1] = _token
                _token = loom.sample_row('lm', -1, {temperature = 1.0, banned = _banned, uniform = _u2})
            end
            if _token >= SOS then break end
            _history[#_history + 1] = _token
            -- `llm_job`'s silence cap is on what reaches the FLOW; the LM is fed every token back.
            if COSYVOICE3_SILENT[_token] then _silent_run = _silent_run + 1 else _silent_run = 0 end
            if _silent_run <= MAX_SILENT_RUN then speech_tokens[#speech_tokens + 1] = _token end
            loom.run_subgraph_and_retain('lm', {n_tokens = 1, n_past = _n_past},
                {text_ids = {0}, speech_ids = {_token}, speech_mask = {1.0},
                 position_ids = {_n_past}, attention_mask = loom.causal_mask(1, _n_past)})
            _n_past = _n_past + 1
        end
        if #speech_tokens == 0 then
            error('cosyvoice3: the LM produced no speech tokens (the first draw was a stop id)')
        end
        if inputs.return_tokens then return _history end
    end
