    -- ===== Voxtral-4B-TTS: text -> frames of 37 codes at 12.5 Hz -> 24 kHz. =====
    --
    -- vllm-omni's path for a preset voice: `encode_speech_request`'s prompt, the voice's rows written over
    -- its [AUDIO] slots, then one frame per LM step until the semantic code is [END_AUDIO].
    -- `inputs.tokens` is the text's ids as `Tekkenizer.encode(text, bos=False, eos=False)` returns them.
    if inputs.seed then loom.seed_rng(inputs.seed) end
    local _n_text = #inputs.tokens
    if _n_text == 0 then error('voxtral-tts: no text ids') end

    -- THE VOICE: `[n, DIM]` rows, the file's own (`voice.default`) unless a voice file sets `voice`.
    local _voice = inputs.voice or loom.get_weight('embed_prompt', 'voice.default')
    if #_voice == 0 or #_voice % DIM ~= 0 then
        error(string.format('voxtral-tts: a voice is rows of %d values; this one holds %d', DIM, #_voice))
    end
    local _n_voice = #_voice / DIM

    -- THE PROMPT: [BOS] [BEGIN_AUDIO] [AUDIO] x n_voice [NEXT_AUDIO_TEXT] text [REPEAT_AUDIO_TEXT]
    -- [BEGIN_AUDIO] -- `InstructTokenizerV7.encode_speech_request`, whose ids these are.
    local _ids = {BOS, BEGIN_AUDIO}
    for _i = 1, _n_voice do _ids[#_ids + 1] = AUDIO end
    _ids[#_ids + 1] = NEXT_AUDIO_TEXT
    for _i = 1, _n_text do _ids[#_ids + 1] = inputs.tokens[_i] end
    _ids[#_ids + 1] = REPEAT_AUDIO_TEXT
    _ids[#_ids + 1] = BEGIN_AUDIO
    local _n = #_ids
    if _n >= LM_MAX_POSITIONS then
        error(string.format('voxtral-tts: a %d-row prompt leaves no room in a %d-position cache', _n,
                            LM_MAX_POSITIONS))
    end
    -- The voice's rows over its slots, zeros (masked out) elsewhere.
    local _rows, _mask = {}, {}
    for _i = 1, _n * DIM do _rows[_i] = 0 end
    for _i = 1, _n_voice * DIM do _rows[2 * DIM + _i] = _voice[_i] end
    for _i = 1, _n do _mask[_i] = (_i > 2 and _i <= 2 + _n_voice) and 1 or 0 end
    loom.run_subgraph_and_retain('embed_prompt', {n_tokens = _n, n_past = 0},
        {ids = _ids, voice = _rows, voice_mask = _mask})
    loom.run_subgraph_and_retain('lm', {n_tokens = _n, n_past = 0},
        {inputs_embeds = {from = 'embed_prompt'}, position_ids = loom.range(0, _n),
         attention_mask = loom.causal_mask(_n, 0)})
    local _n_past = _n

    -- THE LOOP. `inputs.noise` pins the flow head's draws (one `[36]` unit normal per frame, the
    -- [END_AUDIO] frame's included, as the reference draws one there); `inputs.teacher_codes` feeds the
    -- reference's frames back instead of this loop's own (Retro-055) and runs the teacher's frame count.
    local _cfg = inputs.cfg or DEFAULT_CFG
    local _teacher = inputs.teacher_codes
    local _budget = inputs.max_frames or MAX_FRAMES
    if _teacher ~= nil then _budget = #_teacher / N_CODEBOOKS end
    _budget = math.min(_budget, LM_MAX_POSITIONS - _n)
    local _codes, _n_frames, _n_audio = {}, 0, nil
    for _step = 0, _budget - 1 do
        local _x
        if inputs.noise ~= nil then
            _x = voxtral_slice(inputs.noise, _step * N_ACOUSTIC, (_step + 1) * N_ACOUSTIC)
            if #_x < N_ACOUSTIC then error('voxtral-tts: inputs.noise ran out at frame ' .. _step) end
        else
            _x = loom.gaussian_array(N_ACOUSTIC)
        end
        loom.run_subgraph_and_retain('acoustic', {n_tokens = 1, n_past = 0},
            {hidden = {from = 'lm'}, noise = _x, cfg = {_cfg}})
        -- The semantic code: the flow head's argmax with [EMPTY_AUDIO] and the padding rows masked, so
        -- over [END_AUDIO, 2 + 8192). Absolute, which is the code.
        local _semantic = loom.argmax_row_range('acoustic', 0, END_AUDIO, N_SEMANTIC)
        _codes[#_codes + 1] = _semantic
        -- `decode_one_frame`'s `should_decode`: an [END_AUDIO] frame's acoustic codes are
        -- [EMPTY_AUDIO] (+2), not the integration's. Nothing decodes them; they are what the reference
        -- returns, so `return_codes` compares whole frames.
        local _acoustic = loom.get_output('acoustic', 2)
        for _i = 1, N_ACOUSTIC do
            _codes[#_codes + 1] = (_semantic == END_AUDIO) and EMPTY_AUDIO_CODE or _acoustic[_i]
        end
        _n_frames = _n_frames + 1
        if _semantic == END_AUDIO and _n_audio == nil then
            _n_audio = _n_frames - 1
            if _teacher == nil then break end
        end
        if _step == _budget - 1 then break end
        local _fed = _teacher and voxtral_slice(_teacher, _step * N_CODEBOOKS, (_step + 1) * N_CODEBOOKS)
                     or voxtral_slice(_codes, (_n_frames - 1) * N_CODEBOOKS, _n_frames * N_CODEBOOKS)
        loom.run_subgraph_and_retain('embed_frame', {n_tokens = 1, n_past = 0}, {codes = _fed})
        loom.run_subgraph_and_retain('lm', {n_tokens = 1, n_past = _n_past},
            {inputs_embeds = {from = 'embed_frame'}, position_ids = {_n_past},
             attention_mask = loom.causal_mask(1, _n_past)})
        _n_past = _n_past + 1
    end

    -- `return_codes` hands back every frame computed, [END_AUDIO] frame included -- the loop's own
    -- state, which is what an oracle compares frame by frame. Otherwise the codec, on the frames before
    -- [END_AUDIO] (all of them when the budget ran out first).
    if inputs.return_codes then
        wave = _codes
    else
        wave = voxtral_decode(_codes, _n_audio or _n_frames, N_CODEBOOKS, inputs.codec_chunk or CODEC_CHUNK,
                              CODEC_CONTEXT, SAMPLES_PER_FRAME)
    end
