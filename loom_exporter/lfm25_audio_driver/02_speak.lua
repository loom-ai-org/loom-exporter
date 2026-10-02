    -- ===== LFM2.5-Audio, text to speech: `generate_sequential` with a "Perform TTS." voice prompt. =====
    --
    -- Inputs: `tokens` (the text's ids, no BOS), and optionally `voice_prompt` (a voice's system-prompt
    -- ids; a voice file carries one), `temperature` / `top_k` (the audio codes' sampler, the README's
    -- 0.8 / 64 by default; 0 is greedy), `seed`, `max_new_tokens`, `return_codes`. Returns the 24 kHz
    -- waveform (or, with `return_codes`, the frames' codes, frame-major).
    --
    -- The prompt is `ChatState`'s, every piece tokenized on its own, all text: so one prefill.
    if inputs.seed then loom.seed_rng(inputs.seed) end
    local _temperature = inputs.temperature or DEFAULT_TEMPERATURE
    local _top_k = inputs.top_k or DEFAULT_TOP_K
    -- The vocabulary's encode opens with `<|startoftext|>` (the tokenizer adds a BOS); `ChatState.add_text`
    -- does not, and the prompt already has one at its head -- so a leading BOS on the text is dropped.
    local _text = {}
    for _i = 1, #inputs.tokens do
        if not (_i == 1 and inputs.tokens[1] == BOS) then _text[#_text + 1] = inputs.tokens[_i] end
    end
    local _prompt = {}
    for _, part in ipairs({PROMPT_PRE, inputs.voice_prompt or DEFAULT_VOICE, PROMPT_MID, _text, PROMPT_TAIL}) do
        for _i = 1, #part do _prompt[#_prompt + 1] = part[_i] end
    end

    local _n_past, _last_n = 0, 0
    local function _feed(module, n)
        loom.run_subgraph_and_retain('decoder', {n_tokens = n, n_past = _n_past},
            {inputs_embeds = {from = module}, position_ids = loom.range(_n_past, n),
             attention_mask = loom.causal_mask(n, _n_past)})
        _n_past = _n_past + n
        _last_n = n
    end
    loom.run_subgraph_and_retain('embed', {n_tokens = #_prompt, n_past = 0}, {tokens = _prompt})
    _feed('embed', #_prompt)

    -- TEXT until `<|audio_start|>`, then one FRAME per step until a frame opens with end-of-audio, then
    -- text again until `<|im_end|>`. Text is greedy (`text_temperature=None`); each frame's 8 codes are
    -- drawn by the depthformer, row j conditioned on the codes before it.
    local _audio = false
    local _codes, _n_frames = {}, 0
    for _step = 1, inputs.max_new_tokens or MAX_NEW_TOKENS do
        if _n_past >= MAX_SEQ_LEN then error('lfm2.5-audio: the speech outgrew the KV cache') end
        local _hidden = {from = 'decoder', row = _last_n - 1, rows = 1}
        if not _audio then
            loom.run_subgraph_and_retain('lm_head', {n_tokens = 1, n_past = 0}, {hidden = _hidden})
            local _id = loom.argmax_row('lm_head', 0)
            if _id == END_OF_TURN then break end
            loom.run_subgraph_and_retain('embed', {n_tokens = 1, n_past = 0}, {tokens = {_id}})
            _feed('embed', 1)
            if _id == AUDIO_START then _audio = true end
        else
            local _prev, _frame = {}, {}
            for _j = 1, N_CODEBOOKS do _prev[_j] = 0 end
            for _j = 0, N_CODEBOOKS - 1 do
                loom.run_subgraph_and_retain('depth', {n_tokens = 1, n_past = 0}, {hidden = _hidden, prev = _prev})
                local _code = loom.sample_row('depth', _j, {temperature = _temperature, top_k = _top_k})
                _frame[_j + 1] = _code
                -- An end-of-audio first code makes the whole frame end-of-audio; its other codes would
                -- be overwritten, so they are not drawn.
                if _j == 0 and _code == END_OF_AUDIO then break end
                if _j + 1 < N_CODEBOOKS then _prev[_j + 2] = _code end
            end
            if _frame[1] == END_OF_AUDIO then
                for _j = 1, N_CODEBOOKS do _frame[_j] = END_OF_AUDIO end
                _audio = false
            else
                for _j = 1, N_CODEBOOKS do _codes[_n_frames * N_CODEBOOKS + _j] = _frame[_j] end
                _n_frames = _n_frames + 1
            end
            loom.run_subgraph_and_retain('audio_embed', {n_tokens = 1, n_past = 0}, {codes = _frame})
            _feed('audio_embed', 1)
        end
    end
    if _n_frames == 0 then error('lfm2.5-audio: the model produced no audio frames') end
    if inputs.return_codes then return _codes end

    -- The frames, all at once: the detokenizer's window is causal and 30 positions wide, and one call
    -- computes what liquid-audio's one call does. Its `nearest-exact` x6 upsampling is each frame
    -- repeated, so every frame's codes go over DETOK_UPSAMPLE times and the graph reads them as rows.
    local _rows = {}
    for _f = 0, _n_frames - 1 do
        for _r = 1, DETOK_UPSAMPLE do
            for _j = 1, N_CODEBOOKS do _rows[#_rows + 1] = _codes[_f * N_CODEBOOKS + _j] end
        end
    end
    local _n = DETOK_UPSAMPLE * _n_frames
    wave = loom.run_subgraph('detokenizer', {n_codes = _n, n_past = 0},
        {codes = _rows, positions = loom.range(0, _n)})
