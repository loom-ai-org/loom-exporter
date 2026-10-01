    -- ===== SpeechT5: text -> encoder -> an autoregressive loop over mel frames -> HiFi-GAN. =====
    --
    -- `inputs.tokens` are the SentencePiece character ids with `</s>` appended. Optional knobs, with
    -- `_generate_speech`'s defaults: `threshold`, `minlenratio`, `maxlenratio`; `speaker` (a 512-float
    -- x-vector, un-normalised) replaces the built-in voice; `seed` seeds the prenet's dropout draws, and
    -- `masks` pins them outright. `teacher_frames` (one `[80]` row per frame, row-major) feeds a known
    -- spectrogram back in place of the loop's own frames, which is how a gate compares steps without
    -- the feedback compounding rounding.
    local _ids = inputs.tokens
    local _n = #_ids
    if _n == 0 then error('speecht5: no text ids') end
    if _n > MAX_TEXT_POSITIONS then
        error(string.format('speecht5: %d ids exceed the encoder\'s %d positions', _n, MAX_TEXT_POSITIONS))
    end
    if inputs.seed then loom.seed_rng(inputs.seed) end

    loom.run_subgraph_and_retain('encoder', {n_tokens = _n, n_past = 0},
        {input_ids = _ids, position_ids = loom.range(0, _n),
         rel_index = speecht5_relative_index(_n, MAX_RELATIVE_POSITION)})
    loom.run_subgraph_and_retain('cross_kv', {n_enc_frames = _n, n_past = 0}, {xa = {from = 'encoder'}})

    local _speaker = inputs.speaker or loom.get_weight('decoder', 'speaker')
    local _threshold = inputs.threshold or DEFAULT_THRESHOLD
    -- `int(n * ratio / reduction_factor)`, and never past the decoder's position table: the reference
    -- would fail there on a shape, one step in.
    local _maxlen = math.floor(_n * (inputs.maxlenratio or DEFAULT_MAXLENRATIO) / REDUCTION_FACTOR)
    local _minlen = math.floor(_n * (inputs.minlenratio or DEFAULT_MINLENRATIO) / REDUCTION_FACTOR)
    if _maxlen > MAX_SPEECH_POSITIONS then _maxlen = MAX_SPEECH_POSITIONS end
    if _maxlen < 1 then _maxlen = 1 end

    local _frame = {}
    for _i = 1, NUM_MEL_BINS do _frame[_i] = 0.0 end
    local _frames = {}
    local _n_frames = 0
    local _step_len = REDUCTION_FACTOR * NUM_MEL_BINS
    local _teacher = inputs.teacher_frames
    for _step = 0, _maxlen - 1 do
        local _call = {
            frame = _frame, position_ids = {_step},
            prenet_mask_0 = speecht5_draw_mask(PRENET_UNITS, inputs.masks, 2 * _step),
            prenet_mask_1 = speecht5_draw_mask(PRENET_UNITS, inputs.masks, 2 * _step + 1),
            speaker = _speaker, attention_mask = loom.causal_mask(1, _step),
        }
        for _l = 0, N_LAYERS - 1 do
            _call['xk_' .. _l] = {from = 'cross_kv', index = 2 * _l + 1}
            _call['xv_' .. _l] = {from = 'cross_kv', index = 2 * _l + 2}
        end
        loom.run_subgraph_and_retain('decoder', {n_tokens = 1, n_past = _step, n_enc_frames = _n}, _call)
        local _spectrum = loom.get_output('decoder', 1)
        local _logits = loom.get_output('decoder', 2)
        local _base = _n_frames * NUM_MEL_BINS
        for _i = 1, _step_len do _frames[_base + _i] = _spectrum[_i] end
        _n_frames = _n_frames + REDUCTION_FACTOR
        -- The next input is the LAST frame of the pair, or the caller's frame at the same place.
        _frame = {}
        local _src, _off = _spectrum, _step_len - NUM_MEL_BINS
        if _teacher ~= nil and _teacher[_n_frames * NUM_MEL_BINS] ~= nil then
            _src, _off = _teacher, (_n_frames - 1) * NUM_MEL_BINS
        end
        for _i = 1, NUM_MEL_BINS do _frame[_i] = _src[_off + _i] end
        -- `sum(sigmoid(prob_out)) >= threshold`, tested from step `minlen` on; at `maxlen` it stops
        -- regardless. The step it stops on keeps its frames.
        local _p = 0.0
        for _r = 1, REDUCTION_FACTOR do _p = _p + 1.0 / (1.0 + math.exp(-_logits[_r])) end
        local _idx = _step + 1
        if _idx >= _minlen and (_idx >= _maxlen or _p >= _threshold) then break end
    end

    loom.run_subgraph_and_retain('postnet', {n_enc_frames = _n_frames, n_past = 0}, {spectrogram = _frames})
    wave = loom.run_subgraph('vocoder', {n_enc_frames = _n_frames, n_past = 0}, {mel = {from = 'postnet'}})
