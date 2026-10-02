    -- ===== Kyutai STT: Mimi codes for the whole clip, chunk by chunk; then one LM step per frame. =====
    --
    -- Inputs: `waveform` (24 kHz) and `length` (its real sample count, `{n}`). Returns the transcript's
    -- text ids, `<unk>` and `<pad>` dropped as moshi's `run_inference` drops them -- or, with
    -- `return_codes`, the Mimi codes alone.

    -- `stt_config`'s padding: silence before (none in this release) and the text delay plus a second
    -- after, so the transcript can finish; then whole 1920-sample frames only, as moshi keeps.
    local _real = (inputs.length and inputs.length[1]) or #inputs.waveform
    local _n = math.floor((PAD_LEFT + _real + PAD_RIGHT) / FRAME_SAMPLES)
    if _n < 1 then error('kyutai-stt: the clip is shorter than one frame') end
    local function _sample(i)       -- i: 0-based index into the padded signal
        local j = i - PAD_LEFT
        if j < 0 or j >= _real then return 0.0 end
        return inputs.waveform[j + 1]
    end

    -- MIMI, in chunks of CHUNK_FRAMES with CONTEXT_FRAMES of left context whose codes are discarded.
    -- The context covers the encoder's whole receptive field (8 transformer layers of a 249/250-wide
    -- window, plus the convolutions), so every kept frame is what moshi's streamed encoder produces;
    -- positions stay absolute across chunks.
    local _codes = {}               -- frame-major: _codes[f * N_Q + q + 1], f and q 0-based
    for _s = 0, _n - 1, CHUNK_FRAMES do
        local _c0 = math.max(0, _s - CONTEXT_FRAMES)
        local _e = math.min(_n, _s + CHUNK_FRAMES)
        local _m = _e - _c0
        local _wave = {}
        for _i = 0, _m * FRAME_SAMPLES - 1 do _wave[_i + 1] = _sample(_c0 * FRAME_SAMPLES + _i) end
        loom.run_subgraph_and_retain('mimi_encode', {n_codes = _m, n_past = 0},
            {waveform = _wave, positions = loom.range(2 * _c0, 2 * _m)})

        -- The split RVQ, one stage per call: the semantic codebook on its own projection, then the
        -- 31 acoustic ones chained on theirs. `subtract = 0` makes a chain's first call skip the
        -- subtraction it has nothing for (`qwen3_tts_export._RvqStepWrapper`).
        -- Sized to THIS chunk: the last one is usually shorter, and a graph input is checked by length.
        local _zeros = {}
        for _i = 1, _m do _zeros[_i] = 0 end
        local _drawn = {}
        loom.run_subgraph_and_retain('rvq_project_semantic', {n_codes = _m, n_past = 0},
            {rows = {from = 'mimi_encode'}})
        local _first = loom.range(0, CARD)
        loom.run_subgraph_and_retain('rvq_step', {n_codes = _m, n_past = 0},
            {rows = {from = 'rvq_project_semantic'}, prev_ids = _zeros, prev_codebook = _first,
             next_codebook = _first, subtract = {0.0}})
        _drawn[1] = loom.argmax_rows('rvq_step')
        loom.run_subgraph_and_retain('rvq_project_acoustic', {n_codes = _m, n_past = 0},
            {rows = {from = 'mimi_encode'}})
        local _rows = {from = 'rvq_project_acoustic'}
        local _prev = _zeros
        local _subtract = 0.0
        for _q = 1, N_Q - 1 do
            loom.run_subgraph_and_retain('rvq_step', {n_codes = _m, n_past = 0},
                {rows = _rows, prev_ids = _prev, prev_codebook = loom.range(math.max(_q - 1, 1) * CARD, CARD),
                 next_codebook = loom.range(_q * CARD, CARD), subtract = {_subtract}})
            _prev = loom.argmax_rows('rvq_step')
            _drawn[_q + 1] = _prev
            -- The residual this stage produced stays in the engine for the next one.
            _rows = {from = 'rvq_step', index = 2}
            _subtract = 1.0
        end
        for _f = _s, _e - 1 do
            local _row = _f - _c0 + 1
            for _q = 1, N_Q do _codes[_f * N_Q + _q] = _drawn[_q][_row] end
        end
    end

    -- `return_codes` stops here and hands back the codes, frame-major -- the tensor the gate grades
    -- against moshi's streamed Mimi before any id is drawn.
    if inputs.return_codes then return _codes end

    -- THE LM, one step per frame and one more. moshi's `run_inference` steps the first frame's codes
    -- twice: step 0 sees the initial tokens whatever it is handed, step k >= 1 sees step k-1's text id
    -- and frame k-1's codes. Its cache is a ring of LM_CONTEXT cells (loom.cpp ADR-066), so the mask
    -- spans at most that many and every cell is valid once it is full.
    ids = {}
    local _text = TEXT_INITIAL
    for _k = 0, _n do
        local _tokens = {_text}
        if _k == 0 then
            for _q = 1, N_Q do _tokens[_q + 1] = AUDIO_INITIAL end
        else
            for _q = 1, N_Q do _tokens[_q + 1] = _codes[(_k - 1) * N_Q + _q] end
        end
        loom.run_subgraph_and_retain('lm', {n_tokens = 1, n_past = _k},
            {tokens = _tokens, position_ids = {_k},
             attention_mask = loom.causal_mask(1, math.min(_k, LM_CONTEXT - 1))})
        _text = loom.argmax_row('lm', 0)
        if _k >= 1 and _text ~= TEXT_UNK and _text ~= TEXT_PAD then ids[#ids + 1] = _text end
    end
