    local _n_enc = #inputs.tokens

    -- One decoder call's inputs; `_kv` names WHICH cross_kv module's retained K/V it reads (output
    -- `2*layer + 1` is that layer's K, `+ 2` its V -- `cross_kv_input_names`' interleaving).
    local function _decoder_inputs(_column, _n_past, _kv)
        local _in = {
            codes = _column,
            position_ids = loom.range(_n_past, 1),
            attention_mask = loom.causal_mask(1, _n_past),
        }
        for _l = 0, N_LAYERS - 1 do
            _in["xk_" .. _l] = {from = _kv, index = 2 * _l + 1}
            _in["xv_" .. _l] = {from = _kv, index = 2 * _l + 2}
        end
        return _in
    end

    if inputs.seed then loom.seed_rng(inputs.seed) end
    local _temperature = inputs.temperature or TEMPERATURE
    local _top_k = inputs.top_k or TOP_K
    local _top_p = inputs.top_p or TOP_P
    -- `ClassifierFreeGuidanceLogitsProcessor` is `uncond + g * (cond - uncond)`, which is the form
    -- `loom.sample_row` implements, so the checkpoint's number is passed as it is (Dia's needed a + 1).
    -- `g <= 1` is off, `transformers`' own condition for not installing the processor.
    local _guidance = inputs.guidance_scale or GUIDANCE_SCALE
    local _use_cfg = _guidance > 1.0

    -- **The unconditional K/V are ZERO, so the unconditional stream gets a zero input, ONE frame wide.**
    -- `generate()` zeroes the encoder output and its mask, and `forward` multiplies the projected
    -- states by that mask, so every unconditional key and value is 0 and its cross-attention returns 0
    -- over any number of frames: a softmax over equal scores averages identical zero rows. One frame is
    -- the same answer for a 1024-float input instead of `_n_enc` of them.
    if _use_cfg then
        local _zeros = {}
        for _i = 1, HIDDEN do _zeros[_i] = 0 end
        loom.run_subgraph_and_retain('cross_kv_uncond', {n_enc_frames = 1, n_past = 0}, {xa = _zeros})
    end

    local _sample_opts = {temperature = _temperature, top_k = _top_k, top_p = _top_p}
    if _use_cfg then _sample_opts.guidance = {module = 'decoder_uncond', scale = _guidance} end

    -- **`max_new_tokens` counts AUDIO FRAMES**, the meaning loom-py's `text2codes` door and
    -- `loom_cli --n-predict` give it for every codec LM, not `transformers`' decoder steps. N frames
    -- take N + N_CODEBOOKS - 1 steps, because the last codebook trails the first by that many; so
    -- `transformers`' `max_new_tokens = M` is N = M + 1 - N_CODEBOOKS here. The default is the
    -- checkpoint's own `max_length`, converted the same way.
    --
    -- The pattern is `build_delay_pattern_mask`'s at max_length L = N + N_CODEBOOKS (`generate()`
    -- counts the start column): column j's codebook k is generated only when k < j < L - N_CODEBOOKS
    -- + 1 + k, and every other cell is PAD -- the start column, the leading triangle where codebook k
    -- has not begun, and the trailing one where it has finished. Below N_CODEBOOKS - 1 frames HF
    -- skips the pattern entirely and generates undelayed codebooks, which is not the layout the model
    -- was trained on; this driver keeps the pattern at every length.
    local _frames = inputs.max_new_tokens or (MAX_LENGTH - N_CODEBOOKS)
    if _frames < 1 then
        error("musicgen driver: max_new_tokens counts audio frames and must be at least 1, got " .. _frames)
    end
    local _length = _frames + N_CODEBOOKS
    -- Step n reads column n at cache position n, so the last step writes cell L - 2.
    if _length - 1 > MAX_CODES then
        error("musicgen driver: " .. _frames .. " frames need " .. (_length - 1) .. " decoder steps, "
              .. "past the decoder's " .. MAX_CODES .. "-position KV cache")
    end
    local _tail = _length - N_CODEBOOKS + 1

    -- The delayed columns, flat: column j's codebook k (0-based) at index j * N_CODEBOOKS + k + 1.
    local _cols = {}
    local _column = {}
    for _k = 1, N_CODEBOOKS do
        _cols[_k] = PAD
        _column[_k] = PAD
    end

    for _j = 1, _length - 1 do
        local _n_past = _j - 1
        local _axes = {n_tokens = 1, n_past = _n_past, n_enc_frames = _n_enc}
        loom.run_subgraph_and_retain('decoder', _axes, _decoder_inputs(_column, _n_past, 'cross_kv'))
        if _use_cfg then
            loom.run_subgraph_and_retain('decoder_uncond', {n_tokens = 1, n_past = _n_past,
                                                            n_enc_frames = 1},
                                         _decoder_inputs(_column, _n_past, 'cross_kv_uncond'))
        end
        for _k = 0, N_CODEBOOKS - 1 do
            local _v = PAD
            -- A forced cell is not drawn at all. `transformers` draws and then overwrites it, which
            -- only moves its random stream; the ids fed back are the same.
            if _k < _j and _j < _tail + _k then
                _v = loom.sample_row('decoder', _k, _sample_opts)
            end
            _cols[_j * N_CODEBOOKS + _k + 1] = _v
            _column[_k + 1] = _v
        end
    end

    -- Undo the delay: frame t's codebook k is column t + k, for t = 1 .. L - N_CODEBOOKS, which is
    -- exactly the set of non-PAD cells `generate()` keeps when it filters `!= pad_token_id`.
    local _codes = {}
    for _t = 1, _length - N_CODEBOOKS do
        for _k = 0, N_CODEBOOKS - 1 do
            _codes[#_codes + 1] = _cols[(_t + _k) * N_CODEBOOKS + _k + 1]
        end
    end
