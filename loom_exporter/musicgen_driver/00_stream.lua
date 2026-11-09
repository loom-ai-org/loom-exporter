-- ===== MusicGen, live (loom.cpp ADR-104): one decoder step per `stream_feed`, a frame back once it is whole. =====
--
-- The one-pass loop cut at each step, as Dia's is (ADR-103). `stream_open` runs the T5 encoder and projects
-- the cross-attention K/V -- and, under guidance, the unconditional pair -- into the stream's own state,
-- since every step reads them and another stream's call would overwrite a retained output. Each
-- `stream_feed` runs one decoder step (both streams under guidance) and draws its codebooks in the one-pass
-- loop's order, so `infer` -- this stream opened, fed to its end and finished in one call -- gives the codes
-- it always gave.
--
-- **A frame is handed back once it is WHOLE.** The delay pattern puts frame t's codebook k in column t + k,
-- so frame t is complete when column t + N_CODEBOOKS - 1 has been drawn: the first frame comes
-- N_CODEBOOKS - 1 steps in, and the last one with the last step. The state keeps only the last
-- N_CODEBOOKS columns (`recent`), all a frame ever reads. `piece` counts steps and `done` is set after the
-- last one.

local function musicgen_decoder_inputs(column, n_past, kv)
    local args = {codes = column, position_ids = loom.range(n_past, 1), attention_mask = loom.causal_mask(1, n_past)}
    -- Output `2*layer + 1` of the kept cross_kv is that layer's K, `+ 2` its V (`cross_kv_input_names`).
    for l = 0, N_LAYERS - 1 do
        args['xk_' .. l] = {state = kv, index = 2 * l + 1}
        args['xv_' .. l] = {state = kv, index = 2 * l + 2}
    end
    return args
end

function stream_open(inputs)
    local n_enc = #inputs.tokens
    if n_enc == 0 then error('musicgen: no text ids') end
    if inputs.seed then loom.seed_rng(inputs.seed) end
    -- T5 encoder + enc_to_dec_proj: one bidirectional pass over the prompt, then the cross-attention K/V
    -- for every decoder layer, once per prompt.
    loom.run_subgraph_and_retain('encoder', {n_tokens = n_enc, n_past = 0},
        {input_ids = inputs.tokens,
         position_bias = t5_position_bias(ENC_REL_BIAS, N_TEXT_HEAD, REL_BUCKETS, REL_MAX_DISTANCE, n_enc, 0, 1)})
    loom.run_subgraph_and_retain('cross_kv', {n_enc_frames = n_enc, n_past = 0}, {xa = {from = 'encoder'}})
    loom.keep_state('cross_kv', 'cross_kv')

    -- `ClassifierFreeGuidanceLogitsProcessor` is `uncond + g * (cond - uncond)`, which is the form
    -- `loom.sample_row` implements, so the checkpoint's number is passed as it is (Dia's needed a + 1).
    -- `g <= 1` is off, `transformers`' own condition for not installing the processor.
    local guidance = inputs.guidance_scale or GUIDANCE_SCALE
    local use_cfg = guidance > 1.0
    if use_cfg then
        -- **The unconditional K/V are ZERO, so the unconditional stream gets a zero input, ONE frame wide.**
        -- `generate()` zeroes the encoder output and its mask, and `forward` multiplies the projected
        -- states by that mask, so every unconditional key and value is 0 and its cross-attention returns
        -- 0 over any number of frames: a softmax over equal scores averages identical zero rows.
        local zeros = {}
        for i = 1, HIDDEN do zeros[i] = 0 end
        loom.run_subgraph_and_retain('cross_kv_uncond', {n_enc_frames = 1, n_past = 0}, {xa = zeros})
        loom.keep_state('cross_kv_uncond', 'cross_kv_uncond')
    end

    -- **`max_new_tokens` counts AUDIO FRAMES**, the meaning loom-py's `text2codes` door and `loom_cli
    -- --n-predict` give it for every codec LM, not `transformers`' decoder steps. N frames take
    -- N + N_CODEBOOKS - 1 steps, because the last codebook trails the first by that many; so
    -- `transformers`' `max_new_tokens = M` is N = M + 1 - N_CODEBOOKS here. The default is the
    -- checkpoint's own `max_length`, converted the same way.
    local frames = inputs.max_new_tokens or (MAX_LENGTH - N_CODEBOOKS)
    if frames < 1 then
        error('musicgen driver: max_new_tokens counts audio frames and must be at least 1, got ' .. frames)
    end
    local length = frames + N_CODEBOOKS
    -- Step n reads column n at cache position n, so the last step writes cell L - 2.
    if length - 1 > MAX_CODES then
        error('musicgen driver: ' .. frames .. ' frames need ' .. (length - 1) .. ' decoder steps, '
              .. 'past the decoder\'s ' .. MAX_CODES .. '-position KV cache')
    end

    loom.set_state('n_enc', n_enc)
    loom.set_state('temperature', inputs.temperature or TEMPERATURE)
    loom.set_state('top_k', inputs.top_k or TOP_K)
    loom.set_state('top_p', inputs.top_p or TOP_P)
    loom.set_state('guidance', guidance)
    loom.set_state('use_cfg', use_cfg and 1 or 0)
    loom.set_state('length', length)
    -- Column 0 is all PAD: the start column the model conditions on.
    local start = {}
    for k = 1, N_CODEBOOKS do start[k] = PAD end
    loom.set_state('recent', start)
    loom.set_state('step', 1)
    loom.set_state('piece', 0)
    loom.set_state('done', 0)
    return {}
end

function stream_feed(inputs)
    if loom.state('done') ~= 0 then return {} end
    local j, length = loom.state('step'), loom.state('length')
    local recent = loom.state('recent')
    local n_past = j - 1
    -- The column fed back is the last one drawn: the tail of `recent`.
    local column = {}
    for k = 1, N_CODEBOOKS do column[k] = recent[#recent - N_CODEBOOKS + k] end

    loom.run_subgraph_and_retain('decoder', {n_tokens = 1, n_past = n_past, n_enc_frames = loom.state('n_enc')},
                                 musicgen_decoder_inputs(column, n_past, 'cross_kv'))
    local opts = {temperature = loom.state('temperature'), top_k = loom.state('top_k'), top_p = loom.state('top_p')}
    if loom.state('use_cfg') == 1 then
        loom.run_subgraph_and_retain('decoder_uncond', {n_tokens = 1, n_past = n_past, n_enc_frames = 1},
                                     musicgen_decoder_inputs(column, n_past, 'cross_kv_uncond'))
        opts.guidance = {module = 'decoder_uncond', scale = loom.state('guidance')}
    end

    -- The pattern is `build_delay_pattern_mask`'s at max_length L (`generate()` counts the start column):
    -- column j's codebook k is generated only when k < j < L - N_CODEBOOKS + 1 + k, and every other cell is
    -- PAD -- the leading triangle where codebook k has not begun, and the trailing one where it has
    -- finished. A forced cell is not drawn at all: `transformers` draws and then overwrites it, which only
    -- moves its random stream; the ids fed back are the same.
    local tail = length - N_CODEBOOKS + 1
    for k = 0, N_CODEBOOKS - 1 do
        local v = PAD
        if k < j and j < tail + k then v = loom.sample_row('decoder', k, opts) end
        recent[#recent + 1] = v
    end
    -- Keep the last N_CODEBOOKS columns: `recent`'s first column is column j - N_CODEBOOKS + 1 once full.
    if #recent > N_CODEBOOKS * N_CODEBOOKS then
        local trimmed = {}
        for i = N_CODEBOOKS + 1, #recent do trimmed[#trimmed + 1] = recent[i] end
        recent = trimmed
    end
    loom.set_state('recent', recent)
    loom.set_state('step', j + 1)
    loom.set_state('piece', loom.state('piece') + 1)
    if j + 1 > length - 1 then loom.set_state('done', 1) end

    -- Frame t = j - N_CODEBOOKS + 1 is whole now: its codebook k is column t + k, all inside `recent`.
    -- Frames run 1 .. L - N_CODEBOOKS, exactly the non-PAD cells `generate()` keeps.
    local frame = {}
    local t = j - N_CODEBOOKS + 1
    if t >= 1 and t <= length - N_CODEBOOKS then
        local first = j - (#recent / N_CODEBOOKS - 1)          -- the column number of `recent`'s first column
        for k = 0, N_CODEBOOKS - 1 do frame[k + 1] = recent[(t + k - first) * N_CODEBOOKS + k + 1] end
    end
    return frame
end

function stream_finish(inputs)
    return {}
end
