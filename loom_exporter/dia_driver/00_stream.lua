-- ===== Dia, live (loom.cpp ADR-102): one decoder row per `stream_feed`, a frame back once it is whole. =====
--
-- The one-pass loop cut at each row. `stream_open` runs the encoder and projects the cross-attention K/V --
-- the conditional pair and, under guidance, the unconditional one -- into the stream's own state, since
-- every step reads them and another stream's call would overwrite a retained output. Each `stream_feed`
-- runs one decoder row (both streams under guidance) and draws its nine ids, in the one-pass loop's order,
-- so `infer` -- this stream opened, fed to its end and finished in one call -- gives the codes it always gave.
--
-- **A frame is handed back once it is WHOLE.** Audio frame t's channel k is emitted at row t + delay[k], so
-- frame t is complete when row t + MAX_DELAY has been drawn: the first frame comes MAX_DELAY rows in, and
-- after channel 0's EOS the forced tail rows complete the last MAX_DELAY frames. The state keeps only the
-- last MAX_DELAY + 1 rows (`recent`), all a frame ever reads. `piece` counts rows and `done` is set when
-- the one-pass loop would have stopped.

local function dia_decoder_inputs(frame, n_past, kv)
    local args = {codes = frame, position_ids = loom.range(n_past, 1), attention_mask = loom.causal_mask(1, n_past)}
    for l = 0, N_LAYERS - 1 do
        args['xk_' .. l] = {state = kv, index = 2 * l + 1}
        args['xv_' .. l] = {state = kv, index = 2 * l + 2}
    end
    return args
end

-- `guidance.top_k` is NOT `top_k`: it is `DiaClassifierFreeGuidanceLogitsProcessor`'s `guidance_top_k`,
-- which uses the GUIDED logits to pick a shortlist and then draws from the CONDITIONAL ones restricted to
-- it. `transformers` passes `generation_config.top_k` into that slot, so the same number lands in both.
local function dia_sample_opts(lo, hi, greedy)
    local top_k = loom.state('top_k')
    local o = {lo = lo, hi = hi,
               temperature = greedy and 0.0 or loom.state('temperature'),
               top_k = greedy and 0 or top_k,
               top_p = greedy and 1.0 or loom.state('top_p')}
    if loom.state('use_cfg') == 1 then
        o.guidance = {module = 'decoder_uncond', scale = loom.state('guidance') + 1.0, top_k = top_k}
    end
    return o
end

function stream_open(inputs)
    -- One encoder frame per input byte, bound as an axis on every decoder call because the
    -- cross-attention K/V carry it as their own dynamic dimension.
    local n_enc = #inputs.tokens
    if n_enc == 0 then error('dia: no text ids') end
    -- **A sampling driver needs a seed, and this is the only place a caller can set one.**
    if inputs.seed then loom.seed_rng(inputs.seed) end
    loom.run_subgraph_and_retain('encoder', {n_tokens = n_enc, n_past = 0}, {tokens = inputs.tokens})
    loom.run_subgraph_and_retain('cross_kv', {n_enc_frames = n_enc, n_past = 0}, {xa = {from = 'encoder'}})
    loom.keep_state('cross_kv', 'cross_kv')

    -- **Classifier-free guidance, and the arithmetic of the scale is Dia's.** Its own processor combines
    -- as `cond + g * (cond - uncond)`, centred on the CONDITIONAL logits; `loom.sample_row` is centred on
    -- the unconditional ones, so the `+ 1` happens in `dia_sample_opts`. `g <= 1` means off, which is
    -- `transformers`' own condition for not installing the processor at all.
    local guidance = inputs.guidance_scale or GUIDANCE_SCALE
    local use_cfg = guidance > 1.0
    if use_cfg then
        -- The same encoder over an all-zero prompt of the same length (`DiaGenerationMixin`'s
        -- `torch.zeros_like`), projected into its own cross-attention K/V.
        local blank = {}
        for i = 1, n_enc do blank[i] = 0 end
        loom.run_subgraph_and_retain('encoder', {n_tokens = n_enc, n_past = 0}, {tokens = blank})
        loom.run_subgraph_and_retain('cross_kv_uncond', {n_enc_frames = n_enc, n_past = 0},
                                      {xa = {from = 'encoder'}})
        loom.keep_state('cross_kv_uncond', 'cross_kv_uncond')
    end

    loom.set_state('n_enc', n_enc)
    loom.set_state('temperature', inputs.temperature or TEMPERATURE)
    loom.set_state('top_k', inputs.top_k or TOP_K)
    loom.set_state('top_p', inputs.top_p or TOP_P)
    loom.set_state('guidance', guidance)
    loom.set_state('use_cfg', use_cfg and 1 or 0)
    -- **The ceiling is a count of AUDIO FRAMES, and reaching it FORCES an EOS rather than breaking**, as
    -- `DiaEOSDelayPatternLogitsProcessor`'s `reached_max_len` clause does: every frame handed back is
    -- complete on all nine channels. The default leaves MAX_DELAY rows of tail inside the position budget.
    loom.set_state('max_frames', inputs.max_new_tokens or (MAX_CODES - 1 - MAX_DELAY))
    -- Row 0 is the all-BOS frame the model conditions on.
    local bos = {}
    for c = 1, N_CHANNELS do bos[c] = BOS end
    loom.set_state('recent', bos)
    loom.set_state('row', 1)
    loom.set_state('eos_row', -1)
    loom.set_state('piece', 0)
    loom.set_state('done', 0)
    return {}
end

function stream_feed(inputs)
    if loom.state('done') ~= 0 then return {} end
    local row, eos_row = loom.state('row'), loom.state('eos_row')
    local recent = loom.state('recent')
    local n_past, n_enc = row - 1, loom.state('n_enc')
    -- The row fed back is the last one drawn: the tail of `recent`.
    local step = {}
    for c = 1, N_CHANNELS do step[c] = recent[#recent - N_CHANNELS + c] end

    local axes = {n_tokens = 1, n_past = n_past, n_enc_frames = n_enc}
    loom.run_subgraph_and_retain('decoder', axes, dia_decoder_inputs(step, n_past, 'cross_kv'))
    if loom.state('use_cfg') == 1 then
        -- The same codes and positions, its own KV cache (`kv_cache_scope: "private"`) and the
        -- unconditional cross-attention. Second, so the conditional module's output is what `sample_row` reads.
        loom.run_subgraph_and_retain('decoder_uncond', axes, dia_decoder_inputs(step, n_past, 'cross_kv_uncond'))
    end

    -- **A RESTRICTED draw per channel**: only channel 0 may say EOS, no channel PAD or BOS. Channel 0 takes
    -- two calls -- `DiaEOSChannelFilterLogitsProcessor`'s force-EOS-when-it-wins and suppress-it-otherwise:
    -- ask which of [0, EOS] wins greedily, and if it is EOS take it, else draw from [0, EOS).
    local nxt = {}
    if loom.sample_row('decoder', 0, dia_sample_opts(0, EOS + 1, true)) == EOS then
        nxt[1] = EOS
    else
        nxt[1] = loom.sample_row('decoder', 0, dia_sample_opts(0, EOS, false))
    end
    for c = 2, N_CHANNELS do nxt[c] = loom.sample_row('decoder', c - 1, dia_sample_opts(0, EOS, false)) end

    -- The delay scaffold: a channel that has not reached its own offset is fed BOS back.
    for c = 1, N_CHANNELS do
        if row <= DELAY_PATTERN[c] then nxt[c] = BOS end
    end
    -- The ceiling, then channel 0's EOS, then the forced tail: channel k says EOS exactly delay[k] rows after
    -- channel 0 and pads after that (`DiaEOSDelayPatternLogitsProcessor`).
    if eos_row < 0 and row > loom.state('max_frames') then nxt[1] = EOS end
    if eos_row < 0 and nxt[1] == EOS then eos_row = row end
    if eos_row >= 0 then
        local since = row - eos_row
        for c = 1, N_CHANNELS do
            if since > DELAY_PATTERN[c] then nxt[c] = PAD elseif since == DELAY_PATTERN[c] then nxt[c] = EOS end
        end
    end

    -- Keep the last MAX_DELAY + 1 rows: `recent`'s first row is row `row - MAX_DELAY` once it is full.
    for c = 1, N_CHANNELS do recent[#recent + 1] = nxt[c] end
    if #recent > (MAX_DELAY + 1) * N_CHANNELS then
        local trimmed = {}
        for i = N_CHANNELS + 1, #recent do trimmed[#trimmed + 1] = recent[i] end
        recent = trimmed
    end
    loom.set_state('recent', recent)
    loom.set_state('eos_row', eos_row)
    loom.set_state('row', row + 1)
    loom.set_state('piece', loom.state('piece') + 1)

    -- Frame t = row - MAX_DELAY is whole now: its channel k is row t + delay[k], all inside `recent`.
    -- Frames run 1 .. the one before channel 0's EOS, or, at the position backstop, as far as every
    -- channel reaches.
    local frame = {}
    local t = row - MAX_DELAY
    if t >= 1 and (eos_row < 0 or t <= eos_row - 1) then
        local first = row - (#recent / N_CHANNELS - 1)          -- the row number of `recent`'s first row
        for c = 1, N_CHANNELS do frame[c] = recent[(t + DELAY_PATTERN[c] - first) * N_CHANNELS + c] end
    end
    if (eos_row >= 0 and row + 1 > eos_row + MAX_DELAY) or row + 1 >= MAX_CODES then
        loom.set_state('done', 1)
    end
    return frame
end

function stream_finish(inputs)
    return {}
end
