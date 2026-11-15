-- ===== VoxCPM2, live (loom.cpp ADR-111): a few patches per `stream_feed`, decoded as they are drawn. =====
--
-- Upstream's `generate_streaming`: the same loop as one pass, each patch handed to the AudioVAE's
-- `streaming_decode()` as soon as the DiT has drawn it, with every causal conv's last inputs and every
-- transposed conv's overlap carried to the next call. Here that decoder is `vae_step`, its state the
-- stream's `vae` tensors on the device. How many patches a feed draws is `inputs.piece_patches`:
--
--   * absent, or -1: the growing default -- 1 patch (160 ms of audio), then 2, 4, and 8 from then on, so
--     the first audio is ready after one patch and the rest pays little for the cut.
--   * 1 .. MAX_STEP_PATCHES: that many per feed.
--   * 0: the whole clip in one feed, decoded by `vae_decode` in one call.
--
-- The pieces joined are the one-pass waveform to within f32 rounding: the loop is the one `infer` runs
-- (01_generate.lua), call for call, and `vae_step` sums the decoder's terms in another order than
-- `vae_decode` does. Upstream's streaming mode never retries a badcase, and neither does this; `infer`
-- still does.
--
-- Between calls everything is the stream's state: the ids, the knobs, a caller's pinned draws and teacher
-- patches, the loop's counters and the patch the next step is conditioned on (`cond`). The two LMs'
-- history is the model's KV cache, leased or owned (ADR-099). No graph output is read across a call: a
-- step that continues the loop re-runs the LMs on the patch it was handed before the DiT reads them, so
-- another stream (or a one-pass call) between two feeds moves nothing.

-- Kept only when given: a state value is numbers, and one never set reads back as nil.
local function voxcpm_keep(name, value)
    if value ~= nil then loom.set_state(name, value) end
end

-- What the loop changes, read at the start of a feed and written at its end. `n_past` 0 is "not prefilled".
local VOXCPM_LOOP = {'n_past', 'n_patches', 'draw', 'ended', 'vae_started', 'cond'}
-- What it only reads, set once by `voxcpm_begin`.
local VOXCPM_KNOBS = {'ids', 'cfg', 'steps', 'zero_init', 'schedule', 'budget', 'noise', 'forced'}

local function voxcpm_load()
    local st = {}
    for _, name in ipairs(VOXCPM_KNOBS) do st[name] = loom.state(name) end
    for _, name in ipairs(VOXCPM_LOOP) do st[name] = loom.state(name) end
    return st
end

local function voxcpm_save(st)
    for _, name in ipairs(VOXCPM_LOOP) do voxcpm_keep(name, st[name]) end
end

-- A generation from the top: the ids and knobs as `VoxCPM2Model._inference` takes them in zero-shot mode,
-- written to the stream's state, and the loop's own table returned. Draws nothing and runs no graph.
function voxcpm_begin(inputs)
    -- `inputs.tokens` is the text's ids as `loom::VoxCpmVocab::encode` returns them (no BOS);
    -- `<|audio_start|>` is appended here, as the reference appends it after tokenizing.
    local n_text = #inputs.tokens
    if n_text == 0 then error('voxcpm2: no text ids') end
    local ids = {}
    for i = 1, n_text do ids[i] = inputs.tokens[i] end
    ids[n_text + 1] = AUDIO_START_TOKEN
    local n = n_text + 1
    local steps = inputs.n_timesteps or DEFAULT_TIMESTEPS
    -- `_generate`'s budget: `ratio * len(target ids) + 10` patches, and the cache's room.
    local budget = inputs.max_frames or math.min(math.floor(n_text * BADCASE_RATIO + 10), MAX_LEN)
    budget = math.min(budget, LM_MAX_POSITIONS - n)
    if budget < 1 then error(string.format('voxcpm2: %d ids leave no room in a %d-position cache', n,
                                           LM_MAX_POSITIONS)) end
    local cond = {}
    for i = 1, PATCH_SIZE * LATENT_DIM do cond[i] = 0 end
    local st = {
        ids = ids,
        cfg = inputs.cfg or DEFAULT_CFG,
        steps = steps,
        schedule = (steps == DEFAULT_TIMESTEPS) and loom.get_weight('dit_step', 'euler_schedule')
                   or voxcpm_schedule(steps, SWAY_COEF),
        -- CFG-Zero*'s zero-init: the first steps' velocity is ZERO, so `x - dt * 0` leaves the draw as it
        -- is and the step costs nothing. Skipped, not run.
        zero_init = math.max(1, math.floor((steps + 1) * ZERO_INIT_FRACTION)),
        budget = budget,
        -- A caller's pinned draws (one PATCH-major `[4][64]` block per step taken, in the order the
        -- reference draws them) and teacher patches (one per patch kept), which feed the reference's
        -- history back in place of this loop's own (loom.cpp Retro-055): the loop is a feedback system,
        -- and two correct implementations drift apart along it.
        noise = inputs.noise,
        forced = inputs.teacher_patches,
        -- `prefix_feat_cond` starts as the prompt's last patch, which is zeros here.
        n_past = 0, n_patches = 0, draw = 0, ended = 0, vae_started = 0, cond = cond,
    }
    for _, name in ipairs(VOXCPM_KNOBS) do voxcpm_keep(name, st[name]) end
    voxcpm_save(st)
    return st
end

-- One step of the loop: the LMs brought up to the patch `st.cond` holds (the prefill on the first step),
-- one patch drawn and integrated by the guided DiT, and the stop decision. Returns the patch -- `[4][64]`,
-- frame-major -- or nil once the loop has ended. The step that ends it keeps its patch, as the
-- reference's does.
function voxcpm_step(st)
    if st.ended ~= 0 then return nil end
    local patch = PATCH_SIZE * LATENT_DIM
    if st.n_past == 0 then
        -- The prefill: the text's rows, with every feature patch zero (the prompt has no audio), which the
        -- feature encoder still encodes because the reference does -- `audio_mask` then zeroes it.
        local ids = st.ids
        local n = #ids
        local zeros, ones, none = {}, {}, {}
        for i = 1, n * patch do zeros[i] = 0 end
        for i = 1, n do ones[i], none[i] = 1, 0 end
        loom.run_subgraph_and_retain('feat_encode', {n_tokens = n, n_past = 0}, {patches = zeros})
        loom.run_subgraph_and_retain('base_lm', {n_tokens = n, n_past = 0},
            {text_ids = ids, feat_embed = {from = 'feat_encode'}, text_mask = ones, audio_mask = none,
             position_ids = loom.range(0, n), attention_mask = loom.causal_mask(n, 0)})
        loom.run_subgraph_and_retain('residual_lm', {n_tokens = n, n_past = 0},
            {enc = {from = 'base_lm', index = 1}, feat_embed = {from = 'feat_encode'}, audio_mask = none,
             attention_mask = loom.causal_mask(n, 0)})
        st.n_past = n
    else
        -- The previous step's patch (or the teacher's) as the next row of both LMs.
        local n_past = st.n_past
        loom.run_subgraph_and_retain('feat_encode', {n_tokens = 1, n_past = 0}, {patches = st.cond})
        loom.run_subgraph_and_retain('base_lm', {n_tokens = 1, n_past = n_past},
            {text_ids = {0}, feat_embed = {from = 'feat_encode'}, text_mask = {0}, audio_mask = {1},
             position_ids = {n_past}, attention_mask = loom.causal_mask(1, n_past)})
        loom.run_subgraph_and_retain('residual_lm', {n_tokens = 1, n_past = n_past},
            {enc = {from = 'base_lm', index = 1}, feat_embed = {from = 'feat_encode'}, audio_mask = {1},
             attention_mask = loom.causal_mask(1, n_past)})
        st.n_past = n_past + 1
    end

    -- One patch: a unit draw, integrated from t = 1 to 0 by the guided DiT.
    local x
    if st.noise ~= nil then
        x = {}
        for i = 1, patch do x[i] = st.noise[st.draw * patch + i] end
        if x[patch] == nil then error('voxcpm2: inputs.noise ran out at step ' .. st.n_patches) end
    else
        x = loom.gaussian_array(patch)
    end
    st.draw = st.draw + 1
    local schedule = st.schedule
    for k = st.zero_init + 1, st.steps do
        loom.run_subgraph_and_retain('dit_step', {n_tokens = 1, n_past = 0},
            {lm_hidden = {from = 'base_lm', index = 2}, residual_hidden = {from = 'residual_lm'},
             cond = st.cond, x = x, t = {schedule[2 * k - 1]}, dt = {schedule[2 * k]}, cfg = {st.cfg}})
        x = loom.get_output('dit_step', 1)
    end
    local step = st.n_patches
    st.n_patches = step + 1

    -- What the NEXT step is conditioned on: this patch, or the teacher's.
    st.cond = x
    local forced = st.forced
    if forced ~= nil and forced[st.n_patches * patch] ~= nil then
        st.cond = {}
        for i = 1, patch do st.cond[i] = forced[step * patch + i] end
    end

    -- The stop head on the row this patch was generated from; the reference's argmax, so a tie is "go
    -- on". Ignored until the patch after `min_len`.
    local stop = loom.get_output('base_lm', 3)
    if (step > MIN_LEN and stop[2] > stop[1]) or step == st.budget - 1 then st.ended = 1 end
    return x
end

-- `n` patches through `vae_step`, the decoder's state read from and written back to the stream's `vae`
-- tensors. The first call starts from `vae_init`'s fresh state, whose outputs are the states alone;
-- `vae_step`'s are the samples, then the states.
local function voxcpm_decode(st, latents, n)
    local inputs = {latents = latents}
    local shift = 1
    if st.vae_started == 0 then
        loom.run_subgraph_and_retain('vae_init', {n_tokens = 0, n_past = 0}, {zero = {0}})
        loom.keep_state('vae', 'vae_init')
        st.vae_started = 1
        shift = 0
    end
    for k = 1, VAE_STATES do inputs['state_' .. k] = {state = 'vae', index = k + shift} end
    loom.run_subgraph_and_retain('vae_step', {n_codes = PATCH_SIZE * n, n_tokens = 0, n_past = 0}, inputs)
    loom.keep_state('vae', 'vae_step')
    return loom.get_output('vae_step', 1)
end

function stream_open(inputs)
    if inputs.seed then loom.seed_rng(inputs.seed) end
    local piece_patches = inputs.piece_patches or -1
    if piece_patches < -1 or piece_patches > MAX_STEP_PATCHES or piece_patches ~= math.floor(piece_patches) then
        error(string.format('voxcpm2: piece_patches is a patch count up to %d, 0 for the whole clip, or -1 '
                            .. 'for the growing default', MAX_STEP_PATCHES))
    end
    loom.set_state('piece_patches', piece_patches)
    voxcpm_begin(inputs)
    loom.set_state('piece', 0)
    loom.set_state('n_pieces', piece_patches == 0 and 1 or 0)
    loom.set_state('done', 0)
    return {}
end

function stream_feed(inputs)
    if loom.state('done') ~= 0 then return {} end
    local st = voxcpm_load()
    local piece_patches = loom.state('piece_patches')
    local want = piece_patches
    if want < 0 then want = math.min(PIECE_PATCHES_GROWN, 2 ^ loom.state('piece')) end
    if want == 0 then want = st.budget end
    local patch = PATCH_SIZE * LATENT_DIM
    local latents, n = {}, 0
    while n < want and st.ended == 0 do
        local x = voxcpm_step(st)
        for i = 1, patch do latents[n * patch + i] = x[i] end
        n = n + 1
    end
    local wave = {}
    if piece_patches == 0 then
        wave = loom.run_subgraph('vae_decode', {n_codes = PATCH_SIZE * n, n_past = 0}, {latents = latents})
    elseif n > 0 then
        wave = voxcpm_decode(st, latents, n)
    end
    if st.ended ~= 0 then loom.set_state('done', 1) end
    voxcpm_save(st)
    loom.set_state('piece', loom.state('piece') + 1)
    return wave
end

function stream_finish(inputs)
    return {}
end
