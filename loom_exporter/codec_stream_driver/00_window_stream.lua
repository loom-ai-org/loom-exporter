-- ===== A codec whose decoder is NOT causal, live (loom.cpp ADR-102): windows with left and right context. =====
--
-- The one-pass `decode` graph run over windows: every push decodes the frames it has, with up to
-- `LEFT_FRAMES` already handed back before them, and hands back all but the last `LOOKAHEAD_FRAMES`, which
-- wait for the context after them; `stream_finish` decodes those. A frame's samples are kept only once
-- `LOOKAHEAD_FRAMES` frames follow it, so the stream runs that far behind the codes. Both margins cover the
-- decoder's receptive field (DAC 44 kHz: 16 frames each side is 4.1e-07 from one call, 8 is 5.9e-03; SNAC
-- 24 kHz: 3 coarse frames is 7e-16 at f64, 2 is 1.3e-03), so a window gives what one call gives, to f32
-- rounding.
--
-- **A stochastic decoder's noise belongs to its frame** (SNAC, ADR-104). `NOISE_PER_FRAME` is how many draws
-- each NoiseBlock takes per frame, in decoder order (empty for a deterministic codec). They are drawn FRAME-
-- MAJOR -- frame 0's draws for every block, then frame 1's -- as the frames arrive, from the stream's own
-- generator, and kept beside the frame's codes, so a frame decoded again in the next window sees the same
-- noise. `window_one_pass` draws the same way in one go, so a stream and one call with the same seed decode
-- the same waveform.
--
-- The functions hold nothing between calls: the codes not yet decoded (and the left context before them)
-- are a state value, with `start` the frame index of its first frame and `emitted` the frames handed back.

local function noise_per_frame()
    local total = 0
    for _, m in ipairs(NOISE_PER_FRAME) do total = total + m end
    return total
end

local function draw_noise(n_frames)
    if #NOISE_PER_FRAME == 0 or n_frames == 0 then return {} end
    return loom.gaussian_array(noise_per_frame() * n_frames)
end

local function bind_noise(args, noise, from, to)
    -- Frames [from, to) of frame-major `noise` (0-based, relative to its first frame), as the graph's
    -- block-major inputs: `noise_i` is block i's draws for each frame in turn.
    local total, offset = noise_per_frame(), 0
    for i, m in ipairs(NOISE_PER_FRAME) do
        local block = {}
        for f = from, to - 1 do
            local base = f * total + offset
            for j = 1, m do block[#block + 1] = noise[base + j] end
        end
        args['noise_' .. (i - 1)] = block
        offset = offset + m
    end
end

local function window_decode(pending, noise, start, from, to)
    -- Frames [from, to) of the stream, decoded with the context `pending` holds before them; returns the
    -- samples of frames [from, to).
    local first = math.max(start, from - LEFT_FRAMES)
    local codes = {}
    for i = (first - start) * CODES_PER_FRAME + 1, (to - start) * CODES_PER_FRAME do codes[#codes + 1] = pending[i] end
    local args = {codes = codes}
    if #NOISE_PER_FRAME > 0 then bind_noise(args, noise, first - start, to - start) end
    local wave = loom.run_subgraph('decode', {n_codes = to - first, n_past = 0}, args)
    local out = {}
    for i = (from - first) * SAMPLES_PER_FRAME + 1, (to - first) * SAMPLES_PER_FRAME do out[#out + 1] = wave[i] end
    return out
end

local function window_trim(pending, noise, start, emitted)
    -- Drop the frames no future window reaches back to.
    local keep_from = math.max(start, emitted - LEFT_FRAMES)
    if keep_from == start then return pending, noise, start end
    local kept, kept_noise = {}, {}
    for i = (keep_from - start) * CODES_PER_FRAME + 1, #pending do kept[#kept + 1] = pending[i] end
    for i = (keep_from - start) * noise_per_frame() + 1, #noise do kept_noise[#kept_noise + 1] = noise[i] end
    return kept, kept_noise, keep_from
end

function window_one_pass(inputs)
    -- `infer` for a stochastic codec: one call over every frame, its noise drawn frame-major as a stream
    -- draws it. A caller may hand in the graph's own `noise_i` arrays instead (an oracle's exact input).
    local codes = inputs.codes or inputs.tokens
    local n = #codes / CODES_PER_FRAME
    local args = {codes = codes}
    if inputs.noise_0 then
        for i = 1, #NOISE_PER_FRAME do args['noise_' .. (i - 1)] = inputs['noise_' .. (i - 1)] end
    else
        loom.seed_rng(inputs.seed or NOISE_SEED)
        bind_noise(args, draw_noise(n), 0, n)
    end
    return loom.run_subgraph('decode', {n_codes = n, n_past = 0}, args)
end

function stream_open(inputs)
    -- Seeds the stream's own generator (a `CodecStream` always has one), so the first frames draw what
    -- `window_one_pass` draws first.
    if #NOISE_PER_FRAME > 0 then loom.seed_rng(inputs.seed or NOISE_SEED) end
    loom.set_state('pending', {})
    loom.set_state('noise', {})
    loom.set_state('start', 0)
    loom.set_state('emitted', 0)
    return {}
end

function stream_feed(inputs)
    local codes = inputs.codes or inputs.tokens or {}
    if #codes % CODES_PER_FRAME ~= 0 then
        error(string.format('codec stream: %d ids is not a whole number of %d-id frames', #codes, CODES_PER_FRAME))
    end
    local pending, noise = loom.state('pending'), loom.state('noise')
    local start, emitted = loom.state('start'), loom.state('emitted')
    for i = 1, #codes do pending[#pending + 1] = codes[i] end
    for _, v in ipairs(draw_noise(#codes / CODES_PER_FRAME)) do noise[#noise + 1] = v end
    local have = start + #pending / CODES_PER_FRAME
    local out = {}
    if have - emitted > LOOKAHEAD_FRAMES then
        out = window_decode(pending, noise, start, emitted, have)
        -- Keep only the frames with their full lookahead; the rest are decoded again next time.
        local ready = have - LOOKAHEAD_FRAMES
        local keep = (ready - emitted) * SAMPLES_PER_FRAME
        for i = #out, keep + 1, -1 do out[i] = nil end
        emitted = ready
        pending, noise, start = window_trim(pending, noise, start, emitted)
    end
    loom.set_state('pending', pending)
    loom.set_state('noise', noise)
    loom.set_state('start', start)
    loom.set_state('emitted', emitted)
    return out
end

function stream_finish(inputs)
    local pending, noise = loom.state('pending'), loom.state('noise')
    local start, emitted = loom.state('start'), loom.state('emitted')
    local have = start + #pending / CODES_PER_FRAME
    local out = {}
    if have > emitted then out = window_decode(pending, noise, start, emitted, have) end
    loom.set_state('emitted', have)
    return out
end
