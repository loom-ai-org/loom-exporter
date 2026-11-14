-- ===== A frame model, live (loom.cpp ADR-106, ADR-120): each frame's row once the samples it reads have arrived. =====
--
-- Frame t covers samples `[HOP t, HOP (t + 1))` and reads at most LEFT_FRAMES frames' worth of samples
-- before them and RIGHT_FRAMES after (`frame_window_stream.frame_reach` measured both on the traced graph;
-- RIGHT_FRAMES is 0 for a causal model). So the rows of frames `[e, f)` are `infer`'s graph run over the
-- samples from `HOP (e - LEFT_FRAMES)` to `HOP (f + RIGHT_FRAMES)`, the rows `[e, f)` of its answer kept:
-- every sample those frames read is in the window. A window that starts at the stream's first sample, or
-- ends at its last (`stream_finish`), has the clip's own edges, so the first and last frames are the one
-- pass's too.
--
-- `lookahead_frames` (stream_open) trades exactness for latency: a frame is answered once that many
-- frames past it have arrived instead of RIGHT_FRAMES, from a window that ends there. Absent, it is
-- RIGHT_FRAMES and the rows are one call's.
--
-- A push of many frames is cut into windows of at most MAX_PIECE kept frames, so no call is longer than
-- the graph takes: a stream has no length limit, where one call does.
--
-- State: `context`, the audio from sample `base` on (the oldest frame a later window reads, LEFT_FRAMES
-- before the next one to answer) up to where the last window ended; `pending`, the samples pushed since;
-- `received`, the samples pushed so far; `emitted`, the frames answered; `lookahead`. Kept apart so a push
-- that completes no frame touches only `pending`, at most a frame and the look-ahead long: one buffer
-- read and rewritten whole made a 1-sample push cost the 2.5 s of WakeHuBERT's context.

local function fw_window(samples, base, a, to, e, f)
    -- `infer`'s graph over samples `[HOP a, to)` of `samples` (which starts at sample `base`), rows `[e, f)`
    -- of its answer kept (to its end when f is nil).
    local w = {}
    for i = HOP * a - base + 1, to - base do w[#w + 1] = samples[i] end
    local args = {waveform = w}
    if TAKES_LENGTH == 1 then args.length = {#w} end
    local out = loom.run_subgraph('main_topology', {n_samples = #w, n_past = 0}, args)
    -- Cut from the front, where the window's frames and the stream's agree; the window's own count (the
    -- encoder's length for MarbleNet) only says where its answer ends.
    local rows = {}
    local last = f and (f - a) or math.floor(#out / ROW_WIDTH)
    for i = (e - a) * ROW_WIDTH + 1, last * ROW_WIDTH do rows[#rows + 1] = out[i] end
    return rows
end

local function fw_joined(pcm)
    -- Context, pending and the new samples as one array from `base`; the pending tail is emptied.
    local samples = loom.state('context')
    for _, part in ipairs({loom.state('pending'), pcm}) do
        for i = 1, #part do samples[#samples + 1] = part[i] end
    end
    loom.set_state('pending', {})
    return samples, loom.state('base')
end

local function fw_keep(samples, base, e)
    -- The context the next window needs: from LEFT_FRAMES before frame `e`.
    local keep = math.max(base, HOP * math.max(0, e - LEFT_FRAMES))
    local kept = {}
    for i = keep - base + 1, #samples do kept[#kept + 1] = samples[i] end
    loom.set_state('context', kept)
    loom.set_state('base', keep)
end

function stream_open(inputs)
    local lookahead = inputs.lookahead_frames or RIGHT_FRAMES
    if lookahead < 0 or lookahead > RIGHT_FRAMES then
        error(string.format('frame stream: lookahead_frames is 0 to %d (the exact look-ahead)', RIGHT_FRAMES))
    end
    loom.set_state('context', {})
    loom.set_state('pending', {})
    loom.set_state('base', 0)
    loom.set_state('received', 0)
    loom.set_state('emitted', 0)
    loom.set_state('lookahead', lookahead)
    return {}
end

function stream_feed(inputs)
    local pcm = inputs.waveform or {}
    local received = loom.state('received') + #pcm
    loom.set_state('received', received)
    local lookahead = loom.state('lookahead')
    local e = loom.state('emitted')
    local ready = math.floor(received / HOP) - lookahead
    if ready <= e then
        local pending = loom.state('pending')
        for i = 1, #pcm do pending[#pending + 1] = pcm[i] end
        loom.set_state('pending', pending)
        return {}
    end
    local samples, base = fw_joined(pcm)
    local rows = {}
    while ready > e do
        local f = math.min(ready, e + MAX_PIECE)
        local part = fw_window(samples, base, math.max(0, e - LEFT_FRAMES), HOP * (f + lookahead), e, f)
        for i = 1, #part do rows[#rows + 1] = part[i] end
        e = f
    end
    loom.set_state('emitted', e)
    fw_keep(samples, base, e)
    return rows
end

function stream_finish(inputs)
    -- A causal model answered every whole frame as it arrived, and the samples short of one are not a
    -- frame in one call either. Otherwise the last window ends at the stream's last sample, as one call's
    -- clip does, and every frame it answers from `emitted` on is the one pass's.
    local received, e = loom.state('received'), loom.state('emitted')
    if RIGHT_FRAMES == 0 or received == 0 then return {} end
    local a = math.max(0, e - LEFT_FRAMES)
    if received - HOP * a < MIN_SAMPLES then
        error(string.format('frame stream: %d samples, and the model takes at least %d',
                            received - HOP * a, MIN_SAMPLES))
    end
    local samples, base = fw_joined({})
    local rows = fw_window(samples, base, a, received, e, nil)
    loom.set_state('emitted', e + #rows / ROW_WIDTH)
    fw_keep(samples, base, e + #rows / ROW_WIDTH)
    return rows
end
