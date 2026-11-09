-- ===== A codec whose decoder is NOT causal, live (loom.cpp ADR-102): windows with left and right context. =====
--
-- The one-pass `decode` graph run over windows: every push decodes the frames it has, with up to
-- `LEFT_FRAMES` already handed back before them, and hands back all but the last `LOOKAHEAD_FRAMES`, which
-- wait for the context after them; `stream_finish` decodes those. A frame's samples are kept only once
-- `LOOKAHEAD_FRAMES` frames follow it, so the stream runs that far behind the codes. Both margins exceed the
-- decoder's receptive field (DAC 44 kHz: 16 frames each side is 4.1e-07 from one call, 8 is 5.9e-03), so a
-- window gives what one call gives, to f32 rounding.
--
-- The functions hold nothing between calls: the codes not yet decoded (and the left context before them)
-- are a state value, with `start` the frame index of its first frame and `emitted` the frames handed back.

local function window_decode(pending, start, from, to)
    -- Frames [from, to) of the stream, decoded with the context `pending` holds before them; returns the
    -- samples of frames [emit_from, to) where `emit_from` is the caller's.
    local first = math.max(start, from - LEFT_FRAMES)
    local codes = {}
    for i = (first - start) * CODES_PER_FRAME + 1, (to - start) * CODES_PER_FRAME do codes[#codes + 1] = pending[i] end
    local wave = loom.run_subgraph('decode', {n_codes = to - first, n_past = 0}, {codes = codes})
    local out = {}
    for i = (from - first) * SAMPLES_PER_FRAME + 1, (to - first) * SAMPLES_PER_FRAME do out[#out + 1] = wave[i] end
    return out
end

local function window_trim(pending, start, emitted)
    -- Drop the frames no future window reaches back to.
    local keep_from = math.max(start, emitted - LEFT_FRAMES)
    if keep_from == start then return pending, start end
    local kept = {}
    for i = (keep_from - start) * CODES_PER_FRAME + 1, #pending do kept[#kept + 1] = pending[i] end
    return kept, keep_from
end

function stream_open(inputs)
    loom.set_state('pending', {})
    loom.set_state('start', 0)
    loom.set_state('emitted', 0)
    return {}
end

function stream_feed(inputs)
    local codes = inputs.codes or inputs.tokens or {}
    if #codes % CODES_PER_FRAME ~= 0 then
        error(string.format('codec stream: %d ids is not a whole number of %d-id frames', #codes, CODES_PER_FRAME))
    end
    local pending, start, emitted = loom.state('pending'), loom.state('start'), loom.state('emitted')
    for i = 1, #codes do pending[#pending + 1] = codes[i] end
    local have = start + #pending / CODES_PER_FRAME
    local out = {}
    if have - emitted > LOOKAHEAD_FRAMES then
        out = window_decode(pending, start, emitted, have)
        -- Keep only the frames with their full lookahead; the rest are decoded again next time.
        local ready = have - LOOKAHEAD_FRAMES
        local keep = (ready - emitted) * SAMPLES_PER_FRAME
        for i = #out, keep + 1, -1 do out[i] = nil end
        emitted = ready
        pending, start = window_trim(pending, start, emitted)
    end
    loom.set_state('pending', pending)
    loom.set_state('start', start)
    loom.set_state('emitted', emitted)
    return out
end

function stream_finish(inputs)
    local pending, start, emitted = loom.state('pending'), loom.state('start'), loom.state('emitted')
    local have = start + #pending / CODES_PER_FRAME
    local out = {}
    if have > emitted then out = window_decode(pending, start, emitted, have) end
    loom.set_state('emitted', have)
    return out
end
