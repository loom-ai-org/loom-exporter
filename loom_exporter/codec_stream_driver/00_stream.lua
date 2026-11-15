-- ===== A causal codec, live (loom.cpp ADR-102): codes pushed as an LM makes them, samples back. =====
--
-- `stream_open` decodes nothing. `stream_feed({codes = ...})` takes any whole number of frames,
-- frame-major (`CODES_PER_FRAME` ids each), and returns their samples: `codec_step` in calls of at most
-- `MAX_STEP_FRAMES`, the decoder's state carried between them in the stream's `codec` tensors, so the
-- samples are what one call over every frame pushed so far would give. `stream_finish` returns nothing:
-- the decoder is causal, so a frame's samples are final when it is decoded.
--
-- The functions hold nothing between calls: `pos` (frames decoded so far) is a state value, and the
-- decoder state is on the device (ADR-098).

function stream_open(inputs)
    loom.set_state('pos', 0)
    return {}
end

function stream_feed(inputs)
    local codes = inputs.codes or inputs.tokens or {}
    local n = math.floor(#codes / CODES_PER_FRAME)
    if n * CODES_PER_FRAME ~= #codes then
        error(string.format('codec stream: %d ids is not a whole number of %d-id frames', #codes,
                            CODES_PER_FRAME))
    end
    local out = {}
    local at = 0
    while at < n do
        local m = math.min(MAX_STEP_FRAMES, n - at)
        local pos = loom.state('pos')
        local piece = {}
        for i = 1, m * CODES_PER_FRAME do piece[i] = codes[at * CODES_PER_FRAME + i] end
        local args = {codes = piece, positions = loom.range(pos, m),
                      key_positions = loom.range(pos - CODEC_WINDOW, CODEC_WINDOW + m)}
        -- A stream's first call starts from `codec_init`'s fresh state, whose outputs are the states
        -- alone; `codec_step`'s are the samples, then the states.
        local shift = 1
        if pos == 0 then
            loom.run_subgraph_and_retain('codec_init', {n_codes = 0, n_past = 0}, {zero = {0}})
            loom.keep_state('codec', 'codec_init')
            shift = 0
        end
        for k = 1, CODEC_STATES do args['state_' .. k] = {state = 'codec', index = k + shift} end
        loom.run_subgraph_and_retain('codec_step', {n_codes = m, n_past = 0}, args)
        loom.keep_state('codec', 'codec_step')
        local wave = loom.get_output('codec_step', 1)
        local base = #out
        for i = 1, #wave do out[base + i] = wave[i] end
        loom.set_state('pos', pos + m)
        at = at + m
    end
    return out
end

function stream_finish(inputs)
    return {}
end
