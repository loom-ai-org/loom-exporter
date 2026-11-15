-- ===== MOSS-Audio-Tokenizer, live (loom.cpp ADR-103): codes pushed as an LM makes them, samples back. =====
--
-- The decoder is six transformer stages, each its own graph over a RING KV cache of its own: stage k runs
-- `STAGE_PPF[k]` positions per code frame and attends to the last `STAGE_WINDOW[k]` of them, in a ring of
-- `STAGE_RING[k]` cells -- the window plus the most positions one push adds, since a call's keys are
-- written before any of its queries reads them. `stream_feed({codes = ...})` takes whole frames,
-- frame-major (`CODES_PER_FRAME` ids each), and decodes them `MAX_PUSH_FRAMES` at a time, each stage's
-- output the next one's input, and returns the last stage's interleaved samples. `stream_finish` returns
-- nothing: the decoder is causal, so a frame's samples are final when it is decoded.
--
-- The functions hold nothing between calls: `pos` (frames decoded so far) is a state value, and the
-- windows are the stages' caches, leased or owned per stream (ADR-099).

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
        local m = math.min(MAX_PUSH_FRAMES, n - at)
        local pos = loom.state('pos')
        local piece = {}
        for i = 1, m * CODES_PER_FRAME do piece[i] = codes[at * CODES_PER_FRAME + i] end
        -- Stage 1 takes the codes; every later stage the one before it.
        local count, past = m * STAGE_PPF[1], pos * STAGE_PPF[1]
        loom.run_subgraph_and_retain('stage_1', {n_tokens = count, n_past = past},
            {codes = piece, position_ids = loom.range(past, count),
             attention_mask = loom.causal_mask(count, past, STAGE_WINDOW[1], STAGE_RING[1])})
        for k = 2, #STAGE_PPF do
            count, past = m * STAGE_PPF[k], pos * STAGE_PPF[k]
            loom.run_subgraph_and_retain('stage_' .. k, {n_tokens = count, n_past = past},
                {x = {from = 'stage_' .. (k - 1)}, position_ids = loom.range(past, count),
                 attention_mask = loom.causal_mask(count, past, STAGE_WINDOW[k], STAGE_RING[k])})
        end
        local wave = loom.get_output('stage_' .. #STAGE_PPF, 1)
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
