-- ===== Kyutai STT, live (loom.cpp ADR-108): the Mimi encoder a few frames per call, the LM a step per frame. =====
--
-- moshi runs this model as a stream already: Mimi's encoder takes each 80 ms frame with its streaming state
-- (every causal conv's last inputs, the transformer's ring of 250 positions), and the LM takes one step per
-- frame over its own ring of 750. The one-pass `infer` recomputes the encoder in chunks with 1008 frames of
-- left context instead; here `mimi_encode_step` carries the state itself, on the device (`mimi`), and is
-- the one-call encoder to f64 rounding (1e-15 on jfk at pieces of 1, 3, 7, 2, 16 and 5 frames).
--
-- Every LM step's text id is final, so a feed returns the ids its frames completed and the transcript only
-- grows. The text runs `audio_delay_seconds` (0.5 s) behind the audio, as the model was trained;
-- `stream_finish` pushes `stt_config`'s trailing silence (the delay and 1 s more) so the last words come
-- out, and keeps whole frames only, as `infer` does: the stream ends on one call's ids.
--
-- State: `pending`, the samples not yet a whole frame; `frames`, the frames encoded; `text`, the last LM
-- step's id (the next step's input); `mimi`, the encoder's state tensors. The LM's history is the model's
-- KV cache, which a stream leases or owns (ADR-099).

local function kyutai_codes(n)
    -- The split RVQ over `mimi_encode_step`'s `n` rows in one graph, the argmax of each stage on the device
    -- (`rvq_encode`): frame-major codes, `codes[f * S_N_Q + q + 1]`. `infer`'s chain of 34 calls draws the
    -- same codes; at a frame or two a push, those calls cost more than the frame lasts.
    loom.run_subgraph_and_retain('rvq_encode', {n_codes = n, n_past = 0}, {rows = {from = 'mimi_encode_step'}})
    return loom.get_output('rvq_encode', 1)
end

local function kyutai_lm_step(k, tokens)
    -- LM step `k` at position `k`; its cache is a ring of S_LM_CONTEXT cells (ADR-066).
    loom.run_subgraph_and_retain('lm', {n_tokens = 1, n_past = k},
        {tokens = tokens, position_ids = {k}, attention_mask = loom.causal_mask(1, math.min(k, S_LM_CONTEXT - 1))})
    return loom.argmax_row('lm', 0)
end

local function kyutai_frames(wave, n, ids)
    -- `n` whole frames of `wave`: encoded with the carried state, quantized, then one LM step each.
    local f0 = loom.state('frames')
    if f0 == 0 then
        -- moshi's `run_inference` steps the first frame twice: step 0 sees the initial tokens whatever it
        -- is handed, and its id is not kept. Step k >= 1 then sees step k-1's id and frame k-1's codes.
        local tokens = {S_TEXT_INITIAL}
        for q = 1, S_N_Q do tokens[q + 1] = S_AUDIO_INITIAL end
        loom.set_state('text', kyutai_lm_step(0, tokens))
    end
    local inputs = {waveform = wave, positions = loom.range(2 * f0, 2 * n),
                    key_positions = loom.range(2 * f0 - MIMI_CONTEXT, MIMI_CONTEXT + 2 * n),
                    first = {f0 == 0 and 1.0 or 0.0}}
    -- A fresh state from `mimi_encode_init`, whose outputs are the states alone; `mimi_encode_step`'s are
    -- the rows, then the states.
    local shift = 1
    if f0 == 0 then
        loom.run_subgraph_and_retain('mimi_encode_init', {n_tokens = 0, n_past = 0}, {zero = {0}})
        loom.keep_state('mimi', 'mimi_encode_init')
        shift = 0
    end
    for k = 1, MIMI_STATES do inputs['state_' .. k] = {state = 'mimi', index = k + shift} end
    loom.run_subgraph_and_retain('mimi_encode_step', {n_codes = n, n_tokens = 0, n_past = 0}, inputs)
    loom.keep_state('mimi', 'mimi_encode_step')
    local codes = kyutai_codes(n)
    local text = loom.state('text')
    for j = 0, n - 1 do
        local tokens = {text}
        for q = 1, S_N_Q do tokens[q + 1] = codes[j * S_N_Q + q] end
        text = kyutai_lm_step(f0 + j + 1, tokens)
        if text ~= S_TEXT_UNK and text ~= S_TEXT_PAD then ids[#ids + 1] = text end
    end
    loom.set_state('text', text)
    loom.set_state('frames', f0 + n)
end

local function kyutai_drain(pcm)
    -- Appends `pcm` to what is pending and runs every whole frame, MAX_STEP_FRAMES per call.
    local pending = loom.state('pending')
    for i = 1, #pcm do pending[#pending + 1] = pcm[i] end
    local ids = {}
    local whole = math.floor(#pending / S_FRAME_SAMPLES)
    local at = 0
    while whole > 0 do
        local n = math.min(whole, MAX_STEP_FRAMES)
        local wave = {}
        for i = 1, n * S_FRAME_SAMPLES do wave[i] = pending[at + i] end
        kyutai_frames(wave, n, ids)
        at = at + n * S_FRAME_SAMPLES
        whole = whole - n
    end
    local rest = {}
    for i = at + 1, #pending do rest[#rest + 1] = pending[i] end
    loom.set_state('pending', rest)
    return ids
end

function stream_open(inputs)
    -- `stt_config`'s silence before the audio (none in this release) is the stream's first samples.
    local pending = {}
    for i = 1, S_PAD_LEFT do pending[i] = 0.0 end
    loom.set_state('pending', pending)
    loom.set_state('frames', 0)
    loom.set_state('text', S_TEXT_INITIAL)
    return {}
end

function stream_feed(inputs)
    return kyutai_drain(inputs.waveform or {})
end

function stream_finish(inputs)
    -- The trailing silence, then whole frames only: what is left short of a frame is dropped, as `infer`
    -- (and moshi) drop it.
    local tail = {}
    for i = 1, S_PAD_RIGHT do tail[i] = 0.0 end
    local ids = kyutai_drain(tail)
    if loom.state('frames') == 0 then error('kyutai-stt: the stream is shorter than one frame') end
    loom.set_state('pending', {})
    return ids
end
