-- ===== Nemotron 3.5 ASR, live (loom.cpp ADR-109): the cache-aware encoder a chunk at a time =====
--
-- The encoder is trained for this: its attention is chunked (4 frames, 320 ms, see 14 chunks back and
-- none ahead) and every conv is causal, so `transformers`' cache-aware streaming -- conv caches and
-- the last 56 keys and values per layer -- gives the one-pass encoder's rows. `encoder_step` is that
-- step with its 75 states as graph inputs and outputs, kept on the device (`enc`); it equals `encoder`
-- frame for frame (1.9e-14 at f64 on the clips the ADR names).
--
-- A call is whole chunks: chunk c reads mel frames up to 32c + 24, so it runs once the clip's samples
-- reach that frame's window. Its waveform is the centred clip (the one-pass front end's 256 zeros
-- before it) from one sample before frame 32c - 7; the first call's frames before 0 are zeroed in the
-- graph. `stream_finish` runs the rest of the clip, its windows padded with zeros as the one-pass front
-- end pads them (`n_valid` says where the clip ends), so the stream ends on `infer`'s encoder rows.
--
-- The decode is the template's greedy RNN-T loop, frame by frame; its emissions are final, so a feed
-- returns the ids its chunks completed. What it carries across calls is state: the last label, whether
-- the prediction network must run again, and each LSTM layer's h and c (`pred1`.., on the device).
--
-- State: `pcm`, the samples a later window still reads, from clip sample `pcm_start`; `received`;
-- `chunk`, the next chunk; `prompt`, the language prompt id (`stream_open`'s `language`, else the
-- checkpoint's default, `auto`).

local function nemotron_decode(n_frames, out)
    local last_label, stale, primed = loom.state('last_label'), loom.state('stale'), loom.state('primed')
    local zeros = nil
    for t = 0, n_frames - 1 do
        local frame = {from = 'encoder_step', row = t, rows = 1}
        local symbols = 0
        while symbols < S_MAX_SYMBOLS_PER_STEP do
            if stale == 1 then
                loom.run_subgraph_and_retain('embed', {n_tokens = 0, n_past = 0}, {last_label = {last_label}})
                local layer_input = {from = 'embed'}
                for l = 1, S_N_PRED_LAYERS do
                    local cell = 'pred_lstm_l' .. (l - 1) .. '_fwd'
                    local h_prev, c_prev
                    if primed == 1 then
                        h_prev, c_prev = {state = 'pred' .. l, index = 1}, {state = 'pred' .. l, index = 2}
                    else
                        if zeros == nil then
                            zeros = {}
                            for i = 1, S_PRED_HIDDEN do zeros[i] = 0.0 end
                        end
                        h_prev, c_prev = zeros, zeros
                    end
                    loom.run_subgraph_and_retain(cell, {n_tokens = 0, n_past = 0},
                                                  {layer_input = layer_input, h_prev = h_prev, c_prev = c_prev})
                    loom.keep_state('pred' .. l, cell)
                    layer_input = {from = cell, index = 1}
                end
                primed, stale = 1, 0
            end
            loom.run_subgraph_and_retain('joint', {n_tokens = 0, n_past = 0},
                                          {encoder_frame = frame,
                                           decoder_out = {state = 'pred' .. S_N_PRED_LAYERS, index = 1}})
            local k = loom.argmax_row('joint', 0)
            symbols = symbols + 1
            -- A blank advances a frame; a symbol stays on it, at most S_MAX_SYMBOLS_PER_STEP times.
            if k == S_BLANK_ID then break end
            out[#out + 1] = k
            last_label, stale = k, 1
        end
    end
    loom.set_state('last_label', last_label)
    loom.set_state('stale', stale)
    loom.set_state('primed', primed)
end

local function nemotron_encode(window, frame0, n_valid, out)
    local shift = 1
    if loom.state('encoded') == 0 then
        -- A fresh state from `encoder_init`, whose outputs are the states alone; `encoder_step`'s are
        -- the rows, then the states.
        loom.run_subgraph_and_retain('encoder_init', {n_tokens = 0, n_past = 0}, {zero = {0}})
        loom.keep_state('enc', 'encoder_init')
        shift = 0
    end
    local inputs = {waveform = window, frame0 = {frame0}, n_valid = {n_valid}, prompt = {loom.state('prompt')}}
    for k = 1, S_ENC_STATES do inputs['state_' .. k] = {state = 'enc', index = k + shift} end
    loom.run_subgraph_and_retain('encoder_step', {n_samples = #window, n_tokens = 0, n_past = 0}, inputs)
    loom.keep_state('enc', 'encoder_step')
    loom.set_state('encoded', 1)
    nemotron_decode(loom.output_shape('encoder_step', 1)[2], out)
end

-- Clip samples [a, a + n) of the pending audio; past what has arrived they are zeros.
local function nemotron_window(pcm, start, a, n)
    local w = {}
    for i = 1, n do w[i] = pcm[a - start + i] or 0.0 end
    return w
end

-- The clip sample a call at mel frame `frame0` reads from: the preemphasis's neighbour of its window.
local function nemotron_window_start(frame0)
    return frame0 * S_HOP - 1 - S_PAD_LEFT
end

function stream_open(inputs)
    -- The samples before the clip: the virtual frames before 0, one preemphasis neighbour, and the one-
    -- pass front end's centring zeros.
    local lead = S_LEAD_MEL * S_HOP + 1 + S_PAD_LEFT
    local pcm = {}
    for i = 1, lead do pcm[i] = 0.0 end
    loom.set_state('pcm', pcm)
    loom.set_state('pcm_start', -lead)
    loom.set_state('received', 0)
    loom.set_state('chunk', 0)
    loom.set_state('encoded', 0)
    loom.set_state('prompt', inputs.language or S_DEFAULT_PROMPT)
    loom.set_state('last_label', S_BLANK_ID)
    loom.set_state('stale', 1)
    loom.set_state('primed', 0)
    return {}
end

function stream_feed(inputs)
    local pcm, start = loom.state('pcm'), loom.state('pcm_start')
    local received, c = loom.state('received'), loom.state('chunk')
    local n = #pcm
    for i, v in ipairs(inputs.waveform or {}) do pcm[n + i] = v end
    received = received + #(inputs.waveform or {})
    -- Chunk c is ready when its last frame, 32c + 24, has its window's samples: 160 f + 256 of them.
    local ready = math.floor(((received - S_PAD_LEFT) / S_HOP + S_LEAD_MEL + 1) / S_CHUNK_MEL)
    local out = {}
    while ready > c do
        local g = math.min(ready - c, S_MAX_STEP_CHUNKS)
        local frame0 = S_CHUNK_MEL * c - S_LEAD_MEL
        local n_mel = S_CHUNK_MEL * g
        local a = nemotron_window_start(frame0)
        local length = S_HOP * n_mel + S_WINDOW_EXTRA
        nemotron_encode(nemotron_window(pcm, start, a, length), frame0, length - 1, out)
        c = c + g
    end
    -- Keep from the next call's window on.
    local keep = nemotron_window_start(S_CHUNK_MEL * c - S_LEAD_MEL)
    if keep > start then
        local rest = {}
        for i = keep - start + 1, #pcm do rest[#rest + 1] = pcm[i] end
        pcm, start = rest, keep
    end
    loom.set_state('pcm', pcm)
    loom.set_state('pcm_start', start)
    loom.set_state('received', received)
    loom.set_state('chunk', c)
    return out
end

function stream_finish(inputs)
    -- The one-pass front end's frames are the clip's whole hops; the rest of them, through the end.
    local pcm, start = loom.state('pcm'), loom.state('pcm_start')
    local received, c = loom.state('received'), loom.state('chunk')
    local frames = math.floor(received / S_HOP)
    if frames < 1 then error('nemotron-asr: the stream is shorter than one mel frame (10 ms)') end
    local out = {}
    local frame0 = S_CHUNK_MEL * c - S_LEAD_MEL
    local n_mel = frames - frame0
    if n_mel > 0 then
        local a = nemotron_window_start(frame0)
        local length = S_HOP * n_mel + S_WINDOW_EXTRA
        local window = nemotron_window(pcm, start, a, length)
        -- Preemphasised sample i (from 0) is clip sample a + 1 + i: the clip's while below `received`.
        local n_valid = math.max(0, math.min(length - 1, received - a - 1))
        nemotron_encode(window, frame0, n_valid, out)
    end
    loom.set_state('pcm', {})
    loom.set_state('chunk', c + 1)
    return out
end
