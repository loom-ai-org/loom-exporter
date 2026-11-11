-- ===== Silero VAD, live (loom.cpp ADR-106): a frame's probability once its 512 samples have arrived. =====
--
-- This is the model's own streaming contract -- upstream's `audio_forward` calls the JIT once per 512-sample
-- frame, prefixed with the last 64 samples of the frame before, and carries its `LSTMCell` state -- run
-- through `infer`'s three graphs. `frontend` is a stride-512 convolution with a 64-sample left pad, so a
-- window of the clip starting one whole frame back puts every frame after the first on the clip's own
-- frame grid: its first row reads the 64 samples of context it needs from inside that frame, and the row
-- for the frame itself is dropped. One push of `k` frames is one `frontend` call over `512 (k + 1)`
-- samples, the LSTM's h and c continued through `run_recurrent`'s `carry` (ADR-104), and one `head` call.
--
-- Before the first frame the window's lead is zeros, which is the reference's zero context; at
-- `stream_finish` the samples short of a frame are the last frame, zero-padded on the right by `frontend`
-- as one call pads the clip.
--
-- State: `lead`, the frame before the samples not yet run (zeros at first); `pending`, the samples short
-- of a whole frame; `h` and `c`, the cell's carry (absent before the first frame).

local function slice(a, from, to)
    -- Elements [from, to) (0-based) of a flat array.
    local out = {}
    for i = from + 1, to do out[#out + 1] = a[i] end
    return out
end

local function run_frames(samples)
    -- `samples` is whole frames, or a final part frame: their rows, after the frames already run.
    local lead = loom.state('lead')
    local window = {}
    for i = 1, FRAME do window[i] = lead[i] end
    for i = 1, #samples do window[FRAME + i] = samples[i] end
    local n = math.floor((#samples - 1) / FRAME) + 1
    local rows = loom.run_subgraph('frontend', {n_samples = #window, n_past = 0}, {waveform = window})
    local seq = slice(rows, HIDDEN, (n + 1) * HIDDEN)
    -- A state value is a flat array, so the carry's halves are kept under names of their own (Retro-104).
    local carry = {h = loom.state('h'), c = loom.state('c')}
    seq = loom.run_recurrent('lstm_l0_fwd', seq, n, HIDDEN, HIDDEN, false, carry)
    loom.set_state('h', carry.h)
    loom.set_state('c', carry.c)
    loom.set_state('lead', slice(window, #window - FRAME, #window))
    return loom.run_subgraph('head', {n_enc_frames = n, n_past = 0}, {lstm_out = seq})
end

function stream_open(inputs)
    local zeros = {}
    for i = 1, FRAME do zeros[i] = 0.0 end
    loom.set_state('lead', zeros)
    loom.set_state('pending', {})
    loom.set_state('h', nil)
    loom.set_state('c', nil)
    return {}
end

function stream_feed(inputs)
    local pending = loom.state('pending')
    local pcm = inputs.waveform or {}
    for i = 1, #pcm do pending[#pending + 1] = pcm[i] end
    local whole = math.floor(#pending / FRAME) * FRAME
    loom.set_state('pending', slice(pending, whole, #pending))
    if whole == 0 then return {} end
    return run_frames(slice(pending, 0, whole))
end

function stream_finish(inputs)
    local pending = loom.state('pending')
    loom.set_state('pending', {})
    if #pending == 0 then return {} end
    return run_frames(pending)
end
