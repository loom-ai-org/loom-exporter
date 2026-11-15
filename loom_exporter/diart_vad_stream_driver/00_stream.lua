-- ===== pyannote segmentation, live: diart's voice activity detection (loom.cpp ADR-129). =====
--
-- diart (github.com/juanmc2005/diart) is the reference: its `VoiceActivityDetection` over the audio cut by its
-- `rearrange_audio_stream`, as its `Benchmark.run_single` runs a file. Ported here operation by operation,
-- in the same float arithmetic, because its answer turns on it: which frames a crop takes is a `ceil` of
-- a quotient of seconds, and the first answer has 293 rows where exact arithmetic says 294.
--
-- * The audio is taken a STEP at a time. Each step is appended to the chunk; a chunk longer than DURATION
--   loses its oldest samples and its start moves on by STEP (a float sum, as diart's `start_time += step`);
--   a chunk exactly DURATION long is a pipeline call. So the first call needs DURATION of audio, and a
--   DURATION that is not a whole number of steps is first called one step later, its start stamped at
--   STEP although its samples began earlier -- diart's bookkeeping, kept.
-- * A call runs `infer` (the one-pass model, 7 powerset classes per frame) over the chunk. A frame is
--   speech when its most probable class is not `non_speech`: diart's hard powerset -> multilabel, then the
--   max over the three local speakers. The local speakers themselves mean nothing outside their chunk.
-- * The last LATENCY / STEP calls' speech are averaged over the region [end - LATENCY, end - LATENCY +
--   STEP) of the newest, each weighted by a Hamming window over its own chunk (`DelayedAggregation`,
--   'hamming', crops 'loose' with a fixed duration, out-of-range frames repeating the edge row). Those
--   rows are this call's answer: a region of its own resolution. The first call answers from 0 instead,
--   its own crop with the aggregated rows written over its tail (diart's `_prepend`). diart's array there
--   is float32, so those rows are rounded to float32; the door hands every row back as float32 anyway
--   (`FrameUpdate::values`), and rounding twice is rounding once, so it is not restated here.
-- * `stream_finish` is diart's file source: LATENCY - STEP of zeros after the audio, and, for audio that
--   would not fill one chunk, zeros before it so it does (`get_file_padding`); the last step padded with
--   zeros; every time moved back by the zeros before. With zeros before, nothing was answered yet, so the
--   whole stream is re-run from them.
--
-- The rows are speech scores (`loom.stream.labels`), not `infer`'s classes; diart's `Binarize` (tau 0.6)
-- and its 0.05 s patch of the turns are the host's. Each call's rows sit on their region's own grid, so
-- the driver states every row's start and duration in state `row_times` (FrameStream, ADR-129).
--
-- State: `chunk` (the last call's audio, or everything so far before the first call), `pending` (samples
-- short of a step), `start` (the chunk's start, diart's float), `received`, `calls`, the open settings,
-- and the buffers the next aggregation reads: `buf_speech` (their frames, flat), `buf_start`, `buf_res`.

local function dv_rint(x)
    -- Python's `round` and numpy's `rint`: to the nearest integer, a half to the even one.
    local f = math.floor(x)
    local d = x - f
    if d > 0.5 then return f + 1 end
    if d < 0.5 then return f end
    if f % 2 == 0 then return f end
    return f + 1
end

local function dv_duration(a, b)
    -- pyannote.core's `Segment(a, b).duration`: zero for a segment shorter than its precision (1e-6).
    if b - a > 1e-6 then return b - a end
    return 0.0
end

local function dv_hamming(m)
    -- numpy's `hamming(M)`.
    if m == 1 then return {1.0} end
    local h = {}
    for k = 1, m do
        local n = (1 - m) + 2 * (k - 1)
        h[k] = 0.54 + 0.46 * math.cos(math.pi * n / (m - 1))
    end
    return h
end

local function dv_crop(start, res, focus_start, fixed)
    -- `SlidingWindow(start, res, res).crop(Segment(focus_start, ..), 'loose', fixed)`: the first frame
    -- index (0-based) and the count.
    local i = math.ceil((focus_start - res - start) / res)
    local n = math.floor((fixed + res) / res)
    return i, n
end

local function dv_speech(chunk)
    -- One pipeline call's segmentation: speech (1) or not (0) per frame, first maximum on a tie as
    -- torch's argmax takes it.
    local probs = infer({waveform = chunk})
    local frames = math.floor(#probs / N_CLASSES)
    local speech = {}
    for f = 0, frames - 1 do
        local base = f * N_CLASSES
        local best, arg = probs[base + 1], 0
        for c = 1, N_CLASSES - 1 do
            local p = probs[base + c + 1]
            if p > best then best, arg = p, c end
        end
        speech[f + 1] = (arg == NON_SPEECH) and 0 or 1
    end
    return speech, frames
end

local function dv_call(chunk, start, shift, rows, times)
    -- `VoiceActivityDetection.__call__` on one chunk starting at `start` seconds: its rows and their
    -- (start, duration) appended to `rows` / `times`.
    local speech, frames = dv_speech(chunk)
    local st = 1.0 / SAMPLE_RATE
    local ext_start = start + (0 - 0.5) * st + 0.5 * st
    local res = dv_duration(start, ext_start + #chunk * st) / frames

    local buf_speech, buf_start, buf_res = loom.state('buf_speech'), loom.state('buf_start'), loom.state('buf_res')
    for f = 1, frames do buf_speech[#buf_speech + 1] = speech[f] end
    buf_start[#buf_start + 1] = start
    buf_res[#buf_res + 1] = res
    local n_buf = #buf_start

    -- DelayedAggregation: the region `LATENCY` before the newest buffer's end, one STEP long.
    local latency, step = loom.state('latency'), loom.state('step')
    local last_ext = (start + (0 - 0.5) * res + 0.5 * res) + frames * res
    local r_start = last_ext - latency
    local r_end = r_start + step
    local fixed = dv_duration(r_start, r_end)
    local hamming = dv_hamming(frames)
    local num, den, n_out = {}, {}, nil
    for b = 1, n_buf do
        local i, n = dv_crop(buf_start[b], buf_res[b], r_start, fixed)
        n_out = n
        for k = 1, n do
            local idx = math.min(math.max(i + k - 1, 0), frames - 1)
            local h = hamming[idx + 1]
            local x = buf_speech[(b - 1) * frames + idx + 1]
            if b == 1 then num[k], den[k] = h * x, h else num[k], den[k] = num[k] + h * x, den[k] + h end
        end
    end
    local out = {}
    for k = 1, n_out do out[k] = num[k] / den[k] end
    local o_start, o_res = r_start, dv_duration(r_start, r_end) / n_out

    -- `_prepend`: the first call answers from 0.
    if n_buf == 1 and start == 0 then
        local fixed0 = dv_duration(0, r_end)
        local i, n = dv_crop(start, res, 0, fixed0)
        local first = {}
        for k = 1, n do first[k] = speech[math.min(math.max(i + k - 1, 0), frames - 1) + 1] end
        for k = 1, n_out do first[n - n_out + k] = out[k] end
        out, o_start, o_res = first, 0, r_end / n
    end
    for k = 1, #out do
        rows[#rows + 1] = out[k]
        times[#times + 1] = (o_start + (k - 1) * o_res) + shift
        times[#times + 1] = o_res
    end

    -- The buffer keeps the last LATENCY / STEP calls.
    if n_buf == loom.state('n_windows') then
        local kept = {}
        for k = frames + 1, #buf_speech do kept[#kept + 1] = buf_speech[k] end
        buf_speech = kept
        table.remove(buf_start, 1)
        table.remove(buf_res, 1)
    end
    loom.set_state('buf_speech', buf_speech)
    loom.set_state('buf_start', buf_start)
    loom.set_state('buf_res', buf_res)
end

local function dv_steps(samples, shift, rows, times)
    -- `rearrange_audio_stream`'s scan over whole steps of `samples` (a multiple of the step long), each
    -- chunk that comes out a call.
    local step_n, chunk_n = loom.state('step_samples'), loom.state('chunk_samples')
    local step = loom.state('step')
    local chunk, start = loom.state('chunk'), loom.state('start')
    local calls = loom.state('calls')
    for s = 0, #samples - step_n, step_n do
        for i = 1, step_n do chunk[#chunk + 1] = samples[s + i] end
        if #chunk > chunk_n then
            local cut = {}
            for i = #chunk - chunk_n + 1, #chunk do cut[#cut + 1] = chunk[i] end
            chunk = cut
            start = start + step
        end
        if #chunk == chunk_n then
            dv_call(chunk, start, shift, rows, times)
            calls = calls + 1
        end
    end
    loom.set_state('chunk', chunk)
    loom.set_state('start', start)
    loom.set_state('calls', calls)
end

local function dv_reset_scan()
    loom.set_state('chunk', {})
    loom.set_state('start', 0)
    loom.set_state('calls', 0)
    loom.set_state('buf_speech', {})
    loom.set_state('buf_start', {})
    loom.set_state('buf_res', {})
end

function stream_open(inputs)
    local duration = inputs.duration_seconds or DURATION
    local step = inputs.step_seconds or STEP
    local latency = inputs.latency_seconds or step
    if not (step > 0 and step <= latency and latency <= duration) then
        error(string.format('diart VAD stream: needs 0 < step (%g) <= latency (%g) <= duration (%g)',
                            step, latency, duration))
    end
    local chunk_n = dv_rint(SAMPLE_RATE * duration)
    if chunk_n < MIN_SAMPLES then
        error(string.format('diart VAD stream: a %g s chunk is %d samples, and the model takes at least %d',
                            duration, chunk_n, MIN_SAMPLES))
    end
    loom.set_state('duration', duration)
    loom.set_state('step', step)
    loom.set_state('latency', latency)
    loom.set_state('chunk_samples', chunk_n)
    loom.set_state('step_samples', dv_rint(SAMPLE_RATE * step))
    loom.set_state('n_windows', dv_rint(latency / step))
    loom.set_state('pending', {})
    loom.set_state('received', 0)
    loom.set_state('row_times', {})
    dv_reset_scan()
    return {}
end

function stream_feed(inputs)
    local pcm = inputs.waveform or {}
    loom.set_state('received', loom.state('received') + #pcm)
    local pending = loom.state('pending')
    for i = 1, #pcm do pending[#pending + 1] = pcm[i] end
    local step_n = loom.state('step_samples')
    local rows, times = {}, {}
    local whole = #pending - #pending % step_n
    if whole > 0 then
        local rest = {}
        for i = whole + 1, #pending do rest[#rest + 1] = pending[i] end
        dv_steps(pending, 0, rows, times)
        pending = rest
    end
    loom.set_state('pending', pending)
    loom.set_state('row_times', times)
    return rows
end

function stream_finish(inputs)
    -- diart's file source: `get_file_padding`, then the last step padded with zeros.
    local received = loom.state('received')
    local duration, step, latency = loom.state('duration'), loom.state('step'), loom.state('latency')
    local right = latency - step
    local file_duration = received / SAMPLE_RATE
    local left = 0
    if file_duration + right < duration then left = duration - (file_duration + right) end
    local samples = loom.state('pending')
    if left > 0 then
        if loom.state('calls') > 0 then
            error('diart VAD stream: padding before audio that was already answered')
        end
        -- Nothing answered yet, so the chunk holds every sample so far: run them again behind the zeros.
        local all = {}
        for i = 1, dv_rint(left * SAMPLE_RATE) do all[#all + 1] = 0.0 end
        for _, part in ipairs({loom.state('chunk'), samples}) do
            for i = 1, #part do all[#all + 1] = part[i] end
        end
        samples = all
        dv_reset_scan()
    end
    if right > 0 then
        for i = 1, dv_rint(right * SAMPLE_RATE) do samples[#samples + 1] = 0.0 end
    end
    local step_n = loom.state('step_samples')
    while #samples % step_n ~= 0 do samples[#samples + 1] = 0.0 end
    local rows, times = {}, {}
    dv_steps(samples, -left, rows, times)
    loom.set_state('pending', {})
    loom.set_state('row_times', times)
    return rows
end
