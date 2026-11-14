-- ===== A NeMo CTC model, live (loom.cpp ADR-123): NeMo's FrameBatchASR =====
--
-- NeMo's buffered CTC inference (`speech_to_text_buffered_infer_ctc.py`, r2.6, over `FrameBatchASR`):
--   * the clip, padded with `delay * model_stride` seconds of zeros, is turned into a raw log-mel ONCE
--     (normalisation off) and cut into `chunk` seconds of mel frames; the last chunk, partial or empty, is
--     zero-filled;
--   * each chunk slides into a `total`-second mel buffer, whose frames start at NeMo's fill value
--     (`S_BUFFER_FILL`); the buffer is normalised per feature over itself and run through the model
--     (`ctc_buffer`);
--   * of each buffer's argmax ids, `tokens` frames ending `delay` frames before its last are kept, and ONE
--     greedy CTC merge runs over everything kept, in order.
-- The arithmetic below is NeMo's, in the same double operations, so its rounding is NeMo's.
--
-- Live, the whole-clip mel is computed a chunk at a time: `raw_mel` over a window of the clip reaching
-- `S_MEL_MARGIN` frames past the chunk each side, of which only the chunk's frames are kept. A log-mel frame
-- reads n_fft samples around its centre and the pre-emphasis one more, so a kept frame never sees the
-- window's edges, and equals the whole clip's (bit for bit in PyTorch). A chunk runs once its window's last
-- sample has arrived; `stream_finish` appends the zero pad and runs the rest, the partial chunk last.
--
-- State: `pcm` from sample `pcm_start` on, `received`; `next_frame`, the first mel frame of the next chunk;
-- `buffer`, the mel buffer, frame-major (`S_N_MEL` values per frame); `prev`, the merge's previous id;
-- `chunk_frames`, `total_frames`, `tokens`, `delay` from `stream_open`'s `chunk_seconds`/`total_seconds`.

local function ctc_window(pcm, start, s0, s1)
    local w = {}
    for i = 1, s1 - s0 do w[i] = pcm[s0 - start + i] or 0.0 end
    return w
end

-- Mel frames [f0, f1) of the padded clip, from samples [s0, s1) of it (`s1` the clip's end on the last).
local function ctc_mel(pcm, start, f0, f1, n_total)
    local s0 = math.max(f0 - S_MEL_MARGIN, 0) * S_HOP
    local s1 = math.min((f1 + S_MEL_MARGIN) * S_HOP, n_total)
    loom.run_subgraph_and_retain('raw_mel', {n_samples = s1 - s0, n_tokens = 0, n_past = 0},
                                  {waveform = ctc_window(pcm, start, s0, s1), length = {s1 - s0}})
    local out, rows = loom.get_output('raw_mel', 1), {}
    local j0 = f0 - s0 / S_HOP
    for i = 1, (f1 - f0) * S_N_MEL do rows[i] = out[j0 * S_N_MEL + i] end
    return rows
end

-- One chunk of mel frames (`frames`, frame-major, `chunk_frames` of them) through the buffer and the model;
-- the kept frames' ids merged into `out`.
local function ctc_step(frames, out)
    local buffer = loom.state('buffer')
    local shift = loom.state('chunk_frames') * S_N_MEL
    local n = #buffer
    for i = 1, n - shift do buffer[i] = buffer[i + shift] end
    for i = 1, shift do buffer[n - shift + i] = frames[i] end
    loom.set_state('buffer', buffer)
    local total = loom.state('total_frames')
    loom.run_subgraph_and_retain('ctc_buffer', {n_tokens = total, n_past = 0},
                                  {features = buffer, length = {total}})
    local ids = loom.argmax_rows('ctc_buffer')
    -- NeMo's `decoded[len(decoded) - 1 - delay : len(decoded) - 1 - delay + tokens_per_chunk]`.
    local first = #ids - 1 - loom.state('delay')
    local prev = loom.state('prev')
    for i = first + 1, math.min(first + loom.state('tokens'), #ids) do
        local p = ids[i]
        -- `greedy_merge`'s rule, the previous id carried across chunks.
        if (p ~= prev or prev == S_BLANK_ID) and p ~= S_BLANK_ID then out[#out + 1] = p end
        prev = p
    end
    loom.set_state('prev', prev)
end

function stream_open(inputs)
    local chunk_s = inputs.chunk_seconds or S_DEFAULT_CHUNK_SECONDS
    local total_s = inputs.total_seconds or S_DEFAULT_TOTAL_SECONDS
    -- `model_stride_in_secs = feature_stride * model_stride`, then the script's two ceilings and
    -- FeatureFrameBufferer's two floors, in that order of operations.
    local stride_s = S_FEATURE_STRIDE * S_MODEL_STRIDE
    local chunk_frames = math.floor(chunk_s / S_FEATURE_STRIDE)
    local total_frames = math.floor(total_s / S_FEATURE_STRIDE)
    if chunk_frames < 1 then error(S_STREAM_NAME .. ': the chunk must be at least one mel frame') end
    if total_frames < chunk_frames then
        error(S_STREAM_NAME .. ': the buffer (total_seconds) cannot be shorter than the chunk')
    end
    if total_frames > S_MAX_BUFFER_FRAMES then
        error(S_STREAM_NAME .. ': the buffer is longer than the model step takes ('
              .. S_MAX_BUFFER_FRAMES * S_FEATURE_STRIDE .. ' s)')
    end
    loom.set_state('chunk_frames', chunk_frames)
    loom.set_state('total_frames', total_frames)
    loom.set_state('tokens', math.ceil(chunk_s / stride_s))
    local delay = math.ceil((chunk_s + (total_s - chunk_s) / 2) / stride_s)
    loom.set_state('delay', delay)
    loom.set_state('pad', math.floor(delay * stride_s * S_SAMPLE_RATE))
    local buffer = {}
    for i = 1, total_frames * S_N_MEL do buffer[i] = S_BUFFER_FILL end
    loom.set_state('buffer', buffer)
    loom.set_state('pcm', {})
    loom.set_state('pcm_start', 0)
    loom.set_state('received', 0)
    loom.set_state('next_frame', 0)
    loom.set_state('prev', S_BLANK_ID)
    return {}
end

function stream_feed(inputs)
    local pcm, start = loom.state('pcm'), loom.state('pcm_start')
    local n = #pcm
    local samples = inputs.waveform or {}
    for i, v in ipairs(samples) do pcm[n + i] = v end
    local received = loom.state('received') + #samples
    loom.set_state('received', received)
    local out = {}
    local chunk = loom.state('chunk_frames')
    local f0 = loom.state('next_frame')
    -- A chunk's frames are final once its window's last sample is in: whatever follows, pad or speech, is
    -- outside every kept frame's reach.
    while received >= (f0 + chunk + S_MEL_MARGIN) * S_HOP do
        ctc_step(ctc_mel(pcm, start, f0, f0 + chunk, received), out)
        f0 = f0 + chunk
    end
    loom.set_state('next_frame', f0)
    -- Keep what the next window reads.
    local keep = math.max(f0 - S_MEL_MARGIN, 0) * S_HOP
    if keep > start then
        local rest = {}
        for i = keep - start + 1, #pcm do rest[#rest + 1] = pcm[i] end
        pcm, start = rest, keep
    end
    loom.set_state('pcm', pcm)
    loom.set_state('pcm_start', start)
    return out
end

function stream_finish(inputs)
    local received = loom.state('received')
    if received < 1 then error(S_STREAM_NAME .. ': the stream is empty') end
    local pcm, start = loom.state('pcm'), loom.state('pcm_start')
    -- The script's zero pad, then NeMo's frame count for the padded clip: floor(n / hop).
    local n_total = received + loom.state('pad')
    local features = math.floor(n_total / S_HOP)
    local chunk = loom.state('chunk_frames')
    local f0 = loom.state('next_frame')
    local out = {}
    while f0 + chunk <= features do
        ctc_step(ctc_mel(pcm, start, f0, f0 + chunk, n_total), out)
        f0 = f0 + chunk
    end
    -- AudioFeatureIterator's last frame: what is left, zero-filled to a chunk -- all zeros when nothing is.
    local frames = {}
    if f0 < features then frames = ctc_mel(pcm, start, f0, features, n_total) end
    for i = #frames + 1, chunk * S_N_MEL do frames[i] = 0.0 end
    ctc_step(frames, out)
    loom.set_state('next_frame', features)
    loom.set_state('pcm', {})
    return out
end
