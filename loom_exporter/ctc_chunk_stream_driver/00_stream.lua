-- ===== A wav2vec 2.0-style CTC model, live (loom.cpp ADR-126): transformers' strided chunks =====
--
-- The transformers ASR pipeline's chunking (`pipeline(..., chunk_length_s, stride_length_s)`, the reference
-- for these checkpoints, which have no streaming mode of their own):
--   * `chunk_iter` cuts the clip into chunks of `chunk_len` samples starting every `chunk_len - left - right`; each
--     chunk is the model's whole input, its feature extractor's normalisation over the chunk alone (which is
--     what `infer`'s graph does over whatever it is handed);
--   * a chunk's first `left` samples are a stride (none on the first chunk), its last `right` too (none on the
--     last); `rescale_stride` turns them into logit frames and the frames between are kept;
--   * the kept argmax ids of every chunk, in order, are ONE greedy CTC merge (the tokenizer's grouping).
-- Sizes are the pipeline's, in its order of operations: seconds to samples rounded to a whole number of
-- logit frames (Python's round, half to even), the strides to frames the same way.
--
-- Live, a chunk runs once a sample PAST its end has arrived: then it is not the last, which is the one fact
-- about it the pipeline reads off the clip's length. `stream_finish` runs what is left, the last chunk with
-- no right stride. A chunk's ids are final when it runs, so the transcript only grows.
--
-- State: `pcm` from sample `pcm_start` on (`loom.append_state` / `loom.state_slice`), `received`;
-- `next_start`, the first sample of the next chunk; `prev`, the merge's previous id; `chunk_len`,
-- `stride_left`, `stride_right` from `stream_open`'s `chunk_seconds` and `stride_seconds` (or
-- `left_stride_seconds` / `right_stride_seconds`).

-- Python's `round(x)` for a non-negative double: half to even.
local function ctc_py_round(x)
    local f = math.floor(x)
    local d = x - f
    if d > 0.5 then return f + 1 end
    if d < 0.5 then return f end
    if f % 2 == 0 then return f end
    return f + 1
end

-- The chunk of samples [s, e): the model over it, its kept ids merged into `out`.
local function ctc_chunk(s, e, last, out)
    local n = e - s
    if n < CTC_MIN_SAMPLES then
        error(string.format('ctc stream: a chunk of %d samples, and the model takes at least %d',
                            n, CTC_MIN_SAMPLES))
    end
    local w = loom.state_slice('pcm', s - loom.state('pcm_start'), n)
    loom.run_subgraph_and_retain('main_topology', {n_samples = n, n_past = 0}, {waveform = w})
    local ids = loom.argmax_rows('main_topology')
    -- `rescale_stride(stride, 1 / inputs_to_logits_ratio)`, then `items[:, left : token_n - right]`.
    local left = s == 0 and 0 or loom.state('stride_left')
    local right = last and 0 or loom.state('stride_right')
    local token_n = ctc_py_round(n * (1 / CTC_ALIGN))
    local l = ctc_py_round(left / n * token_n)
    local r = ctc_py_round(right / n * token_n)
    local prev = loom.state('prev')
    for i = l + 1, math.min(token_n - r, #ids) do
        local p = ids[i]
        if p ~= prev and p ~= CTC_BLANK then out[#out + 1] = p end
        prev = p
    end
    loom.set_state('prev', prev)
end

function stream_open(inputs)
    local chunk_s = inputs.chunk_seconds or CTC_DEFAULT_CHUNK_SECONDS
    -- The pipeline's default stride is a sixth of the chunk, the same on both sides.
    local stride_s = inputs.stride_seconds or chunk_s / 6
    local left_s = inputs.left_stride_seconds or stride_s
    local right_s = inputs.right_stride_seconds or stride_s
    local function samples(seconds)
        return ctc_py_round(seconds * CTC_RATE / CTC_ALIGN) * CTC_ALIGN
    end
    local chunk_len, left, right = samples(chunk_s), samples(left_s), samples(right_s)
    -- `chunk_len < left + right` is the pipeline's refusal; equal is a step of 0, which `range` refuses.
    if chunk_len <= left + right then
        error('ctc stream: the chunk must be longer than its two strides')
    end
    if chunk_len > CTC_MAX_SAMPLES then
        error(string.format('ctc stream: the chunk is longer than the model takes (%g s)',
                            CTC_MAX_SAMPLES / CTC_RATE))
    end
    -- The last chunk is longer than the left stride (`chunk_iter` drops it otherwise), so a stride the
    -- model takes is what keeps every chunk after the first in its range.
    if left < CTC_MIN_SAMPLES then
        error(string.format('ctc stream: the left stride must be at least %g s', CTC_MIN_SAMPLES / CTC_RATE))
    end
    loom.set_state('chunk_len', chunk_len)
    loom.set_state('stride_left', left)
    loom.set_state('stride_right', right)
    loom.set_state('pcm', {})
    loom.set_state('pcm_start', 0)
    loom.set_state('received', 0)
    loom.set_state('next_start', 0)
    loom.set_state('prev', CTC_BLANK)
    return {}
end

function stream_feed(inputs)
    local start = loom.state('pcm_start')
    local received = start + loom.append_state('pcm', inputs.waveform)
    loom.set_state('received', received)
    local chunk_len = loom.state('chunk_len')
    local step = chunk_len - loom.state('stride_left') - loom.state('stride_right')
    local s = loom.state('next_start')
    local out = {}
    while received > s + chunk_len do
        ctc_chunk(s, s + chunk_len, false, out)
        s = s + step
    end
    loom.set_state('next_start', s)
    if s > start then
        loom.drop_state('pcm', s - start)
        loom.set_state('pcm_start', s)
    end
    return out
end

function stream_finish(inputs)
    local n = loom.state('received')
    if n < 1 then error('ctc stream: the stream is empty') end
    local chunk_len = loom.state('chunk_len')
    local left = loom.state('stride_left')
    local step = chunk_len - left - loom.state('stride_right')
    local s = loom.state('next_start')
    local out = {}
    -- `chunk_iter`'s loop over what is left: every chunk but the last already ran in `stream_feed`.
    while s < n do
        local last = s + chunk_len >= n
        if math.min(s + chunk_len, n) - s > (s == 0 and 0 or left) then
            ctc_chunk(s, math.min(s + chunk_len, n), last, out)
        end
        if last then break end
        s = s + step
    end
    loom.set_state('next_start', n)
    loom.set_state('pcm', {})
    loom.set_state('pcm_start', n)
    return out
end
