-- ===== NeMo's buffered streaming, live (loom.cpp ADR-110, ADR-122): a transducer's chunks over a sliding buffer =====
--
-- A NeMo transducer streams the way NeMo's `speech_to_text_streaming_infer_rnnt.py` runs it: BUFFERED.
-- Every step runs the whole model on a buffer of up to `left + chunk + right` seconds of the clip
-- (`S_STEP_PHASE`: the preprocessor and encoder, so the mel is normalised over the buffer and nothing is
-- reusable across steps) and decodes the buffer's chunk rows. This transcribes that loop at batch 1: the
-- buffer arithmetic below is `StreamingBatchedAudioBuffer`'s (`ContextSize.add_frames_get_removed_`), in
-- samples, so the stream's ids are the script's. They are NOT the one pass's: a step sees less context
-- than the whole clip, which is the point of it.
--
-- Per leaf: the unified model's step also takes its language prompt (`S_STEP_PROMPT`) and the chunked
-- attention mask the script selects for a `chunked_limited_with_rc` encoder (`S_STEP_MASK`, the three
-- sizes in encoder frames); Parakeet's step is its one-pass `encoder` phase, full attention over the
-- buffer, as the script runs a model of any other attention style.
--
-- One difference from the script, forced by liveness: the script knows the clip's length and runs its
-- last step (the chunk and the right context decoded together, no right context left) when the step's
-- end reaches it. A live feed cannot tell the clip has ended, so a step runs only once a sample PAST its
-- end has arrived; the step that ends at the clip's end runs in `stream_finish`, as the script's last.
--
-- State: `pcm`, the clip from sample `pcm_start` on (the samples a later buffer still reads);
-- `received`; `ls`/`rs`, the next step's new samples `[ls, rs)`; `c_left`/`c_chunk`/`c_right`, the
-- buffer's three parts in samples; `ctx`, the three sizes in encoder frames; `prompt`, the language
-- prompt id (`stream_open`'s `language`, else the checkpoint's default) where the step takes one.

local function buffered_frames(seconds)
    -- The script's own rounding: `int(seconds * features_per_sec / subsampling)`.
    return math.floor(seconds * S_FEATURES_PER_SEC / S_SUBSAMPLING)
end

local function buffered_step(pcm, start, total, last, out)
    -- The buffer: the `total` samples ending at the step's end, `rs` (or the clip's end on the last).
    local stop = last and loom.state('received') or loom.state('rs')
    local first = stop - total
    local window = {}
    for i = 1, total do window[i] = pcm[first - start + i] end
    local inputs = {waveform = window, length = {total}}
    if S_STEP_PROMPT == 1 then inputs.prompt = {loom.state('prompt')} end
    if S_STEP_MASK == 1 then
        local ctx = loom.state('ctx')
        inputs.att_left, inputs.att_chunk, inputs.att_right = {ctx[1]}, {ctx[2]}, {ctx[3]}
    end
    loom.run_subgraph_and_retain(S_STEP_PHASE, {n_samples = total, n_tokens = 0, n_past = 0}, inputs)
    -- The script slices the left context off by frames, rounding down (a last step's left part need not
    -- be whole frames), and decodes the chunk's frames -- or, on the last step, every frame after it.
    local row0 = math.floor(loom.state('c_left') / S_FRAME_SAMPLES)
    local rows
    if last then
        rows = loom.output_shape(S_STEP_PHASE, 1)[2] - row0
    else
        rows = math.floor(loom.state('c_chunk') / S_FRAME_SAMPLES)
    end
    transducer_stream_decode(S_STEP_PHASE, row0, rows, out)
end

-- The script's `add_frames_get_removed_` for the samples `[ls, ls + n)`: the old chunk joins the left
-- context, the new samples the right; a full right context gives a chunk up; the left is cut to the
-- buffer's size. With a right context of at least one frame (`stream_open` refuses less), its batch
-- and scalar bookkeeping agree, and this is both.
local function buffered_add(n, last)
    local ctx = loom.state('ctx')
    local left = loom.state('c_left') + loom.state('c_chunk')
    local chunk, right = 0, loom.state('c_right') + n
    if last then
        chunk, right = right, 0
    else
        chunk, right = S_FRAME_SAMPLES * ctx[2], right - S_FRAME_SAMPLES * ctx[2]
    end
    local full = S_FRAME_SAMPLES * (ctx[1] + ctx[2] + ctx[3])
    left = left - math.max(left + chunk + right - full, 0)
    loom.set_state('c_left', left)
    loom.set_state('c_chunk', chunk)
    loom.set_state('c_right', right)
    return left + chunk + right
end

function stream_open(inputs)
    local ctx = {buffered_frames(inputs.left_seconds or S_DEFAULT_LEFT_SECONDS),
                 buffered_frames(inputs.chunk_seconds or S_DEFAULT_CHUNK_SECONDS),
                 buffered_frames(inputs.right_seconds or S_DEFAULT_RIGHT_SECONDS)}
    local frame_ms = math.floor(1000 * S_FRAME_SAMPLES / S_SAMPLE_RATE + 0.5)
    if ctx[2] < 1 then
        error(S_STREAM_NAME .. ': the chunk must be at least one encoder frame (' .. frame_ms .. ' ms)')
    end
    -- NeMo's batch and scalar bookkeeping part ways without a right context, and the script then decodes
    -- the first chunk twice; no reference to match, so no such stream.
    if ctx[3] < 1 then
        error(S_STREAM_NAME .. ': the right context must be at least one encoder frame (' .. frame_ms .. ' ms)')
    end
    if ctx[1] < 0 then error(S_STREAM_NAME .. ': the left context cannot be negative') end
    if S_FRAME_SAMPLES * (ctx[1] + ctx[2] + ctx[3]) > S_MAX_BUFFER_SAMPLES then
        error(S_STREAM_NAME .. ': left + chunk + right is longer than the encoder step takes ('
              .. S_MAX_BUFFER_SAMPLES / S_SAMPLE_RATE .. ' s)')
    end
    loom.set_state('ctx', ctx)
    loom.set_state('pcm', {})
    loom.set_state('pcm_start', 0)
    loom.set_state('received', 0)
    loom.set_state('ls', 0)
    loom.set_state('rs', S_FRAME_SAMPLES * (ctx[2] + ctx[3]))
    loom.set_state('c_left', 0)
    loom.set_state('c_chunk', 0)
    loom.set_state('c_right', 0)
    if S_STEP_PROMPT == 1 then loom.set_state('prompt', inputs.language or S_DEFAULT_PROMPT) end
    transducer_stream_open()
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
    local ctx = loom.state('ctx')
    local chunk = S_FRAME_SAMPLES * ctx[2]
    -- A step runs once a sample past its end is in: only then is it not the clip's last.
    while received > loom.state('rs') do
        local ls, rs = loom.state('ls'), loom.state('rs')
        local total = buffered_add(rs - ls, false)
        buffered_step(pcm, start, total, false, out)
        loom.set_state('ls', rs)
        loom.set_state('rs', rs + chunk)
    end
    -- Keep what the next buffer reads. It is at most a full buffer and ends after `ls`: at `rs` if it is
    -- a feed's step, past `ls` if it is the last (a step runs only with a sample past its end, so the
    -- clip ends after the previous step's end).
    local keep = loom.state('ls') - S_FRAME_SAMPLES * (ctx[1] + ctx[2] + ctx[3])
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
    local out = {}
    -- Every step whose end is before the clip's has run; the one that reaches it is the script's last.
    local ls = loom.state('ls')
    if ls < received then
        local total = buffered_add(received - ls, true)
        buffered_step(loom.state('pcm'), loom.state('pcm_start'), total, true, out)
        loom.set_state('ls', received)
    end
    loom.set_state('pcm', {})
    return out
end
