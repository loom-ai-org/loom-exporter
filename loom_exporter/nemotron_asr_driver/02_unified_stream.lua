-- ===== Nemotron-Labs-Unified-ASR, live (loom.cpp ADR-110): NeMo's buffered streaming mode =====
--
-- The unified model streams the way its reference does (NeMo's `speech_to_text_streaming_infer_rnnt.py`,
-- the recipe its model card gives): BUFFERED, not cache-aware. Every step runs the whole model on a
-- buffer of up to `left + chunk + right` seconds of the clip, under the chunked attention mask of those
-- three sizes (`encoder_step`; the mel is normalised over the buffer, so nothing is reusable across
-- steps), and decodes the buffer's chunk rows. This transcribes that loop at batch 1: the buffer
-- arithmetic below is `StreamingBatchedAudioBuffer`'s (`ContextSize.add_frames_get_removed_`), in
-- samples, so the stream's ids are the script's. They are NOT the one pass's: the streaming mode sees
-- less context than the offline one, which is the point of it.
--
-- One difference from the script, forced by liveness: the script knows the clip's length and runs its
-- last step (the chunk and the right context decoded together, no right context left) when the step's
-- end reaches it. A live feed cannot tell the clip has ended, so a step runs only once a sample PAST its
-- end has arrived; the step that ends at the clip's end runs in `stream_finish`, as the script's last.
--
-- State: `pcm`, the clip from sample `pcm_start` on (the samples a later buffer still reads);
-- `received`; `ls`/`rs`, the next step's new samples `[ls, rs)`; `c_left`/`c_chunk`/`c_right`, the
-- buffer's three parts in samples; `att`, the mask's sizes in encoder frames; `prompt`, the language
-- prompt id (`stream_open`'s `language`, else the checkpoint's default, `auto`).

local function unified_frames(seconds)
    -- The script's own rounding: `int(seconds * features_per_sec / subsampling)`.
    return math.floor(seconds * S_FEATURES_PER_SEC / S_SUBSAMPLING)
end

local function unified_step(pcm, start, total, last, out)
    -- The buffer: the `total` samples ending at the step's end, `rs` (or the clip's end on the last).
    local stop = last and loom.state('received') or loom.state('rs')
    local first = stop - total
    local window = {}
    for i = 1, total do window[i] = pcm[first - start + i] end
    local att = loom.state('att')
    loom.run_subgraph_and_retain('encoder_step', {n_samples = total, n_tokens = 0, n_past = 0},
                                  {waveform = window, length = {total}, prompt = {loom.state('prompt')},
                                   att_left = {att[1]}, att_chunk = {att[2]}, att_right = {att[3]}})
    -- The script slices the left context off by frames, rounding down (a last step's left part need not
    -- be whole frames), and decodes the chunk's frames -- or, on the last step, every frame after it.
    local row0 = math.floor(loom.state('c_left') / S_FRAME_SAMPLES)
    local rows
    if last then
        rows = loom.output_shape('encoder_step', 1)[2] - row0
    else
        rows = math.floor(loom.state('c_chunk') / S_FRAME_SAMPLES)
    end
    transducer_stream_decode('encoder_step', row0, rows, out)
end

-- The script's `add_frames_get_removed_` for the samples `[ls, ls + n)`: the old chunk joins the left
-- context, the new samples the right; a full right context gives a chunk up; the left is cut to the
-- buffer's size. With a right context of at least one frame (`stream_open` refuses less), its batch
-- and scalar bookkeeping agree, and this is both.
local function unified_add(n, last)
    local left = loom.state('c_left') + loom.state('c_chunk')
    local chunk, right = 0, loom.state('c_right') + n
    if last then
        chunk, right = right, 0
    else
        chunk, right = S_FRAME_SAMPLES * loom.state('att')[2], right - S_FRAME_SAMPLES * loom.state('att')[2]
    end
    local att = loom.state('att')
    local full = S_FRAME_SAMPLES * (att[1] + att[2] + att[3])
    left = left - math.max(left + chunk + right - full, 0)
    loom.set_state('c_left', left)
    loom.set_state('c_chunk', chunk)
    loom.set_state('c_right', right)
    return left + chunk + right
end

function stream_open(inputs)
    local att = {unified_frames(inputs.left_seconds or S_DEFAULT_LEFT_SECONDS),
                 unified_frames(inputs.chunk_seconds or S_DEFAULT_CHUNK_SECONDS),
                 unified_frames(inputs.right_seconds or S_DEFAULT_RIGHT_SECONDS)}
    if att[2] < 1 then error('nemotron-unified-asr: the chunk must be at least one encoder frame (80 ms)') end
    -- NeMo's batch and scalar bookkeeping part ways without a right context, and the script then decodes
    -- the first chunk twice; no reference to match, so no such stream.
    if att[3] < 1 then error('nemotron-unified-asr: the right context must be at least one encoder frame (80 ms)') end
    if att[1] < 0 then error('nemotron-unified-asr: the left context cannot be negative') end
    if S_FRAME_SAMPLES * (att[1] + att[2] + att[3]) > S_MAX_BUFFER_SAMPLES then
        error('nemotron-unified-asr: left + chunk + right is longer than the encoder step takes ('
              .. S_MAX_BUFFER_SAMPLES / S_SAMPLE_RATE .. ' s)')
    end
    loom.set_state('att', att)
    loom.set_state('pcm', {})
    loom.set_state('pcm_start', 0)
    loom.set_state('received', 0)
    loom.set_state('ls', 0)
    loom.set_state('rs', S_FRAME_SAMPLES * (att[2] + att[3]))
    loom.set_state('c_left', 0)
    loom.set_state('c_chunk', 0)
    loom.set_state('c_right', 0)
    loom.set_state('prompt', inputs.language or S_DEFAULT_PROMPT)
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
    local chunk = S_FRAME_SAMPLES * loom.state('att')[2]
    -- A step runs once a sample past its end is in: only then is it not the clip's last.
    while received > loom.state('rs') do
        local ls, rs = loom.state('ls'), loom.state('rs')
        local total = unified_add(rs - ls, false)
        unified_step(pcm, start, total, false, out)
        loom.set_state('ls', rs)
        loom.set_state('rs', rs + chunk)
    end
    -- Keep what the next buffer reads. It is at most a full buffer and ends after `ls`: at `rs` if it is
    -- a feed's step, past `ls` if it is the last (a step runs only with a sample past its end, so the
    -- clip ends after the previous step's end).
    local att = loom.state('att')
    local keep = loom.state('ls') - S_FRAME_SAMPLES * (att[1] + att[2] + att[3])
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
    if received < 1 then error('nemotron-unified-asr: the stream is empty') end
    local out = {}
    -- Every step whose end is before the clip's has run; the one that reaches it is the script's last.
    local ls = loom.state('ls')
    if ls < received then
        local total = unified_add(received - ls, true)
        unified_step(loom.state('pcm'), loom.state('pcm_start'), total, true, out)
        loom.set_state('ls', received)
    end
    loom.set_state('pcm', {})
    return out
end
