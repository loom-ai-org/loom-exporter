-- ===== Canary, live (loom.cpp ADR-124): NeMo's streaming AED loop, wait-k or AlignAtt =====
--
-- NeMo streams Canary with `speech_to_text_aed_streaming_infer.py` (NVIDIA-NeMo/Speech): every step runs the
-- whole encoder over a buffer of up to `left + chunk + right` seconds (`StreamingBatchedAudioBuffer`), lets
-- the decoder see the buffer's left context and chunk (the right context too on the last chunk), and decodes
-- with `GreedyBatchedStreamingAEDComputer`, the decoder's cache carried from chunk to chunk. This transcribes
-- that loop at batch 1.
--
--   * Wait-k: nothing until `S_WAITK_LAGGING` chunks are in, then ONE token per chunk; on the last chunk, the
--     rest of the transcript.
--   * AlignAtt: per chunk, tokens while the decoder's layer -2 cross-attention (head mean, `decoder` output 2)
--     peaks at least `S_ALIGNATT_THR` encoder frames before the end of what it may see; at most
--     `10 * int(chunk + right)` tokens a chunk.
--
-- A token the policy rejects is not kept, and the next chunk re-runs the last kept token at its position
-- against the new encoder output: NeMo saves its decoder cache only after a token is kept, and the KV cache
-- here is simply overwritten at that position. An end-of-text before the last chunk is a rejection too; on
-- the last chunk it ends the stream, as NeMo's hallucination check (the same token 5 times, `ab` 3 times,
-- `abc` 3 times) and its length caps do.
--
-- The buffer bookkeeping is the script's, both halves of it: the SCALAR context (which trims the buffer) is
-- told when a chunk is the last; the BATCH context (which gives the encoder its length and the decoder its
-- rows) never is -- the script passes `is_last_chunk_batch=zeros` -- so on the last chunk the two differ.
-- One difference from the script, forced by liveness, as in ADR-110: a step runs once a sample past its end
-- has arrived, and the step that reaches the clip's end runs in `stream_finish`.
--
-- State: `pcm` from `pcm_start`, `received`, `ls`/`rs` the next step's samples, `steps`; `c_*` (scalar) and
-- `b_*` (batch) contexts in samples; `ctx` the sizes in encoder frames; `prompt`; `policy`; `seq` (the prompt
-- and every kept token), `decoding_step`, `active`.

local function canary_frames(seconds)
    return math.floor(seconds * S_FEATURES_PER_SEC / S_SUBSAMPLING)
end

-- `ContextSize.add_frames_get_removed_` (scalar, told about the last chunk) and `ContextSizeBatch.add_frames_`
-- (batch, never told), for `n` new samples. Returns the samples the scalar context drops from the left.
local function canary_add(n, last)
    local ctx = loom.state('ctx')
    local chunk, right = S_FRAME_SAMPLES * ctx[2], S_FRAME_SAMPLES * ctx[3]
    local full = S_FRAME_SAMPLES * (ctx[1] + ctx[2] + ctx[3])
    -- scalar
    local cl, cc, cr = loom.state('c_left') + loom.state('c_chunk'), 0, loom.state('c_right') + n
    if last then
        cc, cr = cr, 0
    elseif cr > chunk then
        cc, cr = chunk, cr - chunk
    end
    local extra = math.max(cl + cc + cr - full, 0)
    cl = cl - extra
    loom.set_state('c_left', cl)
    loom.set_state('c_chunk', cc)
    loom.set_state('c_right', cr)
    -- batch
    local bl, bc, br = loom.state('b_left') + loom.state('b_chunk'), chunk, loom.state('b_right') + n - chunk
    if bc <= 0 then bl = 0 end
    bl = bl - math.max(bl + bc + br - full, 0)
    if bl < 0 then bl = 0 end
    loom.set_state('b_left', bl)
    loom.set_state('b_chunk', bc)
    loom.set_state('b_right', br)
    return extra
end

-- One decoder forward of `tokens` at position `n_past` against this step's cross-attention K/V (`k` rows).
local function canary_forward(tokens, n_past, k)
    local inputs = {tokens = tokens, position_ids = loom.range(n_past, #tokens),
                    attention_mask = loom.causal_mask(#tokens, n_past)}
    for i = 0, S_N_LAYERS - 1 do
        inputs['xk_' .. i] = {from = 'cross_kv', index = 2 * i + 1}
        inputs['xv_' .. i] = {from = 'cross_kv', index = 2 * i + 2}
    end
    loom.run_subgraph_and_retain('decoder', {n_tokens = #tokens, n_past = n_past, n_enc_frames = k}, inputs)
    return loom.argmax_row('decoder', #tokens - 1)
end

-- NeMo's `detect_hallucinations` on the sequence with the candidate at `ccl + 1` (1-based `seq`).
local function canary_hallucination(seq, ccl)
    if ccl < 8 then return false end
    local function t(back) return seq[ccl + 1 - back] end
    if t(0) == t(1) and t(0) == t(2) and t(0) == t(3) and t(0) == t(4) then return true end
    if t(0) == t(2) and t(1) == t(3) and t(0) == t(4) and t(1) == t(5) then return true end
    if t(0) == t(3) and t(1) == t(4) and t(2) == t(5) and t(0) == t(6) and t(1) == t(7) and t(2) == t(8) then
        return true
    end
    return false
end

-- The first forward's input: the whole prompt until a token has been kept, else the last kept token.
local function canary_input(seq)
    if loom.state('decoding_step') < 0 then return loom.state('prompt'), 0 end
    return {seq[#seq]}, #seq - 1
end

local function canary_waitk(k, last, out)
    local seq = loom.state('seq')
    local step = loom.state('decoding_step')
    local active = loom.state('active')
    local input, n_past = canary_input(seq)
    local start_from = step + 1
    local stop = last and S_WAITK_MAX_GENERATION or start_from + 1
    for _ = start_from, stop - 1 do
        local token = canary_forward(input, n_past, k)
        local inner = token ~= S_EOS
        if token == S_EOS and last then active = 0 end
        if not inner then break end
        step = step + #input
        local ccl = #seq
        seq[ccl + 1] = token
        if S_HALLUCINATIONS == 1 and canary_hallucination(seq, ccl) then
            -- Written, then not counted: NeMo leaves the context length where it was.
            seq[ccl + 1] = nil
            active, inner = 0, false
        else
            out[#out + 1] = token
            input, n_past = {token}, ccl
            -- Wait-k checks the cap AFTER counting the token, so the token at the cap is kept.
            if #seq == S_WAITK_MAX_GENERATION - 1 then active, inner = 0, false end
        end
        if not inner then break end
    end
    loom.set_state('seq', seq)
    loom.set_state('decoding_step', step)
    loom.set_state('active', active)
end

local function canary_alignatt(k, rows, last, shift, out)
    local seq = loom.state('seq')
    local step = loom.state('decoding_step')
    local active = loom.state('active')
    local input, n_past = canary_input(seq)
    local start_from = step < 0 and 0 or #seq - 1
    local per_step, cap = 0, loom.state('alignatt_cap')
    for i = start_from, S_ALIGNATT_MAX_GENERATION - 1 do
        local token = canary_forward(input, n_past, k)
        -- The last query row's attention over the rows the decoder saw, past the sink frames.
        local exclude
        if i == 0 and rows <= S_EXCLUDE_SINK_FRAMES then
            exclude = math.floor(rows / 2)
        else
            exclude = (shift == 0) and S_EXCLUDE_SINK_FRAMES or 0
        end
        local xatt = loom.get_output('decoder', 2)
        local base = (#input - 1) * k
        local best = exclude
        for j = exclude + 1, k - 1 do
            if xatt[base + j + 1] > xatt[base + best + 1] then best = j end
        end
        local inner = last or (k - (best + 1) >= S_ALIGNATT_THR)
        if not inner then break end
        if token == S_EOS then
            if last then active = 0 end
            break
        end
        step = step + #input
        local ccl = #seq
        seq[ccl + 1] = token
        local stop = false
        if S_HALLUCINATIONS == 1 and canary_hallucination(seq, ccl) then
            seq[ccl + 1] = nil
            active, stop = 0, true
        elseif ccl == S_ALIGNATT_MAX_GENERATION - 1 then
            seq[ccl + 1] = nil
            active, stop = 0, true
        else
            out[#out + 1] = token
        end
        if stop then break end
        input, n_past = {token}, ccl
        per_step = per_step + 1
        if per_step >= cap and not last then break end
    end
    loom.set_state('seq', seq)
    loom.set_state('decoding_step', step)
    loom.set_state('active', active)
end

-- One step of the script's loop: `n` new samples (`last` on the clip's final step), the buffer ending at
-- `stop` in the clip.
local function canary_step(pcm, start, n, last, stop, out)
    local extra = canary_add(n, last)
    local total = loom.state('c_left') + loom.state('c_chunk') + loom.state('c_right')
    local length = loom.state('b_left') + loom.state('b_chunk') + loom.state('b_right')
    local window = {}
    local first = stop - total
    for i = 1, total do window[i] = pcm[first - start + i] end
    loom.run_subgraph_and_retain('encoder', {n_samples = total, n_past = 0}, {waveform = window, length = {length}})
    local rows = loom.output_shape('encoder', 1)[2]
    -- The rows the decoder may see: the batch context's left and chunk, or all of them on the last chunk.
    local k = last and rows or (math.floor(loom.state('b_left') / S_FRAME_SAMPLES)
                                + math.floor(loom.state('b_chunk') / S_FRAME_SAMPLES))
    local ctx = loom.state('ctx')
    local steps = loom.state('steps')
    local end_of_window = loom.state('first_window') + steps * S_FRAME_SAMPLES * ctx[2]
    local shift = math.max(math.floor(end_of_window / S_FRAME_SAMPLES) - ctx[1] - ctx[2], 0)
    loom.set_state('steps', steps + 1)
    if loom.state('active') == 0 then return end
    -- `GreedyBatchedStreamingAEDComputer.__call__`: nothing until `waitk_lagging` chunks are in.
    if math.floor(k / ctx[2]) < S_WAITK_LAGGING and not last then return end
    loom.run_subgraph_and_retain('cross_kv', {n_enc_frames = k, n_past = 0},
                                  {xa = {from = 'encoder', row = 0, rows = k}})
    if loom.state('policy') == 1 then
        canary_alignatt(k, rows, last, shift, out)
    else
        canary_waitk(k, last, out)
    end
end

function stream_open(inputs)
    local ctx = {canary_frames(inputs.left_seconds or S_DEFAULT_LEFT_SECONDS),
                 canary_frames(inputs.chunk_seconds or S_DEFAULT_CHUNK_SECONDS),
                 canary_frames(inputs.right_seconds or S_DEFAULT_RIGHT_SECONDS)}
    if ctx[2] < 1 then error('canary: the chunk must be at least one encoder frame (80 ms)') end
    if ctx[3] < 1 then error('canary: the right context must be at least one encoder frame (80 ms)') end
    if ctx[1] < 0 then error('canary: the left context cannot be negative') end
    if S_FRAME_SAMPLES * (ctx[1] + ctx[2] + ctx[3]) > S_MAX_BUFFER_SAMPLES then
        error('canary: left + chunk + right is longer than the encoder takes (' .. S_MAX_BUFFER_SAMPLES / S_SAMPLE_RATE
              .. ' s)')
    end
    local policy = inputs.policy or S_DEFAULT_POLICY
    if policy ~= 0 and policy ~= 1 then error('canary: policy is 0 (wait-k) or 1 (AlignAtt)') end
    loom.set_state('policy', policy)
    loom.set_state('ctx', ctx)
    -- `max_tokens_per_one_second * int(chunk_secs + right_context_secs)`, from the seconds asked for.
    loom.set_state('alignatt_cap', S_TOKENS_PER_SECOND
                   * math.floor((inputs.chunk_seconds or S_DEFAULT_CHUNK_SECONDS)
                                + (inputs.right_seconds or S_DEFAULT_RIGHT_SECONDS)))
    -- The prompt, with `infer`'s language precedence: target_language, else task, else the default target.
    local source = inputs.language or S_DEFAULT_LANGUAGE
    local target = inputs.target_language
    if target == nil then
        if inputs.task == nil then target = S_DEFAULT_TARGET
        elseif inputs.task == 0 then target = source
        else target = inputs.task end
    end
    local prompt = {}
    for i = 1, #S_STREAM_PROMPT do prompt[i] = S_STREAM_PROMPT[i] end
    prompt[S_SOURCE_SLOT] = source
    prompt[S_TARGET_SLOT] = target
    loom.set_state('prompt', prompt)
    local seq = {}
    for i = 1, #prompt do seq[i] = prompt[i] end
    loom.set_state('seq', seq)
    loom.set_state('decoding_step', -1)
    loom.set_state('active', 1)
    loom.set_state('pcm', {})
    loom.set_state('pcm_start', 0)
    loom.set_state('received', 0)
    loom.set_state('steps', 0)
    loom.set_state('ls', 0)
    loom.set_state('rs', S_FRAME_SAMPLES * (ctx[2] + ctx[3]))
    loom.set_state('first_window', S_FRAME_SAMPLES * ctx[2])
    for _, name in ipairs({'c_left', 'c_chunk', 'c_right', 'b_left', 'b_chunk', 'b_right'}) do
        loom.set_state(name, 0)
    end
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
        canary_step(pcm, start, rs - ls, false, rs, out)
        loom.set_state('ls', rs)
        loom.set_state('rs', rs + chunk)
    end
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
    if received < 1 then error('canary: the stream is empty') end
    local out = {}
    local ls = loom.state('ls')
    if ls < received then
        -- A clip shorter than its first window: the script's first `end_of_window` is the clip's length.
        if loom.state('steps') == 0 then
            loom.set_state('first_window', math.min(loom.state('first_window'), received))
        end
        canary_step(loom.state('pcm'), loom.state('pcm_start'), received - ls, true, received, out)
        loom.set_state('ls', received)
    end
    loom.set_state('pcm', {})
    return out
end
