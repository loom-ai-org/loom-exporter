-- ===== Zipformer streaming: a live stream, its state in the engine, one encoder call per chunk =====
--
-- upstream's `_infer_jit_streaming` + `DecodeStream` for one stream, greedy, at most one symbol per frame.
-- Four topologies (zipformer_streaming_export.py): `fbank`, `encoder_chunk`, `decoder`, `joint`.
--
-- The live door (ADR-098): `stream_open`, `stream_feed` with each piece of audio, `stream_finish`. These
-- functions hold nothing between calls; everything is the host's `StreamState`, reached through
-- `loom.state` / `loom.set_state` (a cursor, the decoder context, audio not yet consumed) and
-- `loom.keep_state` (the encoder's #STATE_SIZES state tensors and the decoder output, which stay on the
-- device). `infer` (01_transcribe.lua) is a stream opened, fed the whole clip and finished in one call.
--
-- **The fbank runs per chunk, over a window of samples, and gives the whole clip's frames.** Kaldi's
-- `snip_edges=False` frame `t` covers samples [HOP*t - PAD_LEFT, HOP*t - PAD_LEFT + WINDOW), reflected
-- only past the clip's two ends. A window starting at a multiple of HOP, `a = HOP*m`, therefore yields
-- the clip's frame `m + j` as its row `j` for every `j >= 1` whose samples it holds (row 0 reads its own
-- left reflection, which is the clip's only when `a = 0`). A chunk at feature row `row` needs rows
-- [row, row + TAIL): it runs once the samples up to its last frame's end have arrived, from
-- `a = HOP*(row - 1)`. When the input ends, the remaining chunks read a window that reaches the clip's
-- end, so the right reflection and the LOG_EPS rows the fbank phase appends are the clip's own.
--
-- Inputs: `stream_feed` takes `waveform` (16 kHz mono, any length); `stream_open`'s `return_encoder`
-- hands back every chunk's encoder rows from `stream_finish` instead of tokens.

-- The absolute sample a chunk at feature row `row` reads up to, and the one its window starts from.
local function _chunk_end(row)
    return HOP * (row + TAIL - 1) + WINDOW - PAD_LEFT
end

local function _chunk_start(row)
    if row == 0 then return 0 end
    return HOP * (row - 1)
end

-- Absolute samples [a, b) of `pcm`, whose first element is absolute sample `start`.
local function _slice(pcm, start, a, b)
    local out, n = {}, 0
    for i = a - start + 1, b - start do
        n = n + 1
        out[n] = pcm[i]
    end
    return out
end

-- One encoder call on feature rows [row, row + TAIL), from the fbank of `window` (absolute samples from
-- `a`), then the joint over its frames; emitted ids are appended to `out`.
local function _chunk(row, window, a, out)
    loom.run_subgraph_and_retain('fbank', {n_samples = #window, n_past = 0}, {waveform = window})
    local inputs = {features = {from = 'fbank', row = row - a / HOP, rows = TAIL}}
    for k = 1, #STATE_SIZES do
        if row == 0 then
            -- The reference's `get_init_states`: every cache empty and nothing processed.
            local zeros = {}
            for i = 1, STATE_SIZES[k] do zeros[i] = 0.0 end
            inputs['state_' .. k] = zeros
        else
            inputs['state_' .. k] = {state = 'encoder', index = k + 1}
        end
    end
    loom.run_subgraph_and_retain('encoder_chunk', {n_tokens = 0, n_past = 0}, inputs)
    loom.keep_state('encoder', 'encoder_chunk')
    -- `return_encoder` keeps every chunk's encoder rows instead of decoding them: the tensor the gate
    -- grades against upstream's streamed encoder, state carry included.
    if loom.state('return_encoder') then
        local rows = loom.state('encoder_rows') or {}
        for _, v in ipairs(loom.get_output('encoder_chunk', 1)) do rows[#rows + 1] = v end
        loom.set_state('encoder_rows', rows)
        return
    end

    -- The joint on blocks of up to JOINT_BLOCK frames against one decoder output, as the non-streaming
    -- driver does (P4.53): the frames after a block's first emission are discarded and the next block
    -- starts after it, so the emissions are the frame-by-frame loop's.
    local context = loom.state('context')
    local n_frames = loom.output_shape('encoder_chunk', 1)[2]
    local t = 0
    while t < n_frames do
        local rows = math.min(JOINT_BLOCK, n_frames - t)
        loom.run_subgraph_and_retain('joint', {n_enc_frames = rows, n_tokens = 0, n_past = 0},
                                      {encoder_frame = {from = 'encoder_chunk', row = t, rows = rows},
                                       decoder_out = {state = 'decoder'}})
        local ks = loom.get_output('joint', 1)
        local emitted = nil
        for i = 1, rows do
            if ks[i] ~= BLANK_ID then emitted = i break end
        end
        if emitted == nil then
            t = t + rows
        else
            local k = ks[emitted]
            out[#out + 1] = k
            for i = 1, CONTEXT_SIZE - 1 do context[i] = context[i + 1] end
            context[CONTEXT_SIZE] = k
            loom.run_subgraph_and_retain('decoder', {n_tokens = 0, n_past = 0}, {context = context})
            loom.keep_state('decoder', 'decoder')
            t = t + emitted
        end
    end
    loom.set_state('context', context)
end

function stream_open(inputs)
    -- `DecodeStream`'s greedy start is `[blank] * context_size`, not greedy_search's `[-1, ..., blank]`.
    local context = {}
    for i = 1, CONTEXT_SIZE do context[i] = BLANK_ID end
    loom.run_subgraph_and_retain('decoder', {n_tokens = 0, n_past = 0}, {context = context})
    loom.keep_state('decoder', 'decoder')
    loom.set_state('context', context)
    loom.set_state('row', 0)
    loom.set_state('pcm', {})
    loom.set_state('pcm_start', 0)
    loom.set_state('return_encoder', inputs.return_encoder)
    return {}
end

function stream_feed(inputs)
    local pcm, start, row = loom.state('pcm'), loom.state('pcm_start'), loom.state('row')
    local n = #pcm
    for i, v in ipairs(inputs.waveform) do pcm[n + i] = v end
    local stop = start + #pcm
    local out = {}
    while _chunk_end(row) <= stop do
        local a = _chunk_start(row)
        _chunk(row, _slice(pcm, start, a, _chunk_end(row)), a, out)
        row = row + CHUNK_FEATURES
    end
    -- Keep what the next chunk's window starts from, and MIN_WINDOW samples more: if the input ends
    -- before that chunk is complete, its final window reaches back that far (stream_finish).
    local keep = math.max(0, _chunk_start(row) - MIN_WINDOW - HOP)
    if keep > start then
        pcm = _slice(pcm, start, keep, stop)
        start = keep
    end
    loom.set_state('pcm', pcm)
    loom.set_state('pcm_start', start)
    loom.set_state('row', row)
    return out
end

function stream_finish(inputs)
    local pcm, start, row = loom.state('pcm'), loom.state('pcm_start'), loom.state('row')
    local n = start + #pcm
    local out = {}
    if n > 0 then
        -- `get_feature_frames` advances by CHUNK_FEATURES and stops once it has passed the clip's
        -- T feature frames plus upstream's PAD_LENGTH rows of padding.
        local n_features = math.floor((n + HOP / 2) / HOP) + PAD_LENGTH
        while row < n_features do
            -- From a multiple of HOP at or before the chunk's first frame, and at least MIN_WINDOW
            -- samples before the end where the clip allows: the fbank phase's declared shortest input.
            local a = math.min(_chunk_start(row), HOP * math.floor((n - MIN_WINDOW) / HOP))
            a = math.max(a, start)
            _chunk(row, _slice(pcm, start, a, n), a, out)
            row = row + CHUNK_FEATURES
        end
    end
    loom.set_state('pcm', {})
    loom.set_state('pcm_start', n)
    loom.set_state('row', row)
    if loom.state('return_encoder') then return loom.state('encoder_rows') or {} end
    return out
end
