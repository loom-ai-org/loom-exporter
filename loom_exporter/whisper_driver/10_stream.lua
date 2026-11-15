-- ===== Whisper, live (loom.cpp ADR-128): SimulStreaming's AlignAtt =====
--
-- ufal's SimulStreaming (`PaddedAlignAttWhisper.insert_audio` / `infer`, `SimulWhisperOnline.process_iter`),
-- at its CLI's defaults: greedy, a 30 s audio buffer, `frame_threshold` 25, `rewind_threshold` 200, no CIF
-- model -- which `load_cif` turns into always firing, so an update's last word is never held back.
--
-- Every `chunk_seconds` of audio (1.2 s, its `--min-chunk-size`) is a segment of the buffer, and one update:
--   * the segment is appended; while the buffer holds more than one segment and is over 30 s, the oldest
--     segment leaves it, and the oldest committed block of tokens moves into the text context;
--   * the buffer (zero-padded to 30 s) is encoded, and the decoder re-reads, from scratch, `<|startofprev|>`
--     + the context (when there is any), the start of transcript, and every committed block;
--   * it then decodes greedily, suppressing the control tokens (and on its first step a bare space and the
--     end of text), until the end of text, a rewind, or until the token's most-attended frame is within
--     `frame_threshold` frames (4 on the last update) of the audio's end -- that token is dropped;
--   * the tokens decoded are this update's: committed, and appended to the transcript.
-- The most-attended frame is SimulStreaming's: the alignment heads' attention (the decoder's second output)
-- over every token of this update so far, each head z-scored over the tokens per frame, a 7-wide median
-- filter along the frames, the heads averaged, the newest token's row's argmax within the audio's frames.
-- The z-score's mean and deviation are kept as running sums, in double.
--
-- `stream_finish` runs what has gathered as a last short segment, then the last update (`is_last`).
--
-- State: `pcm` (the buffer's audio, `loom.append_state`), `pending` (samples short of a segment), `seg_lens`;
-- `blocks` / `block_lens` (the committed tokens, block by block); `ctx` (the context's text, kept as its
-- ids); `last_attend`; `chunk`, `threshold`, `rewind`, `language`; for the gate, `start_tokens` and `frames` of the
-- last update.

local function ws_forward(tokens, n_past)
    local inputs = {tokens = tokens, position_ids = loom.range(n_past, #tokens),
                    attention_mask = loom.causal_mask(#tokens, n_past)}
    for i = 0, W_N_TEXT_LAYERS - 1 do
        inputs['xk_' .. i] = {from = 'cross_kv', index = 2 * i + 1}
        inputs['xv_' .. i] = {from = 'cross_kv', index = 2 * i + 2}
    end
    loom.run_subgraph_and_retain('decoder', {n_tokens = #tokens, n_past = n_past}, inputs)
end

local function ws_initial()
    local t = {W_SOT}
    if W_LANG_HI > 0 then
        t[#t + 1] = loom.state('language')
        t[#t + 1] = W_TRANSCRIBE
    end
    t[#t + 1] = W_NO_TIMESTAMPS
    return t
end

-- `Tokenizer.split_tokens_on_spaces` over `ids`: per word, its token count.
local function ws_word_lengths(ids)
    local lengths, words, current, offset = {}, {}, {}, 0
    local full = loom.detokenize(ids)
    local bad = '\239\191\189'
    for _, id in ipairs(ids) do
        current[#current + 1] = id
        local decoded = loom.detokenize(current)
        local at = decoded:find(bad, 1, true)
        if at == nil or full:sub(offset + at, offset + at + 2) == bad then
            local special = current[1] >= W_EOT
            local with_space = decoded:sub(1, 1) == ' '
            local stripped = decoded:match('^%s*(.-)%s*$')
            local punctuation = #stripped == 1 and stripped:match('^%p$') ~= nil
            if special or with_space or punctuation or #words == 0 then
                words[#words + 1] = decoded
                lengths[#lengths + 1] = #current
            else
                words[#words] = words[#words] .. decoded
                lengths[#lengths] = lengths[#lengths] + #current
            end
            offset = offset + #decoded
            current = {}
        end
    end
    return words, lengths
end

-- `trim_context`: drop the context's first words while it, or it with the forced tokens, is too long.
local function ws_trim_context(n_forced)
    local ctx = loom.state('ctx')
    local c = #ctx
    local l = n_forced + c
    while c > W_N_TEXT_CTX or l > W_N_TEXT_CTX - 20 do
        local words, lengths = ws_word_lengths(ctx)
        if #words == 0 then break end
        local t = lengths[1]
        ctx = loom.tokenize(table.concat(words, '', 2))
        l, c = l - t, c - t
        if t == 0 then break end
    end
    loom.set_state('ctx', ctx)
end

-- The 7-wide median along the frames, reflect-padded (`whisper.timing.median_filter`).
local function ws_median7(z, n)
    local out, w = {}, {}
    for f = 1, n do
        for j = -3, 3 do
            local i = f + j
            if i < 1 then i = 2 - i elseif i > n then i = 2 * n - i end
            w[j + 4] = z[i]
        end
        for a = 2, 7 do
            local v, b = w[a], a - 1
            while b >= 1 and w[b] > v do w[b + 1] = w[b]; b = b - 1 end
            w[b + 1] = v
        end
        out[f] = w[4]
    end
    return out
end

-- One `infer`: the buffer encoded, the forced tokens re-read, then decoded; returns the new tokens.
local function ws_infer(is_last)
    local seg_lens = loom.state('seg_lens')
    if #seg_lens == 0 then return {} end
    local n = 0
    for _, s in ipairs(seg_lens) do n = n + s end
    local wav = loom.state_slice('pcm', 0, n)
    for i = n + 1, W_N_SAMPLES do wav[i] = 0.0 end
    loom.run_subgraph_and_retain('encoder', {n_samples = W_N_SAMPLES, n_past = 0}, {waveform = wav})
    loom.run_subgraph_and_retain('cross_kv', {n_enc_frames = W_N_AUDIO_CTX, n_past = 0}, {xa = {from = 'encoder'}})
    -- `int((mel_padded frames - N_FRAMES) / 2)`: the encoder frames the audio itself covers.
    local content = math.floor(math.floor(n / W_HOP) / 2)

    -- The forced tokens: committed blocks after the start of transcript, the context before it.
    local initial = ws_initial()
    local blocks, block_lens = loom.state('blocks'), loom.state('block_lens')
    ws_trim_context(#initial + #blocks)
    local current = {}
    local ctx = loom.state('ctx')
    if #ctx > 0 then
        current[1] = W_SOT_PREV
        for _, id in ipairs(ctx) do current[#current + 1] = id end
    end
    for _, id in ipairs(initial) do current[#current + 1] = id end
    for _, id in ipairs(blocks) do current[#current + 1] = id end
    local before = #current
    loom.set_state('start_tokens', current)

    local S, H = W_N_AUDIO_CTX, W_N_ALIGN
    local sums, sumsq, rows = {}, {}, 0
    for i = 1, H * S do sums[i] = 0.0; sumsq[i] = 0.0 end
    local last_attend = loom.state('last_attend')
    local frames = {}
    local first = true
    local threshold = is_last and 4 or loom.state('threshold')
    local rewind = loom.state('rewind')
    while #current < W_N_TEXT_CTX do
        local feed, n_past = current, 0
        if not first then feed, n_past = {current[#current]}, #current - 1 end
        ws_forward(feed, n_past)
        local banned = W_SUPPRESS
        if first then
            banned = {}
            for _, id in ipairs(W_SUPPRESS) do banned[#banned + 1] = id end
            for _, id in ipairs(W_SPACE) do banned[#banned + 1] = id end
            banned[#banned + 1] = W_EOT
        end
        first = false
        local token = loom.sample_row('decoder', #feed - 1, {banned = banned})
        current[#current + 1] = token

        -- The alignment heads over every token so far: running sums per head and frame, then the newest row.
        -- Its rows from the output's own shape: on the first step `feed` IS `current`, which has just grown.
        local att = loom.get_output('decoder', 2)
        local t_new = loom.output_shape('decoder', 2)[2]
        for h = 0, H - 1 do
            for t = 0, t_new - 1 do
                local base = (h * t_new + t) * S
                local acc = h * S
                for f = 1, S do
                    local v = att[base + f]
                    sums[acc + f] = sums[acc + f] + v
                    sumsq[acc + f] = sumsq[acc + f] + v * v
                end
            end
        end
        rows = rows + t_new
        local mean = {}
        for f = 1, S do mean[f] = 0.0 end
        for h = 0, H - 1 do
            local z = {}
            local base = (h * t_new + t_new - 1) * S
            for f = 1, S do
                local m = sums[h * S + f] / rows
                local var = sumsq[h * S + f] / rows - m * m
                if var < 0 then var = 0 end
                z[f] = (att[base + f] - m) / math.sqrt(var)
            end
            local filtered = ws_median7(z, S)
            for f = 1, S do mean[f] = mean[f] + filtered[f] end
        end
        local most, best = 0, -math.huge
        for f = 1, math.min(content, S) do
            local v = mean[f] / H
            if v > best then best, most = v, f - 1 end
        end
        frames[#frames + 1] = most

        if token == W_EOT then
            current[#current] = nil
            break
        end
        if not is_last and last_attend - most > rewind then
            if #current > 1 and current[#current - 1] >= W_DEC_PAD then
                last_attend = most
            else
                -- A rewind: this update is dropped. `torch.cat(self.tokens)` -- the start of transcript
                -- and the committed blocks, without the context -- sliced as if it had it.
                last_attend = -rewind
                current = {}
                for _, id in ipairs(initial) do current[#current + 1] = id end
                for _, id in ipairs(blocks) do current[#current + 1] = id end
                break
            end
        else
            last_attend = most
        end
        if content - most <= threshold then
            current[#current] = nil
            break
        end
    end
    loom.set_state('last_attend', last_attend)
    loom.set_state('frames', frames)
    local new = {}
    for i = before + 1, #current do new[#new + 1] = current[i] end
    for _, id in ipairs(new) do blocks[#blocks + 1] = id end
    block_lens[#block_lens + 1] = #new
    loom.set_state('blocks', blocks)
    loom.set_state('block_lens', block_lens)
    return new
end

-- `insert_audio`: a new segment (`n` samples, already in `pcm`), then the 30 s limit.
local function ws_insert(n)
    local seg_lens = loom.state('seg_lens')
    if n > 0 then seg_lens[#seg_lens + 1] = n end
    local total = 0
    for _, s in ipairs(seg_lens) do total = total + s end
    while #seg_lens > 1 and total / W_RATE > W_MAX_SECONDS do
        local removed = seg_lens[1]
        total = total - removed
        -- `int(TOKENS_PER_SECOND * removed_len)`, truncated toward zero.
        local shift = 50 * (removed / W_RATE)
        loom.set_state('last_attend', loom.state('last_attend') - (shift >= 0 and math.floor(shift) or math.ceil(shift)))
        table.remove(seg_lens, 1)
        loom.drop_state('pcm', removed)
        local blocks, block_lens = loom.state('blocks'), loom.state('block_lens')
        if #block_lens > 0 then
            -- `context.append_token_ids(self.tokens[1])`: its text added to the context's.
            local first = {}
            for i = 1, block_lens[1] do first[i] = blocks[i] end
            local ctx = loom.state('ctx')
            local text = (#ctx > 0 and loom.detokenize(ctx) or '') .. (#first > 0 and loom.detokenize(first) or '')
            loom.set_state('ctx', #text > 0 and loom.tokenize(text) or {})
            local rest = {}
            for i = block_lens[1] + 1, #blocks do rest[#rest + 1] = blocks[i] end
            table.remove(block_lens, 1)
            loom.set_state('blocks', rest)
            loom.set_state('block_lens', block_lens)
        end
    end
    loom.set_state('seg_lens', seg_lens)
end

function stream_open(inputs)
    local chunk_s = inputs.chunk_seconds or W_DEFAULT_CHUNK_SECONDS
    local chunk = math.floor(chunk_s * W_RATE)
    if chunk < 1 or chunk_s >= W_MAX_SECONDS then
        error('whisper stream: chunk_seconds is more than 0 and less than the 30 s buffer')
    end
    local language = inputs.language
    if W_LANG_HI > 0 then
        language = language or W_LANG_DEFAULT
        if language < W_LANG_LO or language >= W_LANG_HI then
            error('whisper stream: language must be a language token id')
        end
    elseif language ~= nil then
        error('whisper stream: this checkpoint is English-only and takes no language')
    end
    loom.set_state('chunk', chunk)
    loom.set_state('threshold', inputs.frame_threshold or W_FRAME_THRESHOLD)
    -- `AlignAttConfig.rewind_threshold` (200 frames; "higher value turns rewinds off").
    loom.set_state('rewind', inputs.rewind_threshold or W_REWIND)
    loom.set_state('language', language or 0)
    loom.set_state('pcm', {})
    loom.set_state('pending', 0)
    loom.set_state('seg_lens', {})
    loom.set_state('blocks', {})
    loom.set_state('block_lens', {})
    loom.set_state('ctx', {})
    loom.set_state('last_attend', -loom.state('rewind'))
    loom.set_state('start_tokens', {})
    loom.set_state('frames', {})
    return {}
end

function stream_feed(inputs)
    loom.append_state('pcm', inputs.waveform)
    local pending = loom.state('pending') + #inputs.waveform
    local chunk = loom.state('chunk')
    local out = {}
    while pending >= chunk do
        ws_insert(chunk)
        pending = pending - chunk
        loom.set_state('pending', pending)
        for _, id in ipairs(ws_infer(false)) do out[#out + 1] = id end
    end
    loom.set_state('pending', pending)
    return out
end

function stream_finish(inputs)
    local out = {}
    local pending = loom.state('pending')
    if pending > 0 then
        ws_insert(pending)
        loom.set_state('pending', 0)
        for _, id in ipairs(ws_infer(false)) do out[#out + 1] = id end
    end
    ws_insert(0)
    for _, id in ipairs(ws_infer(true)) do out[#out + 1] = id end
    return out
end
