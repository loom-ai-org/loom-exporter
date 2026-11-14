-- ===== Qwen3-ASR's streaming mode, live (loom.cpp ADR-121): upstream's re-feed loop, recomputing only what changed. =====
--
-- Upstream's `streaming_transcribe` (qwen-asr 0.0.6, vLLM backend only) streams a model trained offline by
-- decoding again as audio arrives. Its loop, which this is, line for line:
--   * every S_UPDATE_SECONDS of audio (2 s), ALL the audio heard so far is re-fed to the whole model;
--   * from update S_UNFIXED_CHUNKS on (the third), the previous answer minus its last S_UNFIXED_TOKENS ids
--     (5) is appended to the prompt -- the assistant turn already started -- so the model only re-decides
--     the tail and the transcript stops flickering;
--   * `finish` re-feeds whatever arrived after the last whole update the same way (keeping at least one
--     id of the answer); with nothing new it changes nothing.
-- The transcript at the end depends on the cadence: it is upstream's streamed transcript, not `infer`'s
-- one-pass one.
--
-- Upstream holds the answer as TEXT and cuts the prefix from its re-encoding, so this does too, through the
-- file's own vocabulary (`loom.tokenize` / `loom.detokenize`, the engine's). Cutting the model's own ids
-- instead was the first version: it agreed on 13 clips of Qwen3-ASR and broke on oruk-loop, whose
-- non-canonical spelling of a name put other ids in the prompt and changed the next word (`prefix_of`).
--
-- The answer is `language English<asr_text>...` for the stock checkpoint and the text alone for a file
-- whose prompt forces the language (oruk-loop): the transcript is what follows S_TRANSCRIPT_AFTER when the
-- answer holds it, and all of it otherwise -- `infer`'s rule. The answer itself (`raw`, preamble included)
-- is what the next prefix is cut from, as upstream's `_raw_decoded` is.
--
-- Upstream re-encodes everything; this re-encodes the encoder windows that changed (`decode`), which gives the
-- same rows, and keeps the decoder's keys over the rest. The stream is bounded by that cache (S_N_CTX), not by
-- the encoder: past it the stream is refused by name rather than cut. Split a long stream at a pause.
--
-- State: `every` (samples per update), `unfixed_chunks`, `unfixed_tokens`, `max_new`; `audio`, the samples
-- of every whole update so far; `pending`, the samples since; `updates`, how many decodes ran; `raw`, the
-- last answer's ids; `hypothesis`, its transcript ids; `kept`, the encoder windows whose decoder keys are in
-- the cache, with each one's raw log-mel max and min and the floor it was encoded at (`w_max`, `w_min`,
-- `w_floor`).

local function append(dst, src)
    for i = 1, #src do dst[#dst + 1] = src[i] end
end

local function contains(list, value)
    for i = 1, #list do
        if list[i] == value then return true end
    end
    return false
end

local function clip_sample(audio, n_real, n_padded, pos)
    -- Sample `pos` (0-based) of what the whole-clip encoder reads: `transcribe`'s padded clip (zeros to a
    -- whole chunk, the frontend's reflection over their head) inside `center=True`'s reflection of it.
    if pos < 0 then return clip_sample(audio, n_real, n_padded, -pos) end
    if pos >= n_padded then return clip_sample(audio, n_real, n_padded, 2 * (n_padded - 1) - pos) end
    if pos < n_real then return audio[pos + 1] end
    if pos - n_real < math.min(S_REFLECT, n_real - 1) then return audio[2 * n_real - 1 - pos] end
    return 0.0
end

local function window_context(audio, n_real, n_padded, w)
    -- Window `w`'s samples with S_REFLECT either side, and how many of its own are real.
    local first = w * S_WINDOW_SAMPLES
    local len = math.min(S_WINDOW_SAMPLES, n_padded - first)
    local ctx = {}
    for pos = first - S_REFLECT, first + len + S_REFLECT - 1 do
        ctx[#ctx + 1] = clip_sample(audio, n_real, n_padded, pos)
    end
    return ctx, math.min(S_WINDOW_SAMPLES, n_real - first)
end

local function decode(audio, prefix)
    -- One upstream decode -- the prompt, all of `audio` as its rows, the assistant turn started with
    -- `prefix`, greedy to a stop id or the budget -- with only what changed recomputed. Returns the
    -- answer, `prefix` included, the stop id not.
    --
    -- The encoder attends inside fixed 8 s windows counted from the clip's start, so a window's rows depend
    -- on its own frames and on the log-mel floor (`max - 8` over every frame so far). A window is FINISHED
    -- once S_REFLECT samples past its end have arrived (its last frame reads 40 past it). A finished
    -- window's rows, and the decoder's keys over them, are kept until the floor rises above its lowest
    -- frame; the first window that is unfinished or below the floor is where encoding and prefill restart.
    -- So an update re-encodes one window and prefills its rows, where upstream re-feeds the whole stream.
    local n_real = #audio
    local n_padded = math.ceil(n_real / S_SAMPLES_PER_CHUNK) * S_SAMPLES_PER_CHUNK
    local n_windows = math.ceil(n_padded / S_WINDOW_SAMPLES)
    local kept = loom.state('kept')  -- windows whose keys are in the cache
    local w_max, w_min, w_floor = loom.state('w_max'), loom.state('w_min'), loom.state('w_floor')

    -- The floor: every finished window's max is kept; the rest are measured now.
    local ctx, valid, top = {}, {}, -math.huge
    for w = 0, kept - 1 do top = math.max(top, w_max[w + 1]) end
    for w = kept, n_windows - 1 do
        ctx[w], valid[w] = window_context(audio, n_real, n_padded, w)
        local range = loom.run_subgraph('window_mel_range', {n_samples = #ctx[w]}, {waveform = ctx[w]})
        w_max[w + 1], w_min[w + 1] = range[1], range[2]
        top = math.max(top, range[1])
    end
    local floor = top - 8.0
    -- A kept window encoded at another floor, with a frame under either: it and everything after it go.
    -- Either, because the floor can FALL: the max counts the clip end's padding and reflection frames,
    -- which real audio replaces.
    for w = 0, kept - 1 do
        if w_floor[w + 1] ~= floor and w_min[w + 1] < math.max(floor, w_floor[w + 1]) then
            for v = w, kept - 1 do ctx[v], valid[v] = window_context(audio, n_real, n_padded, v) end
            kept = w
            break
        end
    end

    local n_past = #S_AUDIO_PREFIX
    if kept == 0 then
        loom.run_subgraph_and_retain('embed', {n_tokens = n_past, n_past = 0}, {tokens = S_AUDIO_PREFIX})
        loom.run_subgraph_and_retain('decoder', {n_tokens = n_past, n_past = 0},
            {inputs_embeds = {from = 'embed'}, position_ids = loom.range(0, n_past),
             attention_mask = loom.causal_mask(n_past, 0)})
    end
    n_past = n_past + kept * S_WINDOW_ROWS
    for w = kept, n_windows - 1 do
        loom.run_subgraph_and_retain('encoder_window', {n_samples = #ctx[w], n_past = 0},
                                     {waveform = ctx[w], valid_samples = {valid[w]}, floor = {floor}})
        local rows = S_audio_rows(valid[w])
        loom.run_subgraph_and_retain('decoder', {n_tokens = rows, n_past = n_past},
            {inputs_embeds = {from = 'encoder_window', rows = rows}, position_ids = loom.range(n_past, rows),
             attention_mask = loom.causal_mask(rows, n_past)})
        n_past = n_past + rows
        w_floor[w + 1] = floor
        -- Kept only while every window before it is: the keys are a prefix of the cache.
        if kept == w and (w + 1) * S_WINDOW_SAMPLES + S_REFLECT <= n_real then kept = w + 1 end
    end
    for w = #w_max, kept + 1, -1 do w_max[w], w_min[w], w_floor[w] = nil, nil, nil end
    loom.set_state('kept', kept)
    loom.set_state('w_max', w_max)
    loom.set_state('w_min', w_min)
    loom.set_state('w_floor', w_floor)

    local step = {}
    append(step, S_AUDIO_SUFFIX)
    append(step, prefix)
    local max_new = loom.state('max_new')
    if n_past + #step + max_new > S_N_CTX then
        error(string.format("qwen3-asr stream: %.1f s of audio and its transcript have outgrown the decoder's "
            .. "cache (%d positions); the streaming mode keeps all of them. Split the stream at a pause (a VAD "
            .. "finds them).", n_real / S_SAMPLE_RATE, S_N_CTX))
    end
    local answer = {}
    append(answer, prefix)
    local generated = 0
    while true do
        local n_tokens = #step
        loom.run_subgraph_and_retain('embed', {n_tokens = n_tokens, n_past = 0}, {tokens = step})
        loom.run_subgraph_and_retain('decoder', {n_tokens = n_tokens, n_past = n_past},
            {inputs_embeds = {from = 'embed'}, position_ids = loom.range(n_past, n_tokens),
             attention_mask = loom.causal_mask(n_tokens, n_past)})
        loom.run_subgraph_and_retain('lm_head', {n_tokens = n_tokens, n_past = 0}, {hidden = {from = 'decoder'}})
        local token = loom.argmax_row('lm_head', n_tokens - 1)
        generated = generated + 1
        n_past = n_past + n_tokens
        if contains(S_EOS, token) then break end
        answer[#answer + 1] = token
        if generated >= max_new then break end
        step = {token}
    end
    return answer
end

local function transcript(answer)
    for i = 1, #answer do
        if answer[i] == S_TRANSCRIPT_AFTER then
            local kept = {}
            for j = i + 1, #answer do kept[#kept + 1] = answer[j] end
            return kept
        end
    end
    return answer
end

local function whole_characters(text)
    -- Upstream's `'�' not in tok.decode(ids)`: its decode replaces an incomplete or invalid UTF-8
    -- sequence with U+FFFD, `loom.detokenize` hands back the bytes, so both cases are looked for here.
    if string.find(text, "\239\191\189", 1, true) then return false end
    local i, n = 1, #text
    while i <= n do
        local c = string.byte(text, i)
        local len = (c < 0x80 and 1) or (c >= 0xF8 and 0) or (c >= 0xF0 and 4) or (c >= 0xE0 and 3)
                    or (c >= 0xC0 and 2) or 0
        if len == 0 or i + len - 1 > n then return false end
        for j = i + 1, i + len - 1 do
            local d = string.byte(text, j)
            if d < 0x80 or d >= 0xC0 then return false end
        end
        i = i + len
    end
    return true
end

local function prefix_of(raw, final)
    -- Upstream's rule on TEXT: re-encode the answer, drop its last `unfixed_tokens` ids, decode, and back
    -- off further while that cuts a character; the prompt then holds that text's own ids. Re-encoding is what
    -- makes it upstream's: a name the model spelled with other ids than its tokenizer would (oruk-loop wrote
    -- " Voxtrol" as three ids where the tokenizer has two) comes back in the tokenizer's spelling, and the
    -- model continues from that.
    --
    -- `final` keeps at least one id, as `finish_streaming_transcribe` does, but backs off too, which
    -- upstream's finish does NOT: it cut 편 in half on a Korean clip, put a literal U+FFFD in the prompt,
    -- and the transcript ended "훨씬 �연할 거야". The one place this stream departs from upstream on purpose.
    local cur = loom.tokenize(loom.detokenize(raw))
    local stop = #cur - loom.state('unfixed_tokens')
    if final then stop = math.max(1, stop) end
    stop = math.min(stop, #cur)
    while stop > 0 do
        local head = {}
        for i = 1, stop do head[i] = cur[i] end
        local text = loom.detokenize(head)
        if whole_characters(text) then return loom.tokenize(text) end
        stop = stop - 1
    end
    return {}
end

local function update(audio, final)
    -- Upstream's prefix rule, then one decode.
    local updates = loom.state('updates')
    local prefix = {}
    if updates >= loom.state('unfixed_chunks') then
        prefix = prefix_of(loom.state('raw'), final)
    end
    local answer = decode(audio, prefix)
    loom.set_state('raw', answer)
    loom.set_state('hypothesis', transcript(answer))
    loom.set_state('updates', updates + 1)
end

function stream_open(inputs)
    local seconds = inputs.update_seconds or S_UPDATE_SECONDS
    if seconds <= 0 then error("qwen3-asr stream: update_seconds must be > 0") end
    loom.set_state('every', math.max(1, math.floor(seconds * S_SAMPLE_RATE + 0.5)))
    loom.set_state('unfixed_chunks', inputs.unfixed_chunks or S_UNFIXED_CHUNKS)
    loom.set_state('unfixed_tokens', inputs.unfixed_tokens or S_UNFIXED_TOKENS)
    loom.set_state('max_new', inputs.max_new_tokens or S_MAX_NEW_TOKENS)
    loom.set_state('audio', {})
    loom.set_state('pending', {})
    loom.set_state('updates', 0)
    loom.set_state('raw', {})
    loom.set_state('hypothesis', {})
    loom.set_state('kept', 0)
    loom.set_state('w_max', {})
    loom.set_state('w_min', {})
    loom.set_state('w_floor', {})
    return {}
end

function stream_feed(inputs)
    local pending = loom.state('pending')
    append(pending, inputs.waveform or {})
    local every = loom.state('every')
    if #pending < every then
        loom.set_state('pending', pending)
        return {}
    end
    -- Whole updates, each decoded in turn as upstream's `while` does, however many one push completes.
    local audio = loom.state('audio')
    local at = 0
    while #pending - at >= every do
        for i = at + 1, at + every do audio[#audio + 1] = pending[i] end
        at = at + every
        update(audio, false)
    end
    local rest = {}
    for i = at + 1, #pending do rest[#rest + 1] = pending[i] end
    loom.set_state('audio', audio)
    loom.set_state('pending', rest)
    return {}
end

function stream_finish(inputs)
    local pending = loom.state('pending')
    if #pending == 0 then return {} end
    local audio = loom.state('audio')
    append(audio, pending)
    loom.set_state('audio', audio)
    loom.set_state('pending', {})
    update(audio, true)
    return {}
end
