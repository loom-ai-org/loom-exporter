-- ===== Moonshine Streaming, live (loom.cpp ADR-106): the encoder a row at a time, the decoder re-run. =====
--
-- The encoder is what makes this checkpoint a streaming one: per-frame CMVN, causal convolutions and attention
-- bands with no position inside them, so its rows are translation invariant up to the position table at the
-- end. A window over the clip's samples from a row boundary, with `row_offset` naming that row, therefore
-- gives the clip's own rows wherever its left context covers the receptive field (LEFT_ROWS) and the row is
-- RIGHT_ROWS from the window's right edge -- the bands' look-ahead, 240 ms. Measured at f64: 1e-15 from one
-- call on both checkpoints.
--
-- The decoder cross-attends to every row, so a row arriving changes what every token sees: there is no prefix
-- to keep. It is re-run over all the final rows every DECODE_EVERY rows (0.5 s; `stream_open`'s
-- `update_seconds` sets another cadence -- the cost knob, which never changes the transcript at the end) and
-- at the end, and the transcript is
-- the latest one -- `hypothesis` in the stream's state, which the transcript door reads as the whole
-- transcript rather than appending (`revised` when it changed earlier text). The card's token budget is the
-- one-pass rule over the samples heard so far.
--
-- Two things keep the re-runs affordable, both measured on the dev box (tiny, jfk, 100 ms pushes; 0.88x real
-- time without them, 2.3x with):
--   * rows are encoded only when a decode needs them, so a window's 93-row left context is paid once per
--     DECODE_EVERY rows rather than once per push (each push re-encoding its window was 22x the one pass);
--   * a mid-stream decode teacher-forces the last hypothesis in one call and steps on from the first token
--     whose argmax differs -- greedy's own answer, from one call where it agrees, rather than a call a token.
-- `stream_finish` decodes the plain way, token by token from `<s>` over every row, which is `infer`'s loop
-- over the same rows: the stream ends on one call's transcript.
--
-- State: `every`, the decode cadence in rows; `audio`, the samples from row `audio_row` on (the left context the next window reaches back to, then
-- whatever has not made a row final yet); `rows`, every final row, D_MODEL each; `heard`, samples pushed;
-- `decoded`, the row count the last decode saw; `hypothesis`, its ids.

local ROW = 4 * S_FRAME  -- samples per encoder row: two stride-2 convolutions

local function append(dst, src)
    for i = 1, #src do dst[#dst + 1] = src[i] end
end

local function encode(window, first_row, keep_from, keep_to)
    -- The encoder over `window` (whole frames, starting at row `first_row`); rows [keep_from, keep_to) of it.
    local out = loom.run_subgraph('encoder', {n_samples = #window, n_past = 0},
                                  {waveform = window, row_offset = {first_row}})
    local rows = {}
    for i = (keep_from - first_row) * D_MODEL + 1, (keep_to - first_row) * D_MODEL do rows[#rows + 1] = out[i] end
    return rows
end

local function decode(n_real, verify)
    -- Greedy from `<s>` over every final row, to `</s>` or the card's budget for `n_real` samples; with
    -- `verify`, from the last hypothesis teacher-forced.
    local rows = loom.state('rows')
    local n_enc = #rows / D_MODEL
    local max_new = math.min(math.floor(n_real * S_TOKENS_PER_SECOND / S_SAMPLE_RATE), S_MAX_POSITIONS) - 1
    loom.set_state('decoded', n_enc)
    if n_enc == 0 or max_new <= 0 then
        loom.set_state('hypothesis', {})
        return
    end
    loom.run_subgraph_and_retain('cross_kv', {n_enc_frames = n_enc, n_past = 0}, {xa = rows})
    local args = {}
    for l = 0, N_LAYERS - 1 do
        args['xk_' .. l] = {from = 'cross_kv', index = 2 * l + 1}
        args['xv_' .. l] = {from = 'cross_kv', index = 2 * l + 2}
    end
    local gen, step, n_past = {}, {S_DECODER_START}, 0
    local prev = verify and loom.state('hypothesis') or {}
    if #prev > 0 then
        local forced = {S_DECODER_START}
        for i = 1, math.min(#prev, max_new - 1) do forced[#forced + 1] = prev[i] end
        args.tokens = forced
        args.position_ids = loom.range(0, #forced)
        args.attention_mask = loom.causal_mask(#forced, 0)
        loom.run_subgraph_and_retain('decoder', {n_tokens = #forced, n_past = 0, n_enc_frames = n_enc}, args)
        -- Position i's argmax is the token after forced[1..i]; greedy follows the forced tokens while they
        -- are its own choice. The cache then holds positions [0, #gen) of that path, and the next step
        -- feeds gen[#gen] at #gen.
        for i = 1, #forced do
            local token = loom.argmax_row('decoder', i - 1)
            gen[#gen + 1] = token
            if token ~= forced[i + 1] then break end
        end
        local last = gen[#gen]
        if #gen >= max_new or last == S_EOS then
            loom.set_state('hypothesis', gen)
            return
        end
        step, n_past = {last}, #gen
    end
    while true do
        args.tokens = step
        args.position_ids = loom.range(n_past, #step)
        args.attention_mask = loom.causal_mask(#step, n_past)
        loom.run_subgraph_and_retain('decoder', {n_tokens = #step, n_past = n_past, n_enc_frames = n_enc}, args)
        local token = loom.argmax_row('decoder', #step - 1)
        gen[#gen + 1] = token
        n_past = n_past + #step
        if #gen >= max_new or token == S_EOS then break end
        step = {token}
    end
    loom.set_state('hypothesis', gen)
end

local function advance(final_to, window_to, tail)
    -- Make rows [final, final_to) final, from a window ending at row `window_to` (and `tail`, a zero frame
    -- standing in for the clip's partial one, at the end); keep the samples the next window reaches back to.
    local rows = loom.state('rows')
    local final = #rows / D_MODEL
    if final_to <= final then return end
    if final_to > S_MAX_ENC_FRAMES then
        error(string.format("moonshine stream: %.1f s of audio is past the model's position table (%d encoder "
            .. "frames, %.1f s). Split the stream at a pause (a VAD finds them).", final_to * ROW / S_SAMPLE_RATE,
            S_MAX_ENC_FRAMES, S_MAX_ENC_FRAMES * ROW / S_SAMPLE_RATE))
    end
    local audio, audio_row = loom.state('audio'), loom.state('audio_row')
    local first = math.max(0, final - LEFT_ROWS)
    local window = {}
    for i = (first - audio_row) * ROW + 1, math.min(#audio, (window_to - audio_row) * ROW) do window[#window + 1] = audio[i] end
    append(window, tail or {})
    append(rows, encode(window, first, final, final_to))
    loom.set_state('rows', rows)
    local keep = math.max(0, final_to - LEFT_ROWS)
    local kept = {}
    for i = (keep - audio_row) * ROW + 1, #audio do kept[#kept + 1] = audio[i] end
    loom.set_state('audio', kept)
    loom.set_state('audio_row', keep)
end

function stream_open(inputs)
    local every = DECODE_EVERY
    if inputs.update_seconds then
        every = math.max(1, math.floor(inputs.update_seconds * S_SAMPLE_RATE / ROW + 0.5))
    end
    loom.set_state('every', every)
    loom.set_state('audio', {})
    loom.set_state('audio_row', 0)
    loom.set_state('rows', {})
    loom.set_state('heard', 0)
    loom.set_state('decoded', 0)
    loom.set_state('hypothesis', {})
    return {}
end

function stream_feed(inputs)
    local pcm = inputs.waveform or {}
    local audio = loom.state('audio')
    append(audio, pcm)
    loom.set_state('audio', audio)
    local heard = loom.state('heard') + #pcm
    loom.set_state('heard', heard)
    -- Whole rows heard; the last RIGHT_ROWS of them still wait for their look-ahead.
    local whole = math.floor(heard / ROW)
    if whole - RIGHT_ROWS - loom.state('decoded') >= loom.state('every') then
        advance(whole - RIGHT_ROWS, whole)
        decode(heard, true)
    end
    return {}
end

function stream_finish(inputs)
    -- One call's framing: whole frames as they are, the partial one a frame of zeros (01_frames.lua), and the
    -- clip's end standing in for the look-ahead past it.
    local heard = loom.state('heard')
    local audio, audio_row = loom.state('audio'), loom.state('audio_row')
    local whole_frames = math.floor(heard / S_FRAME)
    local n_frames = math.ceil(heard / S_FRAME)
    local tail = {}
    if n_frames > whole_frames then
        for i = 1, S_FRAME do tail[i] = 0.0 end
    end
    -- Drop the partial frame's samples; the window then ends at the last whole frame, plus the zero frame.
    local cut = whole_frames * S_FRAME - audio_row * ROW
    local trimmed = {}
    for i = 1, cut do trimmed[i] = audio[i] end
    loom.set_state('audio', trimmed)
    local n_rows = math.ceil(math.ceil(n_frames / 2) / 2)
    advance(n_rows, n_rows, tail)
    decode(heard)
    return {}
end
