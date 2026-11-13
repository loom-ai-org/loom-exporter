-- ===== SpeechT5, live (loom.cpp ADR-100, ADR-119): a few mel frames of audio per `stream_feed`. =====
--
-- The decoder loop makes two mel frames a step; the postnet (five convolutions of kernel 5) and HiFi-GAN
-- then turn the spectrogram into audio, and both are LOCAL: a sample of frame t reads frames
-- `[t - MEL_LEFT_FRAMES, t + MEL_RIGHT_FRAMES]` and nothing further (`mel_context_frames` walks the
-- layers; 34 either side for `microsoft/speecht5_hifigan`). So a piece of frames `[e, f)` is the postnet
-- and the vocoder over frames `[e - MEL_LEFT_FRAMES, f + MEL_RIGHT_FRAMES)` with the samples of `[e, f)`
-- kept, once the loop has made that many frames. The pieces joined are the one-pass waveform.
--
-- Two cuts, by `inputs.piece_frames`:
--
--   * frames: that many mel frames' worth of audio (HOP_LENGTH samples, 16 ms, each) per feed. A stream's
--     default (absent, or -1) GROWS: 2 frames, then 4, 8, ... up to PIECE_FRAMES_GROWN (256), so the first
--     audio is ready after `(2 + MEL_RIGHT_FRAMES) / 2` decoder steps. `>= 1` fixes the count. The loop's
--     end closes the piece early and flushes the frames the margin held back.
--   * 0 (what `infer` asks for): the whole loop, then the postnet and the vocoder over every frame in one
--     call each -- the one pass as it always ran, bit for bit.
--
-- The functions hold nothing between calls. Everything a later feed reads is the stream's state: the
-- encoder's cross-attention K/V (kept on the device, `loom.keep_state`, since another stream's call would
-- overwrite a retained output), the speaker, the stop rule's numbers, pinned masks or teacher frames when
-- the caller gave them, and the loop's own variables: the step, the frame fed back next, the frames still
-- inside some later piece's window (from `frame_base`), how many there are, how many have been handed out
-- as audio, and whether the loop has stopped. The decoder's self-attention history is the model's KV
-- cache, which a stream leases or owns (ADR-099). The prenet's dropout masks are drawn per step in the
-- one pass's order from the stream's own generator, so a seeded stream draws the one pass's masks.

local SPEECHT5_LOOP = {'step', 'frame', 'frames', 'frame_base', 'n_frames', 'emitted', 'ended'}

local function speecht5_load()
    local st = {}
    for _, name in ipairs(SPEECHT5_LOOP) do st[name] = loom.state(name) end
    return st
end

local function speecht5_save(st)
    for _, name in ipairs(SPEECHT5_LOOP) do loom.set_state(name, st[name]) end
end

-- One decoder step: the frame fed back in, two frames and two stop logits out. Sets `ended` on the step
-- the one pass would break after; that step keeps its frames.
local function speecht5_step(st)
    local step = st.step
    local masks = loom.state('has_masks') == 1 and loom.state('masks') or nil
    local call = {
        frame = st.frame, position_ids = {step},
        prenet_mask_0 = speecht5_draw_mask(PRENET_UNITS, masks, 2 * step),
        prenet_mask_1 = speecht5_draw_mask(PRENET_UNITS, masks, 2 * step + 1),
        speaker = loom.state('speaker'), attention_mask = loom.causal_mask(1, step),
    }
    for l = 0, N_LAYERS - 1 do
        call['xk_' .. l] = {state = 'cross_kv', index = 2 * l + 1}
        call['xv_' .. l] = {state = 'cross_kv', index = 2 * l + 2}
    end
    loom.run_subgraph_and_retain('decoder', {n_tokens = 1, n_past = step, n_enc_frames = loom.state('n_enc')},
                                 call)
    local spectrum = loom.get_output('decoder', 1)
    local logits = loom.get_output('decoder', 2)
    local step_len = REDUCTION_FACTOR * NUM_MEL_BINS
    local base = (st.n_frames - st.frame_base) * NUM_MEL_BINS
    for i = 1, step_len do st.frames[base + i] = spectrum[i] end
    st.n_frames = st.n_frames + REDUCTION_FACTOR
    -- The next input is the LAST frame of the pair, or the caller's frame at the same place.
    local src, off = spectrum, step_len - NUM_MEL_BINS
    if loom.state('has_teacher') == 1 then
        local teacher = loom.state('teacher_frames')
        if teacher[st.n_frames * NUM_MEL_BINS] ~= nil then src, off = teacher, (st.n_frames - 1) * NUM_MEL_BINS end
    end
    st.frame = {}
    for i = 1, NUM_MEL_BINS do st.frame[i] = src[off + i] end
    -- `sum(sigmoid(prob_out)) >= threshold`, tested from step `minlen` on; at `maxlen` it stops
    -- regardless.
    local p = 0.0
    for r = 1, REDUCTION_FACTOR do p = p + 1.0 / (1.0 + math.exp(-logits[r])) end
    st.step = step + 1
    if st.step >= loom.state('minlen') and (st.step >= loom.state('maxlen') or p >= loom.state('threshold')) then
        st.ended = 1
    end
end

-- Frames `[a, b)` through the postnet and the vocoder (`HOP_LENGTH * (b - a)` samples), and the samples
-- of frames `[e, f)` kept: those whose receptive field the window holds whole.
local function speecht5_decode(st, a, b, e, f)
    local spectrogram = {}
    local from = (a - st.frame_base) * NUM_MEL_BINS
    for i = 1, (b - a) * NUM_MEL_BINS do spectrogram[i] = st.frames[from + i] end
    loom.run_subgraph_and_retain('postnet', {n_enc_frames = b - a, n_past = 0}, {spectrogram = spectrogram})
    local wave = loom.run_subgraph('vocoder', {n_enc_frames = b - a, n_past = 0}, {mel = {from = 'postnet'}})
    if e == a and f == b then return wave end
    local out = {}
    local skip = (e - a) * HOP_LENGTH
    for i = 1, (f - e) * HOP_LENGTH do out[i] = wave[skip + i] end
    return out
end

-- Drops the frames no later window reaches: the next piece starts at `emitted`, its window
-- MEL_LEFT_FRAMES before that.
local function speecht5_trim(st)
    local keep = math.max(0, st.emitted - MEL_LEFT_FRAMES)
    if keep <= st.frame_base then return end
    local drop = (keep - st.frame_base) * NUM_MEL_BINS
    local frames = {}
    for i = 1, (st.n_frames - keep) * NUM_MEL_BINS do frames[i] = st.frames[drop + i] end
    st.frames, st.frame_base = frames, keep
end

function stream_open(inputs)
    -- `inputs.tokens` are the SentencePiece character ids with `</s>` appended. Optional knobs, with
    -- `_generate_speech`'s defaults: `threshold`, `minlenratio`, `maxlenratio`; `speaker` (a 512-float
    -- x-vector, un-normalised) replaces the built-in voice; `seed` seeds the prenet's dropout draws, and
    -- `masks` pins them outright. `teacher_frames` (one `[80]` row per frame, row-major) feeds a known
    -- spectrogram back in place of the loop's own frames, which is how a gate compares steps without
    -- the feedback compounding rounding. `piece_frames`: see the top of this file.
    local ids = inputs.tokens
    local n = #ids
    if n == 0 then error('speecht5: no text ids') end
    if n > MAX_TEXT_POSITIONS then
        error(string.format('speecht5: %d ids exceed the encoder\'s %d positions', n, MAX_TEXT_POSITIONS))
    end
    local piece_frames = inputs.piece_frames or -1
    if piece_frames < -1 then
        error('speecht5: piece_frames is a mel frame count, 0 for the whole utterance, or -1 for the growing default')
    end
    if inputs.seed then loom.seed_rng(inputs.seed) end

    loom.run_subgraph_and_retain('encoder', {n_tokens = n, n_past = 0},
        {input_ids = ids, position_ids = loom.range(0, n),
         rel_index = speecht5_relative_index(n, MAX_RELATIVE_POSITION)})
    loom.run_subgraph_and_retain('cross_kv', {n_enc_frames = n, n_past = 0}, {xa = {from = 'encoder'}})
    loom.keep_state('cross_kv', 'cross_kv')

    -- `int(n * ratio / reduction_factor)`, and never past the decoder's position table: the reference
    -- would fail there on a shape, one step in.
    local maxlen = math.floor(n * (inputs.maxlenratio or DEFAULT_MAXLENRATIO) / REDUCTION_FACTOR)
    local minlen = math.floor(n * (inputs.minlenratio or DEFAULT_MINLENRATIO) / REDUCTION_FACTOR)
    if maxlen > MAX_SPEECH_POSITIONS then maxlen = MAX_SPEECH_POSITIONS end
    if maxlen < 1 then maxlen = 1 end
    loom.set_state('n_enc', n)
    loom.set_state('maxlen', maxlen)
    loom.set_state('minlen', minlen)
    loom.set_state('threshold', inputs.threshold or DEFAULT_THRESHOLD)
    loom.set_state('speaker', inputs.speaker or loom.get_weight('decoder', 'speaker'))
    -- A state value is numbers, never nil, so an absent option is a flag beside it.
    loom.set_state('has_masks', inputs.masks and 1 or 0)
    if inputs.masks then loom.set_state('masks', inputs.masks) end
    loom.set_state('has_teacher', inputs.teacher_frames and 1 or 0)
    if inputs.teacher_frames then loom.set_state('teacher_frames', inputs.teacher_frames) end
    loom.set_state('piece_frames', piece_frames)

    local frame = {}
    for i = 1, NUM_MEL_BINS do frame[i] = 0.0 end
    speecht5_save({step = 0, frame = frame, frames = {}, frame_base = 0, n_frames = 0, emitted = 0, ended = 0})
    loom.set_state('piece', 0)
    loom.set_state('n_pieces', piece_frames == 0 and 1 or 0)
    loom.set_state('done', 0)
    return {}
end

function stream_feed(inputs)
    if loom.state('done') ~= 0 then return {} end
    local st = speecht5_load()
    local piece_frames = loom.state('piece_frames')
    local wave = {}
    if piece_frames == 0 then
        -- The whole utterance: every step, then the postnet and the vocoder over every frame in one call.
        while st.ended == 0 do speecht5_step(st) end
        wave = speecht5_decode(st, 0, st.n_frames, 0, st.n_frames)
        loom.set_state('done', 1)
    else
        -- Frames `[e, e + want)` once MEL_RIGHT_FRAMES more are made past them; the loop's end closes the
        -- piece early, with every frame left.
        local want = piece_frames
        if want < 0 then want = math.min(PIECE_FRAMES_GROWN, 2 ^ (loom.state('piece') + 1)) end
        local e = st.emitted
        while st.ended == 0 and st.n_frames < e + want + MEL_RIGHT_FRAMES do speecht5_step(st) end
        local a = math.max(0, e - MEL_LEFT_FRAMES)
        if st.ended == 1 then
            -- The last piece: every frame the loop made from `e` on, at most `want + MEL_RIGHT_FRAMES + 1`.
            if st.n_frames > e then wave = speecht5_decode(st, a, st.n_frames, e, st.n_frames) end
            st.emitted = st.n_frames
            loom.set_state('done', 1)
        else
            local f = e + want
            wave = speecht5_decode(st, a, f + MEL_RIGHT_FRAMES, e, f)
            st.emitted = f
            speecht5_trim(st)
        end
    end
    speecht5_save(st)
    loom.set_state('piece', loom.state('piece') + 1)
    return wave
end

function stream_finish(inputs)
    return {}
end
