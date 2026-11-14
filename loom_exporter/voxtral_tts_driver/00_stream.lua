-- ===== Voxtral-4B-TTS, live (loom.cpp ADR-100, ADR-116): a few frames per `stream_feed`. =====
--
-- The LM makes one frame of 37 codes per step (80 ms of 24 kHz audio), and the codec is CAUSAL with a
-- finite receptive field: the one pass already decodes in chunks that each re-read `CODEC_CONTEXT`
-- frames before them (`voxtral_decode`). So a frame's samples are final the moment it is drawn, and a
-- piece is the same chunked decode started at the first frame not yet handed out. The pieces joined are
-- the one-pass waveform to within f32 rounding (a window's attention runs over another number of rows).
--
-- Two cuts, by `inputs.piece_frames`:
--
--   * frames: that many frames drawn per feed, then decoded. A stream's default (absent, or -1) GROWS:
--     1 frame, then 2, 4 and 8 (`PIECE_FRAMES_GROWN`) from then on. `>= 1` fixes the count.
--   * 0 (what `infer` asks for): every frame, then `voxtral_decode` over all of them in one feed -- the one
--     pass as it always was, and the only cut that can hand back `return_codes`.
--
-- The loop is vllm-omni's, cut per frame. Each step feeds the previous frame back (the prompt on the first
-- step) BEFORE it draws, so the flow head always reads an LM row from a graph run in the same call.
-- Between feeds everything is the stream's state: the prompt's ids and voice rows, the knobs, pinned noise
-- and teacher frames, the step, the position, the frame not yet fed back, the codes from the first frame
-- a later chunk re-reads (`code_base` on; all of them for the one-pass cut), how many frames are audio
-- (`n_audio`, -1 until [END_AUDIO]) and how many are out. The LM's history is the model's KV cache,
-- leased or owned (ADR-099). `piece` counts feeds, `n_pieces` is 1 when the feed is the whole turn, and
-- `done` is set at [END_AUDIO] (or the teacher's last frame, or the budget).

local VOXTRAL_LOOP = {'started', 'step', 'n_past', 'pending', 'codes', 'code_base', 'n_frames', 'n_audio',
                      'emitted', 'ended'}

local function voxtral_load()
    local st = {}
    for _, name in ipairs(VOXTRAL_LOOP) do st[name] = loom.state(name) end
    return st
end

local function voxtral_save(st)
    for _, name in ipairs(VOXTRAL_LOOP) do loom.set_state(name, st[name]) end
end

-- One step: the previous frame fed back (the prompt on the first step), then one frame drawn. Returns
-- false once the loop has ended.
local function voxtral_step(st)
    if st.step >= loom.state('budget') then
        st.ended = 1
        return false
    end
    if st.started == 0 then
        local _ids = loom.state('prompt_ids')
        local _n = #_ids
        loom.run_subgraph_and_retain('embed_prompt', {n_tokens = _n, n_past = 0},
            {ids = _ids, voice = loom.state('prompt_rows'), voice_mask = loom.state('prompt_mask')})
        loom.run_subgraph_and_retain('lm', {n_tokens = _n, n_past = 0},
            {inputs_embeds = {from = 'embed_prompt'}, position_ids = loom.range(0, _n),
             attention_mask = loom.causal_mask(_n, 0)})
        st.n_past = _n
        st.started = 1
    else
        loom.run_subgraph_and_retain('embed_frame', {n_tokens = 1, n_past = 0}, {codes = st.pending})
        loom.run_subgraph_and_retain('lm', {n_tokens = 1, n_past = st.n_past},
            {inputs_embeds = {from = 'embed_frame'}, position_ids = {st.n_past},
             attention_mask = loom.causal_mask(1, st.n_past)})
        st.n_past = st.n_past + 1
    end

    -- `inputs.noise` pins the flow head's draws (one `[36]` unit normal per frame, the [END_AUDIO]
    -- frame's included, as the reference draws one there).
    local _step = st.step
    local _noise, _x = loom.state('noise'), nil
    if _noise ~= nil then
        _x = voxtral_slice(_noise, _step * N_ACOUSTIC, (_step + 1) * N_ACOUSTIC)
        if #_x < N_ACOUSTIC then error('voxtral-tts: inputs.noise ran out at frame ' .. _step) end
    else
        _x = loom.gaussian_array(N_ACOUSTIC)
    end
    loom.run_subgraph_and_retain('acoustic', {n_tokens = 1, n_past = 0},
        {hidden = {from = 'lm'}, noise = _x, cfg = {loom.state('cfg')}})
    -- The semantic code: the flow head's argmax with [EMPTY_AUDIO] and the padding rows masked, so over
    -- [END_AUDIO, 2 + 8192). Absolute, which is the code.
    local _semantic = loom.argmax_row_range('acoustic', 0, END_AUDIO, N_SEMANTIC)
    -- `decode_one_frame`'s `should_decode`: an [END_AUDIO] frame's acoustic codes are [EMPTY_AUDIO] (+2),
    -- not the integration's. Nothing decodes them; they are what the reference returns, so `return_codes`
    -- compares whole frames.
    local _acoustic = loom.get_output('acoustic', 2)
    local _frame = {_semantic}
    for _i = 1, N_ACOUSTIC do
        _frame[_i + 1] = (_semantic == END_AUDIO) and EMPTY_AUDIO_CODE or _acoustic[_i]
    end
    local _base = (st.n_frames - st.code_base) * N_CODEBOOKS
    for _i = 1, N_CODEBOOKS do st.codes[_base + _i] = _frame[_i] end
    st.n_frames = st.n_frames + 1
    st.step = _step + 1

    -- `inputs.teacher_codes` feeds the reference's frames back instead of this loop's own (Retro-055) and
    -- runs the teacher's frame count, past its [END_AUDIO].
    local _teacher = loom.state('teacher_codes')
    if _semantic == END_AUDIO and st.n_audio < 0 then
        st.n_audio = st.n_frames - 1
        if _teacher == nil then
            st.ended = 1
            return false
        end
    end
    st.pending = _teacher and voxtral_slice(_teacher, _step * N_CODEBOOKS, (_step + 1) * N_CODEBOOKS) or _frame
    return true
end

-- Frames `[e, f)`, decoded as `voxtral_decode` decodes from frame e: chunks of `CODEC_CHUNK`, each
-- re-reading `CODEC_CONTEXT` frames before it.
local function voxtral_decode_from(st, e, f)
    local _from = math.max(0, e - CODEC_CONTEXT)
    local _codes = voxtral_slice(st.codes, (_from - st.code_base) * N_CODEBOOKS, (f - st.code_base) * N_CODEBOOKS)
    return voxtral_decode(_codes, f - _from, N_CODEBOOKS, loom.state('codec_chunk'), CODEC_CONTEXT,
                          SAMPLES_PER_FRAME, e - _from)
end

function stream_open(inputs)
    if inputs.seed then loom.seed_rng(inputs.seed) end
    -- `inputs.tokens` is the text's ids as `Tekkenizer.encode(text, bos=False, eos=False)` returns them.
    local _n_text = #inputs.tokens
    if _n_text == 0 then error('voxtral-tts: no text ids') end

    -- THE VOICE: `[n, DIM]` rows, the file's own (`voice.default`) unless a voice file sets `voice`.
    local _voice = inputs.voice or loom.get_weight('embed_prompt', 'voice.default')
    if #_voice == 0 or #_voice % DIM ~= 0 then
        error(string.format('voxtral-tts: a voice is rows of %d values; this one holds %d', DIM, #_voice))
    end
    local _n_voice = #_voice / DIM

    -- THE PROMPT: [BOS] [BEGIN_AUDIO] [AUDIO] x n_voice [NEXT_AUDIO_TEXT] text [REPEAT_AUDIO_TEXT]
    -- [BEGIN_AUDIO] -- `InstructTokenizerV7.encode_speech_request`, whose ids these are.
    local _ids = {BOS, BEGIN_AUDIO}
    for _i = 1, _n_voice do _ids[#_ids + 1] = AUDIO end
    _ids[#_ids + 1] = NEXT_AUDIO_TEXT
    for _i = 1, _n_text do _ids[#_ids + 1] = inputs.tokens[_i] end
    _ids[#_ids + 1] = REPEAT_AUDIO_TEXT
    _ids[#_ids + 1] = BEGIN_AUDIO
    local _n = #_ids
    if _n >= LM_MAX_POSITIONS then
        error(string.format('voxtral-tts: a %d-row prompt leaves no room in a %d-position cache', _n,
                            LM_MAX_POSITIONS))
    end
    -- The voice's rows over its slots, zeros (masked out) elsewhere.
    local _rows, _mask = {}, {}
    for _i = 1, _n * DIM do _rows[_i] = 0 end
    for _i = 1, _n_voice * DIM do _rows[2 * DIM + _i] = _voice[_i] end
    for _i = 1, _n do _mask[_i] = (_i > 2 and _i <= 2 + _n_voice) and 1 or 0 end

    local piece_frames = inputs.piece_frames or -1
    if piece_frames < -1 then
        error('voxtral-tts: piece_frames is a frame count, 0 for the whole turn, or -1 for the growing default')
    end
    local _teacher = inputs.teacher_codes
    local _budget = inputs.max_frames or MAX_FRAMES
    if _teacher ~= nil then _budget = #_teacher / N_CODEBOOKS end
    loom.set_state('prompt_ids', _ids)
    loom.set_state('prompt_rows', _rows)
    loom.set_state('prompt_mask', _mask)
    loom.set_state('budget', math.min(_budget, LM_MAX_POSITIONS - _n))
    loom.set_state('cfg', inputs.cfg or DEFAULT_CFG)
    loom.set_state('noise', inputs.noise)
    loom.set_state('teacher_codes', _teacher)
    loom.set_state('piece_frames', piece_frames)
    loom.set_state('return_codes', inputs.return_codes and 1 or 0)
    loom.set_state('codec_chunk', inputs.codec_chunk or CODEC_CHUNK)
    voxtral_save({started = 0, step = 0, n_past = 0, pending = {}, codes = {}, code_base = 0, n_frames = 0,
                  n_audio = -1, emitted = 0, ended = 0})
    loom.set_state('piece', 0)
    loom.set_state('n_pieces', piece_frames == 0 and 1 or 0)
    loom.set_state('done', 0)
    return {}
end

function stream_feed(inputs)
    if loom.state('done') ~= 0 then return {} end
    local st = voxtral_load()
    local piece_frames = loom.state('piece_frames')
    local wave = {}
    if piece_frames == 0 then
        while voxtral_step(st) do end
        -- `return_codes` hands back every frame computed, [END_AUDIO] frame included -- the loop's own
        -- state, which is what an oracle compares frame by frame. Otherwise the codec, on the frames before
        -- [END_AUDIO] (all of them when the budget ran out first).
        if loom.state('return_codes') == 1 then
            wave = st.codes
        else
            local _n = st.n_audio >= 0 and st.n_audio or st.n_frames
            wave = voxtral_decode(st.codes, _n, N_CODEBOOKS, loom.state('codec_chunk'), CODEC_CONTEXT,
                                  SAMPLES_PER_FRAME)
        end
    else
        local want = piece_frames
        if want < 0 then want = math.min(PIECE_FRAMES_GROWN, 2 ^ loom.state('piece')) end
        local _goal = st.emitted + want
        while st.n_audio < 0 and st.n_frames < _goal do
            if not voxtral_step(st) then break end
        end
        -- With a teacher the loop runs on past [END_AUDIO]; those frames are not audio.
        while st.ended == 0 and st.n_audio >= 0 do
            if not voxtral_step(st) then break end
        end
        local f = st.n_audio >= 0 and st.n_audio or st.n_frames
        if f > st.emitted then
            wave = voxtral_decode_from(st, st.emitted, f)
            st.emitted = f
        end
        -- Drop the frames no later chunk re-reads.
        local _keep = math.max(0, st.emitted - CODEC_CONTEXT)
        if _keep > st.code_base and st.n_audio < 0 then
            st.codes = voxtral_slice(st.codes, (_keep - st.code_base) * N_CODEBOOKS,
                                     (st.n_frames - st.code_base) * N_CODEBOOKS)
            st.code_base = _keep
        end
    end
    if st.ended == 1 then loom.set_state('done', 1) end
    voxtral_save(st)
    loom.set_state('piece', loom.state('piece') + 1)
    return wave
end

function stream_finish(inputs)
    return {}
end
