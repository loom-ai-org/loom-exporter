-- ===== LFM2.5-Audio's TTS, live (loom.cpp ADR-100, ADR-115): a few audio frames per `stream_feed`. =====
--
-- The LM draws one 8-code frame per step (80 ms of 24 kHz audio). The detokenizer that turns frames into
-- samples is CAUSAL up to its ISTFT: each frame's codes repeated x6 into rows, an LFM2 backbone of causal
-- convolutions (kernel 3) and attention over a causal window of 30 rows, then a 'same'-padded ISTFT
-- (n_fft 1280, hop 320) whose frames overlap 1.5 rows into the future. So the samples up to
-- `DETOK_HOLD_SAMPLES` before the last drawn row are final, and they are recomputed exactly from a window
-- that starts `DETOK_CONTEXT_ROWS` rows (the backbone's receptive field; both export constants, derived
-- from the detokenizer's config) before the first row they read. The pieces joined are the one-pass
-- waveform to within f32 rounding.
--
-- Two cuts, by `inputs.piece_frames`:
--
--   * frames: that many frames drawn per feed, then their final samples handed back. A stream's default
--     (absent, or -1) GROWS: 1 frame, then 2, 4 and 8 from then on. `>= 1` fixes the count. The end of the
--     turn flushes the samples held back.
--   * 0 (what `infer` asks for): the whole turn in one feed, every frame through one detokenizer call --
--     the one pass as it always was.
--
-- The LM loop is `generate_sequential`'s, cut per step: TEXT (greedy) until `<|audio_start|>`, one FRAME
-- per step until a frame opens with end-of-audio, then text again until `<|im_end|>`. Each step feeds the
-- previous draw back BEFORE it draws (the first step prefills the prompt instead), so a draw always reads
-- the decoder's row from a graph run in the same call. Between feeds everything is the stream's state:
-- the prompt, the knobs, the step and position, the draw not yet fed back (`pending_kind` 1 a text id,
-- 2 a frame), the frames' codes from the first one a later window still reads (`code_base` on), and how
-- many samples are out. The LM's history is the model's KV cache, leased or owned (ADR-099). `piece`
-- counts feeds, `n_pieces` is 1 when the feed is the whole turn (0 otherwise), `done` is set at the end.

local LFM25_LOOP = {'started', 'step', 'n_past', 'audio', 'ended', 'pending_kind', 'pending_id',
                    'pending_frame', 'codes', 'code_base', 'n_frames', 'emitted'}

local function lfm25_load()
    local st = {}
    for _, name in ipairs(LFM25_LOOP) do st[name] = loom.state(name) end
    return st
end

local function lfm25_save(st)
    for _, name in ipairs(LFM25_LOOP) do loom.set_state(name, st[name]) end
end

-- `n` rows of `module`'s output through the decoder at the stream's position; returns the row count, so
-- the draw that follows reads the last one.
local function lfm25_feed(st, module, n)
    loom.run_subgraph_and_retain('decoder', {n_tokens = n, n_past = st.n_past},
        {inputs_embeds = {from = module}, position_ids = loom.range(st.n_past, n),
         attention_mask = loom.causal_mask(n, st.n_past)})
    st.n_past = st.n_past + n
    return n
end

-- One step of the loop: the previous draw fed back (the prompt on the first step), then one draw. Returns
-- false once the turn has ended (`<|im_end|>`, or the step budget spent).
local function lfm25_step(st)
    if st.step >= loom.state('max_new_tokens') then
        st.ended = 1
        return false
    end
    local _last_n
    if st.started == 0 then
        local _prompt = loom.state('prompt')
        loom.run_subgraph_and_retain('embed', {n_tokens = #_prompt, n_past = 0}, {tokens = _prompt})
        _last_n = lfm25_feed(st, 'embed', #_prompt)
        st.started = 1
    elseif st.pending_kind == 1 then
        loom.run_subgraph_and_retain('embed', {n_tokens = 1, n_past = 0}, {tokens = {st.pending_id}})
        _last_n = lfm25_feed(st, 'embed', 1)
    else
        loom.run_subgraph_and_retain('audio_embed', {n_tokens = 1, n_past = 0}, {codes = st.pending_frame})
        _last_n = lfm25_feed(st, 'audio_embed', 1)
    end
    st.pending_kind = 0
    if st.n_past >= MAX_SEQ_LEN then error('lfm2.5-audio: the speech outgrew the KV cache') end
    st.step = st.step + 1

    local _hidden = {from = 'decoder', row = _last_n - 1, rows = 1}
    if st.audio == 0 then
        -- Text is greedy (`text_temperature=None`).
        loom.run_subgraph_and_retain('lm_head', {n_tokens = 1, n_past = 0}, {hidden = _hidden})
        local _id = loom.argmax_row('lm_head', 0)
        if _id == END_OF_TURN then
            st.ended = 1
            return false
        end
        if _id == AUDIO_START then st.audio = 1 end
        st.pending_kind, st.pending_id = 1, _id
        return true
    end
    -- Each frame's 8 codes by the depthformer, row j conditioned on the codes before it.
    local _prev, _frame = {}, {}
    for _j = 1, N_CODEBOOKS do _prev[_j] = 0 end
    local _temperature, _top_k = loom.state('temperature'), loom.state('top_k')
    for _j = 0, N_CODEBOOKS - 1 do
        loom.run_subgraph_and_retain('depth', {n_tokens = 1, n_past = 0}, {hidden = _hidden, prev = _prev})
        local _code = loom.sample_row('depth', _j, {temperature = _temperature, top_k = _top_k})
        _frame[_j + 1] = _code
        -- An end-of-audio first code makes the whole frame end-of-audio; its other codes would be
        -- overwritten, so they are not drawn.
        if _j == 0 and _code == END_OF_AUDIO then break end
        if _j + 1 < N_CODEBOOKS then _prev[_j + 2] = _code end
    end
    if _frame[1] == END_OF_AUDIO then
        for _j = 1, N_CODEBOOKS do _frame[_j] = END_OF_AUDIO end
        st.audio = 0
    else
        local _base = (st.n_frames - st.code_base) * N_CODEBOOKS
        for _j = 1, N_CODEBOOKS do st.codes[_base + _j] = _frame[_j] end
        st.n_frames = st.n_frames + 1
    end
    st.pending_kind, st.pending_frame = 2, _frame
    return true
end

-- Detokenizer rows `[a, b)` (row r is frame `r // 6`'s codes; positions are the rows' own), and the
-- samples `[s, e)` of the turn kept from it: the window's sample m is the turn's `a * hop + m`.
local function lfm25_detokenize(st, a, b, s, e)
    local _rows = {}
    for _r = a, b - 1 do
        local _base = (math.floor(_r / DETOK_UPSAMPLE) - st.code_base) * N_CODEBOOKS
        for _j = 1, N_CODEBOOKS do _rows[#_rows + 1] = st.codes[_base + _j] end
    end
    local wave = loom.run_subgraph('detokenizer', {n_codes = b - a, n_past = 0},
        {codes = _rows, positions = loom.range(a, b - a)})
    local _skip = s - a * ISTFT_HOP
    if _skip == 0 and e - s == #wave then return wave end
    local out = {}
    for _i = 1, e - s do out[_i] = wave[_skip + _i] end
    return out
end

-- The first row of a window whose samples from `s` on are the one-pass ones: sample n reads the ISTFT
-- frames (rows) from `ceil((n + trim - n_fft + 1) / hop)`, and each of those reads `DETOK_CONTEXT_ROWS`
-- rows before it.
local function lfm25_window_start(s)
    local _first = math.ceil((s + DETOK_TRIM - ISTFT_N_FFT + 1) / ISTFT_HOP)
    return math.max(0, _first - DETOK_CONTEXT_ROWS)
end

function stream_open(inputs)
    if inputs.seed then loom.seed_rng(inputs.seed) end
    -- The vocabulary's encode opens with `<|startoftext|>` (the tokenizer adds a BOS); `ChatState.add_text`
    -- does not, and the prompt already has one at its head -- so a leading BOS on the text is dropped.
    local _text = {}
    for _i = 1, #inputs.tokens do
        if not (_i == 1 and inputs.tokens[1] == BOS) then _text[#_text + 1] = inputs.tokens[_i] end
    end
    -- The prompt is `ChatState`'s, every piece tokenized on its own, all text: so one prefill.
    local _prompt = {}
    for _, part in ipairs({PROMPT_PRE, inputs.voice_prompt or DEFAULT_VOICE, PROMPT_MID, _text, PROMPT_TAIL}) do
        for _i = 1, #part do _prompt[#_prompt + 1] = part[_i] end
    end
    local piece_frames = inputs.piece_frames or -1
    if piece_frames < -1 then
        error('lfm2.5-audio: piece_frames is a frame count, 0 for the whole turn, or -1 for the growing default')
    end
    loom.set_state('prompt', _prompt)
    loom.set_state('piece_frames', piece_frames)
    loom.set_state('temperature', inputs.temperature or DEFAULT_TEMPERATURE)
    loom.set_state('top_k', inputs.top_k or DEFAULT_TOP_K)
    loom.set_state('max_new_tokens', inputs.max_new_tokens or MAX_NEW_TOKENS)
    loom.set_state('return_codes', inputs.return_codes and 1 or 0)
    lfm25_save({started = 0, step = 0, n_past = 0, audio = 0, ended = 0, pending_kind = 0, pending_id = 0,
                pending_frame = {}, codes = {}, code_base = 0, n_frames = 0, emitted = 0})
    loom.set_state('piece', 0)
    loom.set_state('n_pieces', piece_frames == 0 and 1 or 0)
    loom.set_state('done', 0)
    return {}
end

function stream_feed(inputs)
    if loom.state('done') ~= 0 then return {} end
    local st = lfm25_load()
    local piece_frames = loom.state('piece_frames')
    local wave = {}
    if piece_frames == 0 then
        while lfm25_step(st) do end
        if st.n_frames == 0 then error('lfm2.5-audio: the model produced no audio frames') end
        if loom.state('return_codes') == 1 then
            wave = st.codes
        else
            -- The frames, all at once: what liquid-audio's one call computes.
            local _n = DETOK_UPSAMPLE * st.n_frames
            wave = lfm25_detokenize(st, 0, _n, 0, _n * ISTFT_HOP)
        end
    else
        local want = piece_frames
        if want < 0 then want = math.min(PIECE_FRAMES_GROWN, 2 ^ loom.state('piece')) end
        local _goal = st.n_frames + want
        while st.n_frames < _goal do
            if not lfm25_step(st) then break end
        end
        if st.ended == 1 and st.n_frames == 0 then error('lfm2.5-audio: the model produced no audio frames') end
        -- Every row drawn so far; on the turn's end, every sample, else those no later row's ISTFT frame
        -- overlaps.
        local b = DETOK_UPSAMPLE * st.n_frames
        local e = b * ISTFT_HOP
        if st.ended == 0 then e = e - DETOK_TRIM end
        if e > st.emitted then
            local a = lfm25_window_start(st.emitted)
            wave = lfm25_detokenize(st, a, b, st.emitted, e)
            st.emitted = e
        end
        -- Drop the frames no later window reaches.
        local _keep = math.floor(lfm25_window_start(st.emitted) / DETOK_UPSAMPLE)
        if _keep > st.code_base then
            local _codes, _drop = {}, (_keep - st.code_base) * N_CODEBOOKS
            for _i = 1, (st.n_frames - _keep) * N_CODEBOOKS do _codes[_i] = st.codes[_drop + _i] end
            st.codes, st.code_base = _codes, _keep
        end
    end
    if st.ended == 1 then loom.set_state('done', 1) end
    lfm25_save(st)
    loom.set_state('piece', loom.state('piece') + 1)
    return wave
end

function stream_finish(inputs)
    return {}
end
