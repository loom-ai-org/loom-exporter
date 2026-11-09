-- ===== Pocket-TTS, live (loom.cpp ADR-100, ADR-101): a frame, or a sentence chunk, per `stream_feed`. =====
--
-- Two cuts, by `inputs.piece_frames`:
--
--   * frames: that many latent frames per feed, decoded together by `mimi_step` with the decoder's state
--     carried between calls (upstream's `generate_audio_stream`, which decodes whatever latents are
--     queued). The pieces joined are the one-pass waveform to within f32 rounding: the step and the
--     one-call decoder sum in a different order. A stream's default (absent, or -1) GROWS: 1 frame, then
--     2, 4 and 8 from then on, so the first audio is ready after one frame and the rest pays little for
--     the cut -- one frame per call costs about 20% over the one-pass decode on the dev box, eight about 5%
--     (ADR-101). `>= 1` fixes the count.
--   * 0 (what `infer` asks for): one sentence chunk per feed, every frame of it through `mimi_decoder` in
--     one call. The reference decodes each chunk from a fresh state, so the chunks joined are the one-pass
--     waveform exactly, and this is the one-pass path bit for bit.
--
-- The functions hold nothing between calls. Everything is the stream's state: the chunks (`chunk_ids`,
-- cut at `chunk_ends`, with `chunk_short`), the knobs, a caller's voice, pinned draws and teacher latents,
-- the run-wide counters `draw` and `kept`, the open chunk's loop variables, and Mimi's state tensors
-- (`mimi`, on the device). The flow LM's history is the model's KV cache, which a stream leases or owns
-- (ADR-099). `piece` counts feeds, `n_pieces` is the chunk count when a feed is a chunk (0 when it is a
-- frame) and `done` is set after the last, as the host's door reads them.

-- Kept only when given: a state value is numbers, and one never set reads back as nil.
local function pocket_keep(name, value)
    if value ~= nil then loom.set_state(name, value) end
end

-- The open chunk's loop variables, read at the start of a feed and written at its end.
local POCKET_LOOP = {'chunk', 'chunks_done', 'n_past', 'step', 'budget', 'after', 'eos_step', 'n_frames',
                     'draw', 'kept', 'mimi_pos', 'latent'}

local function pocket_load()
    local st = {}
    for _, name in ipairs(POCKET_LOOP) do st[name] = loom.state(name) end
    return st
end

local function pocket_save(st)
    for _, name in ipairs(POCKET_LOOP) do pocket_keep(name, st[name]) end
end

-- Opens chunk `c`: the voice written into the flow LM's KV cache, the text prefilled after it.
local function pocket_open_chunk(st, c)
    local ends, all = loom.state('chunk_ends'), loom.state('chunk_ids')
    local text = {}
    for i = (c == 1 and 0 or ends[c - 1]) + 1, ends[c] do text[#text + 1] = all[i] end
    local _n_text = #text
    -- The voice is the flow LM's KV cache after its own prefill, so it is WRITTEN into the cache rather
    -- than run: rows [0, n_voice) are the voice, and the text continues at position n_voice. Every chunk
    -- starts from it afresh (`copy_state=True`). A caller's own saved state (`voice_kv`, per layer K then
    -- V, `[n, 1024]` each) replaces the built-in one.
    local _n_voice = loom.seed_kv('lm', loom.state('voice_kv') or 'voice.kv')
    if _n_voice + _n_text >= LM_MAX_POSITIONS then
        error(string.format('pocket-tts: a %d-row voice and %d text tokens leave no room in a '
                            .. '%d-position cache', _n_voice, _n_text, LM_MAX_POSITIONS))
    end
    loom.run_subgraph_and_retain('text_embed', {n_tokens = _n_text, n_past = 0}, {text_ids = text})
    loom.run_subgraph_and_retain('lm', {n_tokens = _n_text, n_past = _n_voice},
        {inputs_embeds = {from = 'text_embed'}, position_ids = loom.range(_n_voice, _n_text),
         attention_mask = loom.causal_mask(_n_text, _n_voice)})
    st.chunk = c
    st.n_past = _n_voice + _n_text
    -- `TTSModel._estimate_max_gen_len`, and the tail the reference keeps after the EOS head fires:
    -- `prepare_text_prompt`'s guess for the chunk, plus the 2 `generate_audio_stream` adds.
    local _budget = loom.state('max_frames') or
        math.ceil((_n_text / TOKENS_PER_SECOND_ESTIMATE + GEN_SECONDS_PADDING) * FRAME_RATE)
    st.budget = math.min(_budget, LM_MAX_POSITIONS - st.n_past)
    st.after = loom.state('frames_after_eos') or
        ((loom.state('chunk_short')[c] == 1 and SHORT_CHUNK_FRAMES_AFTER_EOS or LONG_CHUNK_FRAMES_AFTER_EOS)
         + FRAMES_AFTER_EOS_PADDING)
    st.step, st.eos_step, st.n_frames, st.mimi_pos = 0, -1, 0, 0
    -- Step 0's input is the checkpoint's `bos_emb` (the reference feeds NaN and swaps it in).
    st.latent = loom.get_weight('step_embed', 'bos_emb')
end

-- One step of the open chunk's loop. Returns the latent it keeps, or nil once the loop has ended (the
-- budget spent, or the frames after EOS run out -- the step that ends it keeps nothing).
local function pocket_step(st, noise, forced)
    if st.step >= st.budget then return nil end
    loom.run_subgraph_and_retain('step_embed', {n_tokens = 1, n_past = 0}, {latent = st.latent})
    loom.run_subgraph_and_retain('lm', {n_tokens = 1, n_past = st.n_past},
        {inputs_embeds = {from = 'step_embed'}, position_ids = {st.n_past},
         attention_mask = loom.causal_mask(1, st.n_past)})
    st.n_past = st.n_past + 1

    -- One unit draw per step, EVERY step including the one EOS ends on, as the reference draws.
    local _x = {}
    if noise ~= nil then
        for _i = 1, LATENT_DIM do _x[_i] = noise[st.draw * LATENT_DIM + _i] end
    else
        _x = loom.gaussian_array(LATENT_DIM)
    end
    st.draw = st.draw + 1
    -- One sample of the flow head: LSD integrated over `n_steps`, or a one-step head
    -- (`pocket_flow_head`, from whichever of the two head fragments this export shipped).
    _x = pocket_flow_head(_x, loom.state('std'), loom.state('n_steps'))

    -- `_autoregressive_generation`: the EOS head is ignored for the first frames, then the loop runs
    -- `after` more steps and ends WITHOUT keeping the latent of the step it ends on.
    local _step = st.step
    st.step = _step + 1
    local _eos = loom.get_output('lm', 2)[1]
    if st.eos_step < 0 and _eos > loom.state('threshold') and _step >= MIN_FRAMES_BEFORE_EOS then
        st.eos_step = _step
    end
    if st.eos_step >= 0 and _step >= st.eos_step + st.after then
        st.step = st.budget
        return nil
    end
    st.n_frames = st.n_frames + 1
    st.kept = st.kept + 1
    st.latent = _x
    -- TEACHER FORCING: a caller's latents are fed back in place of the ones this loop draws, while every
    -- step still computes, emits and EOS-tests its own. The loop is a feedback system, so f32 rounding
    -- compounds along it and two correct implementations drift apart (3.8e-03 on the waveform by the end
    -- of a 6 s clip); fed the same history, each step is comparable on its own. It is also how a
    -- continuation would be prompted.
    if forced ~= nil and forced[st.kept * LATENT_DIM] ~= nil then
        st.latent = {}
        for _i = 1, LATENT_DIM do st.latent[_i] = forced[(st.kept - 1) * LATENT_DIM + _i] end
    end
    return _x
end

-- The reference's fade-in of each chunk's first samples (a fresh decoder state clicks). Exact: an f32
-- times an f32 is exact in a double, and the one rounding back to f32 is torch's `*=`.
local function pocket_fade_in(wave)
    if FADE_IN_SAMPLES > 0 then
        local _fade = loom.get_weight('step_embed', 'fade_in')
        for _i = 1, math.min(FADE_IN_SAMPLES, #wave) do wave[_i] = wave[_i] * _fade[_i] end
    end
end

-- `n` latents through `mimi_step`, the decoder's state read from and written back to the stream's `mimi`
-- tensors. A chunk's first call starts from `mimi_init`'s fresh state, whose outputs are the states
-- alone; `mimi_step`'s are the samples, then the states.
local function pocket_decode(st, latents, n)
    local first = st.mimi_pos == 0
    local inputs = {latents = latents, positions = loom.range(st.mimi_pos, n * MIMI_UPSAMPLE),
                    key_positions = loom.range(st.mimi_pos - MIMI_CONTEXT, MIMI_CONTEXT + n * MIMI_UPSAMPLE)}
    local shift = 1
    if first then
        loom.run_subgraph_and_retain('mimi_init', {n_tokens = 0, n_past = 0}, {zero = {0}})
        loom.keep_state('mimi', 'mimi_init')
        shift = 0
    end
    for k = 1, MIMI_STATES do inputs['state_' .. k] = {state = 'mimi', index = k + shift} end
    loom.run_subgraph_and_retain('mimi_step', {n_codes = n, n_tokens = 0, n_past = 0}, inputs)
    loom.keep_state('mimi', 'mimi_step')
    st.mimi_pos = st.mimi_pos + n * MIMI_UPSAMPLE
    local wave = loom.get_output('mimi_step', 1)
    if first then pocket_fade_in(wave) end
    return wave
end

local function pocket_close_chunk(st)
    if st.n_frames == 0 then error('pocket-tts: the flow LM produced no frames') end
    st.chunk = 0
    st.chunks_done = st.chunks_done + 1
    if st.chunks_done == #loom.state('chunk_ends') then loom.set_state('done', 1) end
end

function stream_open(inputs)
    -- `inputs.tokens` is what `loom::PocketTtsVocab::encode` returns: each sentence chunk's prepared ids,
    -- opened by a header saying which tail the reference guessed for it. The header is not fed to the
    -- model: the reference feeds the tokenizer's output straight to the conditioner.
    if inputs.seed then loom.seed_rng(inputs.seed) end
    local chunks = pocket_split_chunks(inputs.tokens, CHUNK_HEADER_SHORT, CHUNK_HEADER_LONG,
                                       loom.get_weight('step_embed', 'word_start'), SHORT_CHUNK_MAX_WORDS)
    if #chunks == 0 then error('pocket-tts: no text ids') end
    local ids, ends, short = {}, {}, {}
    for c, chunk in ipairs(chunks) do
        for _, id in ipairs(chunk.ids) do ids[#ids + 1] = id end
        ends[c] = #ids
        short[c] = chunk.short and 1 or 0
    end
    local piece_frames = inputs.piece_frames or -1
    if piece_frames < -1 then
        error('pocket-tts: piece_frames is a frame count, 0 for a chunk, or -1 for the growing default')
    end
    loom.set_state('chunk_ids', ids)
    loom.set_state('chunk_ends', ends)
    loom.set_state('chunk_short', short)
    loom.set_state('piece_frames', piece_frames)
    loom.set_state('std', math.sqrt(inputs.temperature or DEFAULT_TEMPERATURE))
    loom.set_state('threshold', inputs.eos_threshold or DEFAULT_EOS_THRESHOLD)
    loom.set_state('n_steps', inputs.n_steps or DEFAULT_DECODE_STEPS)
    pocket_keep('max_frames', inputs.max_frames)
    pocket_keep('frames_after_eos', inputs.frames_after_eos)
    pocket_keep('voice_kv', inputs.voice_kv)
    pocket_keep('noise', inputs.noise)
    pocket_keep('teacher_latents', inputs.teacher_latents)
    pocket_save({chunk = 0, chunks_done = 0, draw = 0, kept = 0})
    loom.set_state('piece', 0)
    loom.set_state('n_pieces', piece_frames == 0 and #chunks or 0)
    loom.set_state('done', 0)
    return {}
end

function stream_feed(inputs)
    if loom.state('done') ~= 0 then return {} end
    local st = pocket_load()
    local noise, forced = loom.state('noise'), loom.state('teacher_latents')
    local piece_frames = loom.state('piece_frames')
    local wave = {}
    if st.chunk == 0 then pocket_open_chunk(st, st.chunks_done + 1) end
    if piece_frames == 0 then
        -- A whole chunk: every latent, then Mimi over all of them in one call from a fresh state. The
        -- reference's decoder thread starts afresh per chunk too, and its streamed calls compute what one
        -- call does.
        local latents = {}
        while true do
            local x = pocket_step(st, noise, forced)
            if x == nil then break end
            local base = (st.n_frames - 1) * LATENT_DIM
            for i = 1, LATENT_DIM do latents[base + i] = x[i] end
        end
        if st.n_frames > 0 then
            wave = loom.run_subgraph('mimi_decoder', {n_codes = st.n_frames, n_past = 0},
                {latents = latents, positions = loom.range(0, st.n_frames * MIMI_UPSAMPLE)})
            pocket_fade_in(wave)
        end
        pocket_close_chunk(st)
    else
        -- Up to `want` latents, decoded in one call: the chunk's end closes the piece early.
        local want = piece_frames
        if want < 0 then want = math.min(PIECE_FRAMES_GROWN, 2 ^ loom.state('piece')) end
        local latents, n, ended = {}, 0, false
        while n < want do
            local x = pocket_step(st, noise, forced)
            if x == nil then
                ended = true
                break
            end
            for i = 1, LATENT_DIM do latents[n * LATENT_DIM + i] = x[i] end
            n = n + 1
        end
        if n > 0 then wave = pocket_decode(st, latents, n) end
        if ended then pocket_close_chunk(st) end
    end
    pocket_save(st)
    loom.set_state('piece', loom.state('piece') + 1)
    return wave
end

function stream_finish(inputs)
    return {}
end
