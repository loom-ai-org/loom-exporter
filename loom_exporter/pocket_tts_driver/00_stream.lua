-- ===== Pocket-TTS, live (loom.cpp ADR-100): one sentence chunk per `stream_feed`. =====
--
-- The reference generates and decodes chunk by chunk, each chunk from the voice afresh and Mimi from a
-- fresh state, so a chunk's audio does not depend on the chunks after it and the chunks joined are the
-- one-pass waveform: a stream hands each one back as soon as it is made. `infer` (01_flow_lm.lua) is this
-- stream opened, fed to its end and finished inside one call.
--
-- The functions hold nothing between calls. What a chunk needs past `stream_open` is the stream's state:
-- the chunks (`chunk_ids`, cut at `chunk_ends`, with `chunk_short`), the knobs, a caller's voice, pinned
-- draws and teacher latents, and the run-wide counters `draw` and `kept`. `piece` is the next chunk's
-- index, `n_pieces` how many there are and `done` set after the last, as the host's door reads them.

-- Kept only when given: a state value is numbers, and one never set reads back as nil.
local function pocket_keep(name, value)
    if value ~= nil then loom.set_state(name, value) end
end

-- One chunk: its text ids through the flow LM to latents, then Mimi to samples.
local function pocket_chunk(text, short)
    local _n_text = #text
    local _std, _threshold, _n_steps = loom.state('std'), loom.state('threshold'), loom.state('n_steps')
    local _noise, _forced = loom.state('noise'), loom.state('teacher_latents')
    -- Run-wide counters, so a caller's pinned draws (`noise`, one `[32]` row per step TAKEN, row-major)
    -- and teacher latents (one row per frame KEPT) run on across chunks in the order the reference
    -- consumes them.
    local _draw, _kept = loom.state('draw'), loom.state('kept')

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
    local _n_past = _n_voice + _n_text

    -- `TTSModel._estimate_max_gen_len`, and the tail the reference keeps after the EOS head fires:
    -- `prepare_text_prompt`'s guess for the chunk, plus the 2 `generate_audio_stream` adds.
    local _budget = loom.state('max_frames') or
        math.ceil((_n_text / TOKENS_PER_SECOND_ESTIMATE + GEN_SECONDS_PADDING) * FRAME_RATE)
    _budget = math.min(_budget, LM_MAX_POSITIONS - _n_past)
    local _after = loom.state('frames_after_eos') or
        ((short and SHORT_CHUNK_FRAMES_AFTER_EOS or LONG_CHUNK_FRAMES_AFTER_EOS) + FRAMES_AFTER_EOS_PADDING)

    -- Step 0's input is the checkpoint's `bos_emb` (the reference feeds NaN and swaps it in).
    local _latent = loom.get_weight('step_embed', 'bos_emb')
    local _latents = {}
    local _n_frames = 0
    local _eos_step = nil
    for _step = 0, _budget - 1 do
        loom.run_subgraph_and_retain('step_embed', {n_tokens = 1, n_past = 0}, {latent = _latent})
        loom.run_subgraph_and_retain('lm', {n_tokens = 1, n_past = _n_past},
            {inputs_embeds = {from = 'step_embed'}, position_ids = {_n_past},
             attention_mask = loom.causal_mask(1, _n_past)})
        _n_past = _n_past + 1

        -- One unit draw per step, EVERY step including the one EOS ends on, as the reference draws.
        local _x = {}
        if _noise ~= nil then
            for _i = 1, LATENT_DIM do _x[_i] = _noise[_draw * LATENT_DIM + _i] end
        else
            _x = loom.gaussian_array(LATENT_DIM)
        end
        _draw = _draw + 1
        -- One sample of the flow head: LSD integrated over `n_steps`, or a one-step head
        -- (`pocket_flow_head`, from whichever of the two head fragments this export shipped).
        _x = pocket_flow_head(_x, _std, _n_steps)

        -- `_autoregressive_generation`: the EOS head is ignored for the first frames, then the loop runs
        -- `_after` more steps and ends WITHOUT keeping the latent of the step it ends on.
        local _eos = loom.get_output('lm', 2)[1]
        if _eos_step == nil and _eos > _threshold and _step >= MIN_FRAMES_BEFORE_EOS then
            _eos_step = _step
        end
        if _eos_step ~= nil and _step >= _eos_step + _after then break end
        for _i = 1, LATENT_DIM do _latents[_n_frames * LATENT_DIM + _i] = _x[_i] end
        _n_frames = _n_frames + 1
        _kept = _kept + 1
        _latent = _x
        -- TEACHER FORCING: a caller's latents are fed back in place of the ones this loop draws, while
        -- every step still computes, emits and EOS-tests its own. The loop is a feedback system, so f32
        -- rounding compounds along it and two correct implementations drift apart (3.8e-03 on the
        -- waveform by the end of a 6 s clip); fed the same history, each step is comparable on its own.
        -- It is also how a continuation would be prompted.
        if _forced ~= nil and _forced[_kept * LATENT_DIM] ~= nil then
            _latent = {}
            for _i = 1, LATENT_DIM do _latent[_i] = _forced[(_kept - 1) * LATENT_DIM + _i] end
        end
    end
    if _n_frames == 0 then error('pocket-tts: the flow LM produced no frames') end
    loom.set_state('draw', _draw)
    loom.set_state('kept', _kept)

    -- Mimi, every frame of the chunk in one call from a fresh state: the reference's decoder thread
    -- starts afresh per chunk too, and its streamed calls compute what one call does.
    local _wave = loom.run_subgraph('mimi_decoder', {n_codes = _n_frames, n_past = 0},
        {latents = _latents, positions = loom.range(0, _n_frames * MIMI_UPSAMPLE)})
    -- The reference's fade-in of each chunk's first samples (a fresh decoder state clicks). Exact: an f32
    -- times an f32 is exact in a double, and the one rounding back to f32 is torch's `*=`.
    if FADE_IN_SAMPLES > 0 then
        local _fade = loom.get_weight('step_embed', 'fade_in')
        for _i = 1, math.min(FADE_IN_SAMPLES, #_wave) do _wave[_i] = _wave[_i] * _fade[_i] end
    end
    return _wave
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
    loom.set_state('chunk_ids', ids)
    loom.set_state('chunk_ends', ends)
    loom.set_state('chunk_short', short)
    loom.set_state('std', math.sqrt(inputs.temperature or DEFAULT_TEMPERATURE))
    loom.set_state('threshold', inputs.eos_threshold or DEFAULT_EOS_THRESHOLD)
    loom.set_state('n_steps', inputs.n_steps or DEFAULT_DECODE_STEPS)
    pocket_keep('max_frames', inputs.max_frames)
    pocket_keep('frames_after_eos', inputs.frames_after_eos)
    pocket_keep('voice_kv', inputs.voice_kv)
    pocket_keep('noise', inputs.noise)
    pocket_keep('teacher_latents', inputs.teacher_latents)
    loom.set_state('draw', 0)
    loom.set_state('kept', 0)
    loom.set_state('piece', 0)
    loom.set_state('n_pieces', #chunks)
    loom.set_state('done', 0)
    return {}
end

function stream_feed(inputs)
    local ends = loom.state('chunk_ends')
    local c = loom.state('piece') + 1
    if c > #ends then
        loom.set_state('done', 1)
        return {}
    end
    local all, text = loom.state('chunk_ids'), {}
    for i = (c == 1 and 0 or ends[c - 1]) + 1, ends[c] do text[#text + 1] = all[i] end
    local wave = pocket_chunk(text, loom.state('chunk_short')[c] == 1)
    loom.set_state('piece', c)
    if c == #ends then loom.set_state('done', 1) end
    return wave
end

function stream_finish(inputs)
    return {}
end
