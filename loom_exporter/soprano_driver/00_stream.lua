-- ===== Soprano, live (loom.cpp ADR-100, ADR-114): a few hidden rows, or a sentence, per `stream_feed`. =====
--
-- Each sentence is its own generation from position 0 and its own decoder call, so its audio does not
-- depend on the sentences after it. Within one, the decoder (a x4 linear upsample, 8 ConvNeXt blocks of
-- kernel 3, an ISTFT of n_fft 2048 at hop 512) is LOCAL: a sample depends on the hidden rows a few either
-- side of it and on nothing further. So a piece of rows `[e, f)` is decoded from rows
-- `[e - DECODER_LEFT_ROWS, f + DECODER_RIGHT_ROWS)` -- the receptive field either side, which the export
-- derives from the decoder's own config -- and its samples are kept, as upstream's `infer_stream` decodes
-- a sliding window of hidden states with its `RECEPTIVE_FIELD` of 4 rows. The pieces joined are the
-- one-pass waveform to within f32 rounding.
--
-- Two cuts, by `inputs.piece_tokens`:
--
--   * rows: that many hidden rows' worth of audio (2048 samples, 64 ms each) per feed. A stream's default
--     (absent, or -1) GROWS: 1 row, then 2, 4, 8 and 16 (`PIECE_TOKENS_GROWN`) from then on, so the first
--     audio is ready after `1 + DECODER_RIGHT_ROWS` draws. `>= 1` fixes the count. A sentence's end closes
--     the piece early and flushes the rows the margin held back.
--   * 0 (what `infer` asks for): one sentence per feed, every row of it through `decoder` in one call --
--     the reference's non-streaming `infer`, bit for bit.
--
-- The functions hold nothing between calls. Everything is the stream's state: the prompts (`prompt_ids`,
-- cut at `prompt_ends`), the sampler's knobs, and the open sentence's loop variables: its index, the LM's
-- position and step, the repetition penalty's history (prompt and every id drawn), the id drawn last and
-- not yet fed back, the hidden rows still inside some later piece's window (from `row_base`), how many
-- rows there are and how many have been handed out as audio. The LM's history is the model's KV cache,
-- which a stream leases or owns (ADR-099). Each step feeds the previous draw back BEFORE it draws, so a
-- draw always reads the logits and hidden row of a graph run in the same call. `piece` counts feeds,
-- `n_pieces` is the sentence count when a feed is a sentence (0 when it is a few rows) and `done` is set
-- after the last, as the host's door reads them.

local SOPRANO_LOOP = {'sentence', 'sentences_done', 'n_past', 'step', 'limit', 'history', 'pending',
                      'rows', 'row_base', 'n_rows', 'emitted', 'ended'}

local function soprano_load()
    local st = {}
    for _, name in ipairs(SOPRANO_LOOP) do st[name] = loom.state(name) end
    return st
end

local function soprano_save(st)
    -- A state value is numbers: a table of them, or one. `nil` is never written (it would read back as
    -- whatever was there), so every field is given a value before a save.
    for _, name in ipairs(SOPRANO_LOOP) do loom.set_state(name, st[name]) end
end

-- Opens sentence `c`: its prompt through the LM, whose last row the first draw reads.
local function soprano_open_sentence(st, c)
    local ends, all = loom.state('prompt_ends'), loom.state('prompt_ids')
    local prompt = {}
    for i = (c == 1 and 0 or ends[c - 1]) + 1, ends[c] do prompt[#prompt + 1] = all[i] end
    -- `tokenizer(..., truncation=True, max_length=512)`: the reference keeps the first 512.
    if #prompt > MAX_PROMPT_TOKENS then
        local _cut = {}
        for _i = 1, MAX_PROMPT_TOKENS do _cut[_i] = prompt[_i] end
        prompt = _cut
    end
    local _n_prompt = #prompt
    loom.run_subgraph_and_retain('lm', {n_tokens = _n_prompt, n_past = 0},
        {input_ids = prompt, position_ids = loom.range(0, _n_prompt),
         attention_mask = loom.causal_mask(_n_prompt, 0)})
    st.sentence = c
    st.n_past = _n_prompt
    st.step = 0
    st.limit = math.min(loom.state('max_new_tokens'), LM_MAX_POSITIONS - _n_prompt)
    -- `RepetitionPenaltyLogitsProcessor` reads every id of `input_ids`, which in `generate` is the prompt
    -- AND what has been drawn so far -- so the history starts as the prompt.
    st.history = prompt
    st.pending = -1
    st.rows, st.row_base, st.n_rows, st.emitted, st.ended = {}, 0, 0, 0, 0
end

-- One step of the open sentence's loop: the previous draw fed back (none after the prefill, which ran in
-- this same call), then one draw. Keeps the hidden row that produced a non-EOS id; returns false once the
-- sentence has ended -- at `[STOP]`, or the step budget spent.
local function soprano_step(st)
    if st.step >= st.limit then
        st.ended = 1
        return false
    end
    if st.pending >= 0 then
        loom.run_subgraph_and_retain('lm', {n_tokens = 1, n_past = st.n_past},
            {input_ids = {st.pending}, position_ids = {st.n_past},
             attention_mask = loom.causal_mask(1, st.n_past)})
        st.n_past = st.n_past + 1
        st.pending = -1
    end
    local _id = loom.sample_row('lm', 0, {
        temperature = loom.state('temperature'), top_k = loom.state('top_k'), top_p = loom.state('top_p'),
        repetition_penalty = loom.state('repetition_penalty'), penalized = st.history})
    st.step = st.step + 1
    if _id == EOS_ID then
        st.ended = 1
        return false
    end
    -- The row that PRODUCED this id is the one the decoder reads; the row that produces `[STOP]` is
    -- never kept.
    local _h = loom.get_output('lm', 2)
    local _base = (st.n_rows - st.row_base) * HIDDEN_SIZE
    for _i = 1, HIDDEN_SIZE do st.rows[_base + _i] = _h[_i] end
    st.n_rows = st.n_rows + 1
    st.history[#st.history + 1] = _id
    st.pending = _id
    return true
end

-- Rows `[a, b)` through the decoder (`2048 * (b - a - 1)` samples, the first at row a's position), and
-- the samples of rows `[e, f)` kept: those are the ones whose receptive field the window holds whole.
local function soprano_decode(st, a, b, e, f)
    local hidden = {}
    local _from = (a - st.row_base) * HIDDEN_SIZE
    for _i = 1, (b - a) * HIDDEN_SIZE do hidden[_i] = st.rows[_from + _i] end
    local wave = loom.run_subgraph('decoder', {n_codes = b - a, n_past = 0}, {hidden = hidden})
    if e == a and f == b - 1 then return wave end
    local out = {}
    local _skip = (e - a) * SAMPLES_PER_ROW
    for _i = 1, (f - e) * SAMPLES_PER_ROW do out[_i] = wave[_skip + _i] end
    return out
end

-- Drops the rows no later window reaches: the next piece starts at `emitted`, and its window
-- `DECODER_LEFT_ROWS` before that.
local function soprano_trim(st)
    local _keep = math.max(0, st.emitted - DECODER_LEFT_ROWS)
    if _keep <= st.row_base then return end
    local _drop = (_keep - st.row_base) * HIDDEN_SIZE
    local _rows = {}
    for _i = 1, (st.n_rows - _keep) * HIDDEN_SIZE do _rows[_i] = st.rows[_drop + _i] end
    st.rows, st.row_base = _rows, _keep
end

local function soprano_close_sentence(st)
    st.sentence = 0
    st.sentences_done = st.sentences_done + 1
    -- Nothing of the closed sentence is read again.
    st.history, st.rows, st.row_base, st.n_rows, st.emitted = {}, {}, 0, 0, 0
    if st.sentences_done == #loom.state('prompt_ends') then loom.set_state('done', 1) end
end

function stream_open(inputs)
    -- `inputs.tokens` is what the vocabulary returns: every sentence already in the reference's prompt
    -- form, `[STOP][TEXT]{sentence}[START]`, one after another. `[STOP]` is the model's EOS and the
    -- prompt's first id, so it is also where one sentence's ids end and the next begin -- the cleaned
    -- text has no brackets left to spell it with. Ids that do not open with it are a caller's own bare
    -- sentence and are wrapped here, as one.
    if inputs.seed then loom.seed_rng(inputs.seed) end
    local ids, ends = {}, {}
    if inputs.tokens[1] ~= EOS_ID then
        ids[1], ids[2] = EOS_ID, TEXT_ID
        for _i = 1, #inputs.tokens do ids[#ids + 1] = inputs.tokens[_i] end
        ids[#ids + 1] = START_ID
        ends[1] = #ids
    else
        for _i = 1, #inputs.tokens do
            local _id = inputs.tokens[_i]
            if _id == EOS_ID and _i > 1 then ends[#ends + 1] = #ids end
            ids[#ids + 1] = _id
        end
        ends[#ends + 1] = #ids
    end
    if #ends == 0 then error('soprano: no text ids') end
    local piece_tokens = inputs.piece_tokens or -1
    if piece_tokens < -1 then
        error('soprano: piece_tokens is a row count, 0 for a sentence, or -1 for the growing default')
    end
    loom.set_state('prompt_ids', ids)
    loom.set_state('prompt_ends', ends)
    loom.set_state('piece_tokens', piece_tokens)
    loom.set_state('temperature', inputs.temperature or DEFAULT_TEMPERATURE)
    loom.set_state('top_k', inputs.top_k or DEFAULT_TOP_K)
    loom.set_state('top_p', inputs.top_p or DEFAULT_TOP_P)
    loom.set_state('repetition_penalty', inputs.repetition_penalty or DEFAULT_REPETITION_PENALTY)
    loom.set_state('max_new_tokens', inputs.max_new_tokens or MAX_NEW_TOKENS)
    soprano_save({sentence = 0, sentences_done = 0, n_past = 0, step = 0, limit = 0, history = {},
                  pending = -1, rows = {}, row_base = 0, n_rows = 0, emitted = 0, ended = 0})
    loom.set_state('piece', 0)
    loom.set_state('n_pieces', piece_tokens == 0 and #ends or 0)
    loom.set_state('done', 0)
    return {}
end

function stream_feed(inputs)
    if loom.state('done') ~= 0 then return {} end
    local st = soprano_load()
    local piece_tokens = loom.state('piece_tokens')
    local wave = {}
    if st.sentence == 0 then soprano_open_sentence(st, st.sentences_done + 1) end
    if piece_tokens == 0 then
        -- A whole sentence: every row, then the decoder over all of them in one call, as the reference's
        -- non-streaming `infer` decodes. L rows decode to `2048 * (L - 1)` samples, so a sentence needs
        -- two rows to make a sound.
        while soprano_step(st) do end
        if st.n_rows >= 2 then wave = soprano_decode(st, 0, st.n_rows, 0, st.n_rows - 1) end
        soprano_close_sentence(st)
    else
        -- Rows `[e, e + want)` once `DECODER_RIGHT_ROWS` more are drawn past them; the sentence's end
        -- closes the piece early, with every row left.
        local want = piece_tokens
        if want < 0 then want = math.min(PIECE_TOKENS_GROWN, 2 ^ loom.state('piece')) end
        local e = st.emitted
        while st.n_rows < e + want + DECODER_RIGHT_ROWS do
            if not soprano_step(st) then break end
        end
        local a = math.max(0, e - DECODER_LEFT_ROWS)
        if st.ended == 1 then
            -- The last rows: everything up to the sentence's final sample, `2048 * (L - 1)`.
            if st.n_rows >= 2 and st.n_rows - 1 > e then
                wave = soprano_decode(st, a, st.n_rows, e, st.n_rows - 1)
            end
            soprano_close_sentence(st)
        else
            local f = e + want
            wave = soprano_decode(st, a, f + DECODER_RIGHT_ROWS, e, f)
            st.emitted = f
            soprano_trim(st)
        end
    end
    soprano_save(st)
    loom.set_state('piece', loom.state('piece') + 1)
    return wave
end

function stream_finish(inputs)
    return {}
end
