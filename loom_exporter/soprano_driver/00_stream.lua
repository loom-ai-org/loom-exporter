-- ===== Soprano, live (loom.cpp ADR-100): one sentence per `stream_feed`. =====
--
-- Each sentence is its own generation from position 0 and its own decoder call, so its audio does not
-- depend on the sentences after it and the sentences joined are the one-pass waveform: a stream hands
-- each one back as soon as it is made. `infer` (01_generate.lua) is this stream opened, fed to its end
-- and finished inside one call.
--
-- The functions hold nothing between calls. What a sentence needs past `stream_open` is the stream's
-- state: the prompts (`prompt_ids`, cut at `prompt_ends`) and the sampler's knobs. `piece` is the next
-- sentence's index, `n_pieces` how many there are and `done` set after the last, as the host's door reads
-- them.

-- One sentence: its prompt through the LM, one hidden row per id drawn, the rows decoded.
local function soprano_sentence(prompt)
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

    local _temperature, _top_k, _top_p = loom.state('temperature'), loom.state('top_k'), loom.state('top_p')
    local _penalty = loom.state('repetition_penalty')
    -- `RepetitionPenaltyLogitsProcessor` reads every id of `input_ids`, which in `generate` is the prompt
    -- AND what has been drawn so far -- so the history starts as the prompt.
    local _history = {}
    for _i = 1, _n_prompt do _history[_i] = prompt[_i] end
    local _rows, _n_rows = {}, 0
    local _n_past = _n_prompt
    for _step = 1, math.min(loom.state('max_new_tokens'), LM_MAX_POSITIONS - _n_prompt) do
        local _id = loom.sample_row('lm', 0, {
            temperature = _temperature, top_k = _top_k, top_p = _top_p,
            repetition_penalty = _penalty, penalized = _history})
        if _id == EOS_ID then break end
        -- The row that PRODUCED this id is the one the decoder reads, so it is kept before the id is fed
        -- back; the row that produces `[STOP]` is never kept.
        local _h = loom.get_output('lm', 2)
        local _base = _n_rows * HIDDEN_SIZE
        for _i = 1, HIDDEN_SIZE do _rows[_base + _i] = _h[_i] end
        _n_rows = _n_rows + 1
        _history[#_history + 1] = _id
        loom.run_subgraph_and_retain('lm', {n_tokens = 1, n_past = _n_past},
            {input_ids = {_id}, position_ids = {_n_past},
             attention_mask = loom.causal_mask(1, _n_past)})
        _n_past = _n_past + 1
    end

    -- L rows decode to `2048 * (L - 1)` samples, so a sentence needs two rows to make a sound.
    if _n_rows < 2 then return {} end
    return loom.run_subgraph('decoder', {n_codes = _n_rows, n_past = 0}, {hidden = _rows})
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
    loom.set_state('prompt_ids', ids)
    loom.set_state('prompt_ends', ends)
    loom.set_state('temperature', inputs.temperature or DEFAULT_TEMPERATURE)
    loom.set_state('top_k', inputs.top_k or DEFAULT_TOP_K)
    loom.set_state('top_p', inputs.top_p or DEFAULT_TOP_P)
    loom.set_state('repetition_penalty', inputs.repetition_penalty or DEFAULT_REPETITION_PENALTY)
    loom.set_state('max_new_tokens', inputs.max_new_tokens or MAX_NEW_TOKENS)
    loom.set_state('piece', 0)
    loom.set_state('n_pieces', #ends)
    loom.set_state('done', 0)
    return {}
end

function stream_feed(inputs)
    local ends = loom.state('prompt_ends')
    local c = loom.state('piece') + 1
    if c > #ends then
        loom.set_state('done', 1)
        return {}
    end
    local all, prompt = loom.state('prompt_ids'), {}
    for i = (c == 1 and 0 or ends[c - 1]) + 1, ends[c] do prompt[#prompt + 1] = all[i] end
    local wave = soprano_sentence(prompt)
    loom.set_state('piece', c)
    if c == #ends then loom.set_state('done', 1) end
    return wave
end

function stream_finish(inputs)
    return {}
end
