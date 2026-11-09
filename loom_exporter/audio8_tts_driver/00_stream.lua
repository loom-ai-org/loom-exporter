-- ===== Audio8-TTS's talker, live (loom.cpp ADR-102): one frame of codes per `stream_feed`. =====
--
-- `stream_open` builds the prompt, prefills the slow stack and draws frame 0; each `stream_feed` feeds the
-- previous frame back (its semantic token and ten codes, one slow-stack step) and draws the next. The work
-- is the one-pass loop's, cut where one iteration's feedback meets the next one's draw, so the draws come
-- in the same order and `infer` -- this stream opened, fed to its end and finished in one call -- gives the
-- codes it always gave.
--
-- Between calls everything is the stream's: the step, the next row, the repetition-aware sampler's window
-- and the knobs; the slow stack's history is the model's KV cache, leased or owned (ADR-099). `piece`
-- counts the draws and `done` is set at EOS or the frame budget.

-- The reference's sampler (`ArkttsLegacyTopKTopPLogitsProcessor`, then the temperature): top-k and top-p
-- are measured on the UNTEMPERED row, top-p's mass over the whole row and without the id that crosses it,
-- and the temperature shapes only the survivors' draw (loom.cpp ADR-095).
local function audio8_options(t, p, hi)
    return {temperature = t, top_k = loom.state('top_k'), top_p = p, top_p_mass = 'row',
            truncation = 'untempered', top_p_crossing = 'exclude', lo = 0, hi = hi}
end

-- The draw half of one step: the semantic token from the slow stack's last row, then the fast stack's
-- nine. Returns the frame's ten codes, or {} at EOS.
local function audio8_draw(step)
    local _temperature, _top_p = loom.state('temperature'), loom.state('top_p')
    local _semantic = loom.sample_row('slow', 0, audio8_options(_temperature, _top_p, EOS_INDEX + 1))
    -- Repetition-aware sampling: a semantic draw that repeats one of the last RAS_WINDOW is redrawn at
    -- the checkpoint's `ras_*` settings. The reference's window starts as zeros AFTER frame 0 -- frame 0's
    -- id is never in it -- and holds real ids from frame 1 on; `recent` is that window without its zeros,
    -- which no index can equal here. Greedy decoding skips it, as the reference does.
    local _recent = loom.state('recent') or {}
    if _temperature > 0 and _semantic ~= EOS_INDEX then
        for _i = 1, #_recent do
            if _recent[_i] == _semantic then
                _semantic = loom.sample_row('slow', 0, audio8_options(RAS_TEMPERATURE, RAS_TOP_P, EOS_INDEX + 1))
                break
            end
        end
    end
    loom.set_state('piece', loom.state('piece') + 1)
    loom.set_state('step', step)
    if _semantic == EOS_INDEX then
        loom.set_state('done', 1)
        return {}
    end
    if step > 0 then
        _recent[#_recent + 1] = _semantic
        if #_recent > RAS_WINDOW then table.remove(_recent, 1) end
        loom.set_state('recent', _recent)
    end

    -- The fast stack: codebook k from the slow hidden state and codebooks 0..k-1, re-run from the start
    -- each time (it never sees more than ten positions, and is not cached).
    local _hidden = {from = 'slow', index = 2}
    local _frame = {_semantic}
    for _k = 1, N_CODEBOOKS - 1 do
        loom.run_subgraph_and_retain('fast', {n_tokens = _k + 1, n_past = 0},
            {slow_hidden = _hidden, codes = _frame,
             position_ids = loom.range(0, _k + 1),
             attention_mask = loom.causal_mask(_k + 1, 0)})
        _frame[_k + 1] = loom.sample_row('fast', 0, audio8_options(_temperature, _top_p, CODEBOOK_SIZE))
    end

    -- The slow stack's next row: the semantic token and the ten codes.
    local _next = {SEMANTIC_BEGIN + _semantic}
    for _g = 1, N_CODEBOOKS do _next[#_next + 1] = _frame[_g] end
    loom.set_state('next_row', _next)
    if step + 1 >= loom.state('max_new') then loom.set_state('done', 1) end
    return _frame
end

function stream_open(inputs)
    -- THE PROMPT: the export's pre-encoded template around the caller's ids, one row per position --
    -- the id, then ten codes. A text row's codes are the absent code, whose embedding row is zero in
    -- the export: the reference's `where(semantic, codebook_sum, 0)`.
    local _rows = {}
    local _n_prompt = 0
    local function _text_rows(_list)
        for _i = 1, #_list do
            _rows[#_rows + 1] = _list[_i]
            for _g = 1, N_CODEBOOKS do _rows[#_rows + 1] = CODEBOOK_SIZE end
            _n_prompt = _n_prompt + 1
        end
    end

    -- THE REFERENCE (voice cloning): its transcript in the system turn, then its frames, each a
    -- semantic row -- `SEMANTIC_BEGIN + code 0` and all ten codes, the same layout as a generated
    -- frame -- then the user turn. Both inputs or neither: the clip alone is not a prompt the model
    -- was trained on.
    local _ref = inputs.reference_codes
    if _ref ~= nil and #_ref > 0 then
        if inputs.reference_tokens == nil or #inputs.reference_tokens == 0 then
            error('reference_codes needs reference_tokens, the transcript of the reference clip')
        end
        if #_ref % N_CODEBOOKS ~= 0 then
            error('reference_codes holds ' .. #_ref .. ' codes, not a whole number of ' .. N_CODEBOOKS ..
                  '-code frames')
        end
        _text_rows(PROMPT_REFERENCE_HEAD)
        _text_rows(inputs.reference_tokens)
        _text_rows(PROMPT_REFERENCE_SPEECH)
        for _at = 0, #_ref - N_CODEBOOKS, N_CODEBOOKS do
            _rows[#_rows + 1] = SEMANTIC_BEGIN + _ref[_at + 1]
            for _g = 1, N_CODEBOOKS do
                local _code = _ref[_at + _g]
                -- The absent code is a real row of the table (zero), so a code outside the codebook
                -- would be read as silence or as the next codebook's row, not refused.
                if _code < 0 or _code >= CODEBOOK_SIZE or _code ~= math.floor(_code) then
                    error('reference code ' .. _code .. ' (frame ' .. (_at / N_CODEBOOKS + 1) ..
                          ', codebook ' .. (_g - 1) .. ') is not in [0, ' .. CODEBOOK_SIZE .. ')')
                end
                _rows[#_rows + 1] = _code
            end
            _n_prompt = _n_prompt + 1
        end
        _text_rows(PROMPT_REFERENCE_TO_TEXT)
    else
        _text_rows(PROMPT_NO_REFERENCE_HEAD)
    end
    _text_rows(inputs.tokens)
    _text_rows(PROMPT_TAIL)
    if _n_prompt >= MAX_POSITIONS then
        error('the prompt is ' .. _n_prompt .. ' rows and this file caches ' .. MAX_POSITIONS ..
              ' positions; shorten the reference or the text')
    end

    loom.run_subgraph_and_retain('embed', {n_tokens = _n_prompt, n_past = 0}, {rows = _rows})
    loom.run_subgraph_and_retain('slow', {n_tokens = _n_prompt, n_past = 0},
        {inputs_embeds = {from = 'embed'},
         position_ids = loom.range(0, _n_prompt),
         attention_mask = loom.causal_mask(_n_prompt, 0)})

    if inputs.seed then loom.seed_rng(inputs.seed) end
    local _max_new = inputs.max_new_tokens or MAX_NEW_TOKENS
    if _max_new > MAX_POSITIONS - _n_prompt then _max_new = MAX_POSITIONS - _n_prompt end
    loom.set_state('temperature', inputs.temperature or TEMPERATURE)
    loom.set_state('top_k', inputs.top_k or TOP_K)
    loom.set_state('top_p', inputs.top_p or TOP_P)
    loom.set_state('max_new', _max_new)
    loom.set_state('n_prompt', _n_prompt)
    loom.set_state('piece', 0)
    loom.set_state('done', 0)
    if _max_new <= 0 then
        loom.set_state('done', 1)
        return {}
    end
    return audio8_draw(0)
end

function stream_feed(inputs)
    if loom.state('done') ~= 0 then return {} end
    local _step = loom.state('step') + 1
    local _past = loom.state('n_prompt') + _step - 1
    loom.run_subgraph_and_retain('embed', {n_tokens = 1, n_past = 0}, {rows = loom.state('next_row')})
    loom.run_subgraph_and_retain('slow', {n_tokens = 1, n_past = _past},
        {inputs_embeds = {from = 'embed'},
         position_ids = loom.range(_past, 1),
         attention_mask = loom.causal_mask(1, _past)})
    return audio8_draw(_step)
end

function stream_finish(inputs)
    return {}
end
