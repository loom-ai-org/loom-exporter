-- ===== MOSS-TTS, live (loom.cpp ADR-102): one frame of codes per `stream_feed`. =====
--
-- `stream_open` builds the prompt, prefills the global stack and draws frame 0; each `stream_feed` feeds the
-- previous frame back (the assistant slot id and its twelve codes, one global step) and draws the next. The
-- work is the one-pass loop's, cut where one iteration's feedback meets the next one's draw, so the draws
-- come in the same order -- pinned ones included -- and `infer` (this stream opened, fed to its end and
-- finished in one call) gives the codes it always gave.
--
-- Between calls everything is the stream's: the step, the next row, a caller's pinned draws and the knobs;
-- the global stack's history is the model's KV cache, leased or owned (ADR-099). `piece` counts the draws
-- and `done` is set at the stop decision or the frame budget.

-- The draw half of one step: continue-or-stop, then the twelve codebooks from the local stack. Returns the
-- frame's twelve codes, or {} at the stop.
local function moss_draw(step)
    local _draws = loom.state('draws')
    -- The merged head: twelve audio windows of CODEBOOK_SIZE, then the two-way continue/stop pair.
    local _stop_lo = N_VQ * CODEBOOK_SIZE
    local _hidden = {from = 'global'}
    loom.run_subgraph_and_retain('local_first', {n_tokens = 1, n_past = 0},
        {global_hidden = _hidden, position_ids = {0}, attention_mask = {0.0}})
    -- Continue or stop FIRST, then codebook 0 from the same row: the reference's draw order, which is what
    -- a pinned-draw oracle has to reproduce.
    local _base = step * (N_VQ + 1)
    local _decision = loom.sample_row('local_first', 0, {
        temperature = loom.state('t_temperature'), top_k = loom.state('t_top_k'), top_p = loom.state('t_top_p'),
        lo = _stop_lo, hi = _stop_lo + 2, uniform = _draws and _draws[_base + 1]})
    loom.set_state('piece', loom.state('piece') + 1)
    loom.set_state('step', step)
    if _decision ~= _stop_lo then
        loom.set_state('done', 1)
        return {}
    end

    local _a_temperature, _a_top_k, _a_top_p = loom.state('a_temperature'), loom.state('a_top_k'), loom.state('a_top_p')
    local _prefix = {}
    local _module = 'local_first'
    for _g = 0, N_VQ - 1 do
        -- A windowed draw returns the ABSOLUTE index, which is the merged embedding table's row for this
        -- code: `_prefix` holds those unchanged and the offset comes off once, below.
        local _row = loom.sample_row(_module, 0, {
            temperature = _a_temperature, top_k = _a_top_k, top_p = _a_top_p,
            lo = _g * CODEBOOK_SIZE, hi = (_g + 1) * CODEBOOK_SIZE,
            uniform = _draws and _draws[_base + _g + 2]})
        _prefix[#_prefix + 1] = _row
        if _g < N_VQ - 1 then
            local _n = #_prefix + 1
            loom.run_subgraph_and_retain('local_steps', {n_tokens = _n, n_past = 0},
                {global_hidden = _hidden, rows = _prefix,
                 position_ids = loom.range(0, _n),
                 attention_mask = loom.causal_mask(_n, 0)})
            _module = 'local_steps'
        end
    end

    -- This frame's codes, and the global stack's next row: the assistant slot id plus them.
    local _codes, _next = {}, {SLOT_ID}
    for _g = 0, N_VQ - 1 do
        local _code = _prefix[_g + 1] - _g * CODEBOOK_SIZE
        _codes[#_codes + 1] = _code
        _next[#_next + 1] = _code
    end
    loom.set_state('next_row', _next)
    if step + 1 >= loom.state('max_new') then loom.set_state('done', 1) end
    return _codes
end

function stream_open(inputs)
    -- THE PROMPT: the export's pre-encoded template around the caller's text ids. The processor
    -- encodes each template piece on its own and concatenates the ids, and so does the export, so
    -- the driver only concatenates. `language` picks one of the pre-encoded "after the reference"
    -- pieces: 0 (the default) is the template with no language, i is the contract's i-th language.
    local _language = inputs.language or 0
    if _language < 0 or _language + 2 > #PROMPT_AFTER_OFFSETS then
        error('language ' .. _language .. ' is not one this file declares (0 .. ' ..
              (#PROMPT_AFTER_OFFSETS - 2) .. ')')
    end

    -- One row per prompt id: the text id, then twelve pad codes. The pad code's embedding row is zero
    -- in the export, which is the reference's mask.
    local _rows = {}
    local _n_prompt = 0
    local function _text_rows(_list, _lo, _hi)
        for _i = _lo or 1, _hi or #_list do
            _rows[#_rows + 1] = _list[_i]
            for _g = 1, N_VQ do _rows[#_rows + 1] = PAD_CODE end
            _n_prompt = _n_prompt + 1
        end
    end

    -- THE REFERENCES (voice cloning): where the template says "None", each reference as an
    -- `<audio_start>` row, one row per frame -- the USER slot id and its twelve codes -- and an
    -- `<audio_end>` row, back to back with no separator (the reference's direct clone path). A voice
    -- file sets both inputs; `reference_frames` absent means one reference of every frame given.
    local _ref = inputs.reference_codes
    _text_rows(PROMPT_HEAD)
    if _ref == nil or #_ref == 0 then
        _text_rows(PROMPT_NO_REFERENCE)
    else
        if #_ref % N_VQ ~= 0 then
            error('reference_codes holds ' .. #_ref .. ' codes, not a whole number of ' .. N_VQ ..
                  '-code frames')
        end
        local _frames = inputs.reference_frames or {#_ref / N_VQ}
        local _total = 0
        for _r = 1, #_frames do _total = _total + _frames[_r] end
        if _total * N_VQ ~= #_ref then
            error('reference_frames adds up to ' .. _total .. ' frames and reference_codes holds ' ..
                  (#_ref / N_VQ))
        end
        local _at = 0
        for _r = 1, #_frames do
            _text_rows({AUDIO_START_ID})
            for _f = 1, _frames[_r] do
                _rows[#_rows + 1] = USER_SLOT_ID
                for _g = 1, N_VQ do
                    local _code = _ref[_at + _g]
                    -- The pad code is a real row of the table (zero), so a code outside the codebook
                    -- would be read as "absent" or as another codebook's row, not refused.
                    if _code < 0 or _code >= CODEBOOK_SIZE or _code ~= math.floor(_code) then
                        error('reference code ' .. _code .. ' (frame ' .. (_at / N_VQ + 1) ..
                              ', codebook ' .. (_g - 1) .. ') is not in [0, ' .. CODEBOOK_SIZE .. ')')
                    end
                    _rows[#_rows + 1] = _code
                end
                _at = _at + N_VQ
                _n_prompt = _n_prompt + 1
            end
            _text_rows({AUDIO_END_ID})
        end
    end
    _text_rows(PROMPT_AFTER, PROMPT_AFTER_OFFSETS[_language + 1] + 1, PROMPT_AFTER_OFFSETS[_language + 2])
    _text_rows(inputs.tokens)
    _text_rows(PROMPT_TAIL)
    if _n_prompt >= MAX_POSITIONS then
        error('the prompt is ' .. _n_prompt .. ' rows and this file caches ' .. MAX_POSITIONS ..
              ' positions; shorten the references or the text')
    end

    loom.run_subgraph_and_retain('embed', {n_tokens = _n_prompt, n_past = 0}, {rows = _rows})
    loom.run_subgraph_and_retain('global', {n_tokens = _n_prompt, n_past = 0},
        {inputs_embeds = {from = 'embed'},
         position_ids = loom.range(0, _n_prompt),
         attention_mask = loom.causal_mask(_n_prompt, 0)})

    if inputs.seed then loom.seed_rng(inputs.seed) end
    loom.set_state('a_temperature', inputs.temperature or AUDIO_TEMPERATURE)
    loom.set_state('a_top_k', inputs.top_k or AUDIO_TOP_K)
    loom.set_state('a_top_p', inputs.top_p or AUDIO_TOP_P)
    loom.set_state('t_temperature', inputs.text_temperature or TEXT_TEMPERATURE)
    loom.set_state('t_top_k', inputs.text_top_k or TEXT_TOP_K)
    loom.set_state('t_top_p', inputs.text_top_p or TEXT_TOP_P)
    -- `draws` pins every uniform: N_VQ + 1 per frame (continue/stop, then each codebook), in the
    -- reference's draw order, used or not. It is what makes a SAMPLED decode gateable against the
    -- reference (ADR-047); absent, the engine's own stream draws.
    if inputs.draws ~= nil then loom.set_state('draws', inputs.draws) end
    local _max_new = inputs.max_new_tokens or MAX_NEW_TOKENS
    if _max_new > MAX_POSITIONS - _n_prompt then _max_new = MAX_POSITIONS - _n_prompt end
    loom.set_state('max_new', _max_new)
    loom.set_state('n_prompt', _n_prompt)
    loom.set_state('piece', 0)
    loom.set_state('done', 0)
    if _max_new <= 0 then
        loom.set_state('done', 1)
        return {}
    end
    return moss_draw(0)
end

function stream_feed(inputs)
    if loom.state('done') ~= 0 then return {} end
    local _step = loom.state('step') + 1
    local _past = loom.state('n_prompt') + _step - 1
    loom.run_subgraph_and_retain('embed', {n_tokens = 1, n_past = 0}, {rows = loom.state('next_row')})
    loom.run_subgraph_and_retain('global', {n_tokens = 1, n_past = _past},
        {inputs_embeds = {from = 'embed'},
         position_ids = loom.range(_past, 1),
         attention_mask = loom.causal_mask(1, _past)})
    return moss_draw(_step)
end

function stream_finish(inputs)
    return {}
end
