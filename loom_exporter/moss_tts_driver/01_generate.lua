    -- THE PROMPT: the export's pre-encoded template around the caller's text ids. The processor
    -- encodes each template piece on its own and concatenates the ids, and so does the export, so
    -- the driver only concatenates. `language` picks one of the pre-encoded "after the reference"
    -- pieces: 0 (the default) is the template with no language, i is the contract's i-th language.
    local _language = inputs.language or 0
    if _language < 0 or _language + 2 > #PROMPT_AFTER_OFFSETS then
        error('language ' .. _language .. ' is not one this file declares (0 .. ' ..
              (#PROMPT_AFTER_OFFSETS - 2) .. ')')
    end
    local _ids = {}
    for _i = 1, #PROMPT_HEAD do _ids[#_ids + 1] = PROMPT_HEAD[_i] end
    for _i = PROMPT_AFTER_OFFSETS[_language + 1] + 1, PROMPT_AFTER_OFFSETS[_language + 2] do
        _ids[#_ids + 1] = PROMPT_AFTER[_i]
    end
    for _i = 1, #inputs.tokens do _ids[#_ids + 1] = inputs.tokens[_i] end
    for _i = 1, #PROMPT_TAIL do _ids[#_ids + 1] = PROMPT_TAIL[_i] end

    -- One row per prompt id: the text id, then twelve pad codes. The pad code's embedding row is zero
    -- in the export, which is the reference's mask.
    local _width = N_VQ + 1
    local _rows = {}
    for _i = 1, #_ids do
        _rows[#_rows + 1] = _ids[_i]
        for _g = 1, N_VQ do _rows[#_rows + 1] = PAD_CODE end
    end
    local _n_prompt = #_ids

    loom.run_subgraph_and_retain('embed', {n_tokens = _n_prompt, n_past = 0}, {rows = _rows})
    loom.run_subgraph_and_retain('global', {n_tokens = _n_prompt, n_past = 0},
        {inputs_embeds = {from = 'embed'},
         position_ids = loom.range(0, _n_prompt),
         attention_mask = loom.causal_mask(_n_prompt, 0)})

    if inputs.seed then loom.seed_rng(inputs.seed) end
    local _a_temperature = inputs.temperature or AUDIO_TEMPERATURE
    local _a_top_k = inputs.top_k or AUDIO_TOP_K
    local _a_top_p = inputs.top_p or AUDIO_TOP_P
    local _t_temperature = inputs.text_temperature or TEXT_TEMPERATURE
    local _t_top_k = inputs.text_top_k or TEXT_TOP_K
    local _t_top_p = inputs.text_top_p or TEXT_TOP_P
    local _max_new = inputs.max_new_tokens or MAX_NEW_TOKENS
    -- `draws` pins every uniform: N_VQ + 1 per frame (continue/stop, then each codebook), in the
    -- reference's draw order, used or not. It is what makes a SAMPLED decode gateable against the
    -- reference (ADR-047); absent, the engine's own stream draws.
    local _draws = inputs.draws
    if _max_new > MAX_POSITIONS - _n_prompt then _max_new = MAX_POSITIONS - _n_prompt end

    -- The merged head: twelve audio windows of CODEBOOK_SIZE, then the two-way continue/stop pair.
    local _stop_lo = N_VQ * CODEBOOK_SIZE
    local _hidden = {from = 'global'}
    local _codes = {}
    for _step = 0, _max_new - 1 do
        loom.run_subgraph_and_retain('local_first', {n_tokens = 1, n_past = 0},
            {global_hidden = _hidden, position_ids = {0}, attention_mask = {0.0}})
        -- Continue or stop FIRST, then codebook 0 from the same row: the reference's draw order,
        -- which is what a pinned-draw oracle has to reproduce.
        local _base = _step * (N_VQ + 1)
        local _decision = loom.sample_row('local_first', 0, {
            temperature = _t_temperature, top_k = _t_top_k, top_p = _t_top_p,
            lo = _stop_lo, hi = _stop_lo + 2, uniform = _draws and _draws[_base + 1]})
        if _decision ~= _stop_lo then break end

        local _prefix = {}
        local _module = 'local_first'
        for _g = 0, N_VQ - 1 do
            -- A windowed draw returns the ABSOLUTE index, which is the merged embedding table's row
            -- for this code: `_prefix` holds those unchanged and the offset comes off once, below.
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
        local _next = {SLOT_ID}
        for _g = 0, N_VQ - 1 do
            local _code = _prefix[_g + 1] - _g * CODEBOOK_SIZE
            _codes[#_codes + 1] = _code
            _next[#_next + 1] = _code
        end
        loom.run_subgraph_and_retain('embed', {n_tokens = 1, n_past = 0}, {rows = _next})
        loom.run_subgraph_and_retain('global', {n_tokens = 1, n_past = _n_prompt + _step},
            {inputs_embeds = {from = 'embed'},
             position_ids = loom.range(_n_prompt + _step, 1),
             attention_mask = loom.causal_mask(1, _n_prompt + _step)})
    end
