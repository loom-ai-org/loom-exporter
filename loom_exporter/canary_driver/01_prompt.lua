    -- How many encoder rows this clip produced -- one number, read without copying the tensor. It is
    -- the cross-attention length every later call binds, and the base of NeMo's token budget below.
    local _n_enc = loom.output_shape('encoder', 1)[2]

    -- The canary2 prompt with its two language slots filled. The SOURCE defaults to what the
    -- checkpoint's own default turn names. The TARGET, in order: an explicit `target_language`; else
    -- `task` (0 = transcribe, i.e. the source; otherwise the export's contract maps `translate` to
    -- <|en|>); else DEFAULT_TARGET, which is English -- a product decision recorded in the export, and
    -- for English audio it is simply the transcript.
    local _source = inputs.language or DEFAULT_LANGUAGE
    local _target = inputs.target_language
    if _target == nil then
        local _task = inputs.task
        if _task == nil then
            _target = DEFAULT_TARGET
        elseif _task == 0 then
            _target = _source
        else
            _target = _task
        end
    end
    local _prompt = {}
    for i = 1, #PROMPT do _prompt[i] = PROMPT[i] end
    _prompt[SOURCE_SLOT] = _source
    _prompt[TARGET_SLOT] = _target

    -- NeMo's own budget for an encoder-decoder search: at most `max_generation_delta` tokens more than
    -- the encoder has frames, inside the decoder's positional table, minus the prompt already in it.
    local _max_new = math.min(MAX_POSITIONS, _n_enc + MAX_GENERATION_DELTA) - #_prompt
    if inputs.max_new_tokens == nil then inputs.max_new_tokens = _max_new end
