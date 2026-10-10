    -- The language prompt the encoder conditions on: the id the engine resolved from `language=`
    -- through the contract's language table, else the checkpoint's own default (`auto`, which
    -- identifies the language itself).
    local _prompt = {inputs.language or DEFAULT_PROMPT}
