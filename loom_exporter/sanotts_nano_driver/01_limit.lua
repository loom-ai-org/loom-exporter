    -- Upstream refuses an utterance past the duration net's training length (`max_tokens`, BOS and EOS
    -- included) rather than extrapolate its length hint, and a raw id outside the table has no row.
    if #inputs.tokens > MAX_TOKENS then
        error(string.format("sanotts: %d phoneme ids; this voice takes at most %d per call (BOS and EOS "
            .. "included). Split the text at its sentences and synthesise each.", #inputs.tokens, MAX_TOKENS))
    end
    for i = 1, #inputs.tokens do
        local id = inputs.tokens[i]
        if id < 0 or id >= DUR_VOCAB then
            error(string.format("sanotts: phoneme id %d is outside this voice's %d-symbol table", id, DUR_VOCAB))
        end
    end
