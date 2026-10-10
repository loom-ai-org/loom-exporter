    -- How many encoder rows this clip produced: the cross-attention length every later call binds.
    local _n_enc = loom.output_shape('encoder', 1)[2]
    -- The decode starts from `<s>` alone: an English-only model with no task or language tokens.
    local _prompt = {DECODER_START}
