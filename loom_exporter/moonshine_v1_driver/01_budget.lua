    local _wav = inputs.waveform
    local _n_real = (inputs.length and inputs.length[1]) or #_wav

    -- The card's budget, less the start token. A default budget past the KV cache (the decoder's
    -- `max_position_embeddings`) is refused BEFORE the encoder runs: the reference decodes on past it,
    -- so capping it here would cut the transcript short with no error. A caller's own budget is only
    -- held to the cache.
    local _budget = math.floor(_n_real * TOKENS_PER_SECOND / SAMPLE_RATE)
    if inputs.max_new_tokens == nil then
        if _budget > MAX_POSITIONS then
            error(string.format("moonshine: %.1f s of audio is a %d-token budget; the model decodes at "
                .. "most %d (%.1f s). Split the audio (a VAD finds the pauses) and transcribe each part.",
                _n_real / SAMPLE_RATE, _budget, MAX_POSITIONS,
                MAX_POSITIONS / TOKENS_PER_SECOND))
        end
        inputs.max_new_tokens = _budget - 1
    end
    local _max_new = math.min(inputs.max_new_tokens, MAX_POSITIONS - 1)
    inputs.max_new_tokens = _max_new
    -- A clip under 0.31 s has no budget at all; one under MIN_SAMPLES (56 ms) has no encoder row.
    if _max_new <= 0 then return {} end
    if #_wav < MIN_SAMPLES then
        error(string.format("moonshine: %d samples is shorter than the encoder's stem accepts (%d).",
            #_wav, MIN_SAMPLES))
    end
