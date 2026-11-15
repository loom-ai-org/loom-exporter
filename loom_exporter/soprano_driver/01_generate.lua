    -- ===== Soprano: one generation per sentence, its hidden rows decoded, the audio concatenated. =====
    --
    -- Inputs: `tokens` (required), and the sampler's knobs, each defaulting to the reference's own:
    -- `temperature` (0.001), `top_k` (50), `top_p` (0.95), `repetition_penalty` (1.2, over the prompt
    -- and every id drawn so far), `max_new_tokens` (512 per sentence) and `seed`. Returns the waveform
    -- at 32 kHz.
    --
    -- One pass is the live stream (00_stream.lua) opened, fed to its end and finished inside this call,
    -- against the fresh state the engine gives every one-pass call -- so the one-pass gate is the live
    -- path's gate too. A sentence per feed, the reference's non-streaming decode, unless the caller asks
    -- for rows (`piece_tokens`), which is the live cut run in one call.
    if #tokens == 0 then error('soprano: no text ids') end
    if inputs.piece_tokens == nil then inputs.piece_tokens = 0 end
    stream_open(inputs)
    wave = {}
    while loom.state('done') == 0 do
        local _piece = stream_feed({})
        local _base = #wave
        for _i = 1, #_piece do wave[_base + _i] = _piece[_i] end
    end
    stream_finish({})
    if #wave == 0 then error('soprano: the LM produced no audio rows') end
