    -- ===== The flow LM: text -> latents, one continuous 32-d frame per step, chunk by chunk. =====
    --
    -- One pass is the live stream (00_stream.lua) opened, fed to its end and finished inside this call,
    -- against the fresh state the engine gives every one-pass call -- so the one-pass gate is the live
    -- path's gate too. `inputs.tokens` and every knob go to `stream_open` as they came.
    if #tokens == 0 then error('pocket-tts: no text ids') end
    stream_open(inputs)
    wave = {}
    while loom.state('done') == 0 do
        local _piece = stream_feed({})
        local _base = #wave
        for _i = 1, #_piece do wave[_base + _i] = _piece[_i] end
    end
    stream_finish({})
