    -- ===== The flow LM: text -> latents, one continuous 32-d frame per step, chunk by chunk. =====
    --
    -- One pass is the live stream (00_stream.lua) opened, fed to its end and finished inside this call,
    -- against the fresh state the engine gives every one-pass call -- so the one-pass gate is the live
    -- path's gate too. `inputs.tokens` and every knob go to `stream_open` as they came, and a whole chunk
    -- per feed unless the caller asks for frames (`piece_frames`), which is the live cut run in one call.
    if #tokens == 0 then error('pocket-tts: no text ids') end
    if inputs.piece_frames == nil then inputs.piece_frames = 0 end
    stream_open(inputs)
    wave = {}
    while loom.state('done') == 0 do
        local _piece = stream_feed({})
        local _base = #wave
        for _i = 1, #_piece do wave[_base + _i] = _piece[_i] end
    end
    stream_finish({})
