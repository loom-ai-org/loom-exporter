    -- ===== SpeechT5: text -> encoder -> an autoregressive loop over mel frames -> HiFi-GAN. =====
    --
    -- Inputs: `tokens` (required) and the options `stream_open` (01_stream.lua) lists: `threshold`,
    -- `minlenratio`, `maxlenratio`, `speaker`, `seed`, `masks`, `teacher_frames`. Returns the waveform at
    -- 16 kHz.
    --
    -- One pass is the live stream opened, fed to its end and finished inside this call, against the fresh
    -- state the engine gives every one-pass call -- so the one-pass gate is the live path's gate too. The
    -- whole utterance in one feed (the postnet and the vocoder over every frame at once, as the reference
    -- runs them) unless the caller asks for pieces (`piece_frames`), which is the live cut run in one call.
    if #tokens == 0 then error('speecht5: no text ids') end
    if inputs.piece_frames == nil then inputs.piece_frames = 0 end
    stream_open(inputs)
    wave = {}
    while loom.state('done') == 0 do
        local _piece = stream_feed({})
        local _base = #wave
        for _i = 1, #_piece do wave[_base + _i] = _piece[_i] end
    end
    stream_finish({})
