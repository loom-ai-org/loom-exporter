    -- One pass: a stream opened, fed the whole clip and finished inside this call, against the fresh
    -- state the engine gives every one-pass call -- so the one-pass gate is the live path's gate too.
    stream_open(inputs)
    tokens = stream_feed({waveform = waveform})
    for _, k in ipairs(stream_finish(inputs)) do tokens[#tokens + 1] = k end
