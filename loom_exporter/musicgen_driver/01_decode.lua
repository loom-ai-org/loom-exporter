    -- One pass is the live stream (00_stream.lua) opened, fed to its end and finished inside this call,
    -- against the fresh state the engine gives every one-pass call -- so the one-pass gate is the live
    -- path's gate too. A FLAT, frame-major array: N_CODEBOOKS codes a frame, delay already undone.
    if inputs.tokens == nil then error('musicgen: no text ids') end
    local _codes = {}
    stream_open(inputs)
    while loom.state('done') == 0 do
        for _, c in ipairs(stream_feed({})) do _codes[#_codes + 1] = c end
    end
    stream_finish({})
