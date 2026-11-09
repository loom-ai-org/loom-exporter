    -- One pass is the live stream (00_stream.lua) opened, fed to its end and finished inside this call,
    -- against the fresh state the engine gives every one-pass call -- so the one-pass gate is the live
    -- path's gate too. Frame-major, sixteen codes a frame.
    if inputs.tokens == nil or #inputs.tokens == 0 then error('qwen3-tts: no text ids') end
    local _codes = {}
    for _, c in ipairs(stream_open(inputs)) do _codes[#_codes + 1] = c end
    while loom.state('done') == 0 do
        for _, c in ipairs(stream_feed({})) do _codes[#_codes + 1] = c end
    end
    stream_finish({})
