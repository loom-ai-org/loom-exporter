    -- Every sentence in order, joined (upstream's `Paradee.__call__` concatenates them as they are).
    stream_open(inputs)
    wave = {}
    while loom.state('done') == 0 do
        for _, v in ipairs(paradee_next()) do wave[#wave + 1] = v end
    end
    stream_finish({})
