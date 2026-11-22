    -- Every sentence in order, each followed by 0.12 s of silence (upstream's `speak`, GAP_S, the last one
    -- included); the live door yields the sentences alone, as upstream's `stream` does.
    stream_open(inputs)
    wave = {}
    local gap = math.floor(0.12 * SAMPLE_RATE + 0.5)
    while loom.state('done') == 0 do
        for _, v in ipairs(kale_next()) do wave[#wave + 1] = v end
        for _ = 1, gap do wave[#wave + 1] = 0.0 end
    end
    stream_finish({})
