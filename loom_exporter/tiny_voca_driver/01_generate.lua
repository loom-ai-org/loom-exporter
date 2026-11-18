    -- One pass: every utterance, joined, peak-limited ONCE (`speak_text`); the stream's pieces in order.
    stream_open(inputs)
    wave = {}
    while loom.state('done') == 0 do
        for _, v in ipairs(voca_next()) do wave[#wave + 1] = v end
    end
    if inputs.no_peak_limit == nil or inputs.no_peak_limit == 0 then wave = voca_limit(wave) end
    stream_finish({})
