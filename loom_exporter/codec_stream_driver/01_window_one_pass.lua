    -- One call over every frame, its noise drawn frame-major as the stream draws it, so the two agree for a
    -- seed (00_window_stream.lua, ADR-104).
    local wave = window_one_pass(inputs)
