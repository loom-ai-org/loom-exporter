    -- The DC blocker upstream runs after the iSTFT, H(z) = (1 - z^-1) / (1 - R z^-1), as its recursion
    -- (upstream convolves a 4096-tap truncation of the same impulse response; the tail it drops is
    -- below R^4096 = 1.6e-5). Then clipped to [-1, 1], as upstream returns it.
    local waveform = {}
    local _prev_x, _prev_y = 0.0, 0.0
    for i = 1, #_wave do
        local x = _wave[i]
        local y = x - _prev_x + DC_BLOCK_R * _prev_y
        _prev_x, _prev_y = x, y
        if y > 1.0 then y = 1.0 elseif y < -1.0 then y = -1.0 end
        waveform[i] = y
    end
