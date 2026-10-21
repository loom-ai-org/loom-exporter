    -- Upstream's DC blocker: the first DC_BLOCK_TAPS taps of H(z) = (1 - z^-1) / (1 - R z^-1), i.e.
    -- h[0] = 1, h[k] = (R - 1) R^(k-1), convolved with the waveform. Run as the recursion, minus the
    -- tail the truncation drops: past tap K the impulse response is (R - 1) R^(K-1) times a one-pole
    -- sum s[m] = x[m] + R s[m-1], so y_trunc[n] = y_iir[n] - (R - 1) R^(K-1) s[n - K]. Exact in exact
    -- arithmetic; without the correction the result differs from upstream's by up to R^K = 1.6e-5 of the
    -- signal. Upstream does not clip.
    local waveform = {}
    local _tail = (DC_BLOCK_R - 1.0) * DC_BLOCK_R ^ (DC_BLOCK_TAPS - 1)
    local _s = {}
    local _y = 0.0
    for i = 1, #_wave do
        local x = _wave[i]
        _s[i] = x + DC_BLOCK_R * (_s[i - 1] or 0.0)
        -- y_iir[n] = x[n] + (R - 1) * sum_{k>=1} R^(k-1) x[n-k] = x[n] + (R - 1) * s[n-1]
        _y = x + (DC_BLOCK_R - 1.0) * (_s[i - 1] or 0.0)
        local back = i - DC_BLOCK_TAPS
        if back >= 1 then _y = _y - _tail * _s[back] end
        waveform[i] = to_f32(_y)
    end
