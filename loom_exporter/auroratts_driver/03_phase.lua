    -- The harmonic phase at the first sample of each frame, in double precision (upstream's float64
    -- cumsum, auroratts_export.py's module docstring). Inside frame j the sample-rate f0 runs linearly
    -- from f0[j] to f0[j+1] over HOP samples, so the frame adds HOP*f0[j] + (HOP-1)/2*(f0[j+1]-f0[j])
    -- cycles' worth of f0, times 2*pi/sr. `_phase0[j+1]` is frame j's start; `_phase0[1]` is never read
    -- (the synth graph's box kernel takes each sample's value from the NEXT entry).
    local _f0 = loom.get_output('decode', 1)
    local _two_pi = 2 * math.pi
    local _w = _two_pi / SAMPLE_RATE
    local _phase0 = {0.0}
    local _p = 0.0
    for j = 1, _n_frames - 1 do                  -- frame j - 1, 0-based
        _phase0[j + 1] = _p
        _p = math.fmod(_p + (HOP * _f0[j] + (HOP - 1) / 2 * (_f0[j + 1] - _f0[j])) * _w, _two_pi)
    end
    local _seed = inputs.seed or 0
    local _noise = aten_randn(_seed % 4294967296, (_n_frames - 1) * HOP)
