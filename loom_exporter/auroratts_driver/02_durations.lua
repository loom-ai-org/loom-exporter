    -- `TinyTTS.infer`: dur = ((exp(logd) - 1) / speed).round().clamp(min=1), every step in float32 as
    -- torch takes it, and `round` is half-to-even. The Gaussian centres are `cumsum(dur) - dur / 2`,
    -- exact in float32 for integer durations; frame t sits at t + 0.5.
    local _logd = loom.get_output('predict', 1)
    local _speed = to_f32(inputs.speed or 1.0)
    local _dur, _centers = {}, {}
    local _n_frames = 0
    for i = 1, _n do
        local d = round_half_to_even(to_f32(to_f32(to_f32(math.exp(_logd[i])) - 1) / _speed))
        if d < 1 then d = 1 end
        _dur[i] = d
        _n_frames = _n_frames + d
        _centers[i] = _n_frames - d / 2
    end
    local _frame_pos = {}
    for t = 1, _n_frames do _frame_pos[t] = (t - 1) + 0.5 end
    local _pitch_shift = {inputs.pitch_shift or 0.0}
