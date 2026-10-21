    -- Upstream's `exp(log_d).round().clamp_min(1)` (no upper clamp on this path), `torch.round`'s ties
    -- to even. `length_scale` multiplies the float32 exp before the rounding; at 1.0 that is upstream.
    local _scale = to_f32(inputs.length_scale or 1.0)
    if _scale <= 0 then error("vocetta2: length_scale must be positive") end
    local _dur = {}
    local _n_frames, _max_d = 0, 1
    for i = 1, _n do
        local d = round_half_to_even(to_f32(to_f32(math.exp(_log_duration[i])) * _scale))
        if d < 1 then d = 1 end
        _dur[i] = d
        _n_frames = _n_frames + d
        if d > _max_d then _max_d = d end
    end

    -- The token stack's features, `[1, 2, n]`: position, and `log1p(d) / log1p(max d)` -- a float32
    -- tensor over a Python float, so the divisor is rounded to float32 first.
    local _token_features = {}
    local _tpos = linspace01(_n)
    local _log_max = to_f32(math.log(1.0 + _max_d))
    for i = 1, _n do
        _token_features[i] = _tpos[i]
        _token_features[_n + i] = to_f32(to_f32(math.log(1.0 + _dur[i])) / _log_max)
    end

    -- The `repeat_interleave`: which token each frame repeats (the gather's index), and `[1, 3, T]` --
    -- the frame's position in the utterance, its token's position (the token linspace, repeated), and
    -- its position inside its token (`linspace(0, 1, d)`, or 0 for a one-frame token).
    local _frame_index, _frame_features = {}, {}
    local _fpos = linspace01(_n_frames)
    local _inner = {}
    local f = 0
    for i = 1, _n do
        local d = _dur[i]
        if _inner[d] == nil then _inner[d] = linspace01(d) end
        local inner = _inner[d]
        for j = 1, d do
            f = f + 1
            _frame_index[f] = i - 1
            _frame_features[f] = _fpos[f]
            _frame_features[_n_frames + f] = _tpos[i]
            _frame_features[2 * _n_frames + f] = inner[j]
        end
    end
