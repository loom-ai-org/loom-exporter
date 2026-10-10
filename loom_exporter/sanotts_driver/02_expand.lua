    -- `predict_durations`: `round(exp(log_d).clamp_min(1) * length_scale)`, clamped to [1, max], with
    -- `torch.round`'s ties to even.
    local _scale = to_f32(inputs.length_scale or LENGTH_SCALE)
    if _scale <= 0 then error("sanotts: length_scale must be positive") end
    local _dur = {}
    local _n_frames, _max_d = 0, 1
    for i = 1, _n do
        local e = to_f32(math.exp(_log_duration[i]))
        if e < 1.0 then e = 1.0 end
        local d = round_half_to_even(to_f32(e * _scale))
        if d < 1 then d = 1 elseif d > MAX_DURATION then d = MAX_DURATION end
        _dur[i] = d
        _n_frames = _n_frames + d
        if d > _max_d then _max_d = d end
    end

    -- The token stack's features, `[1, 2, n]`: position, and `log1p(d) / log1p(max d)`.
    local _token_features = {}
    local _tpos = linspace01(_n)
    local _log_max = to_f32(math.log(1.0 + _max_d))
    for i = 1, _n do
        _token_features[i] = _tpos[i]
        _token_features[_n + i] = to_f32(to_f32(math.log(1.0 + _dur[i])) / _log_max)
    end

    -- `expand_features`: which token each frame repeats (the gather's index), and `[1, 3, T]` --
    -- the frame's position in the utterance, its token's position, its position inside its token.
    local _frame_index, _frame_features = {}, {}
    local _fpos = linspace01(_n_frames)
    local _token_count = math.max(_n - 1, 1)
    local f = 0
    for i = 1, _n do
        local d = _dur[i]
        local tpos = to_f32((i - 1) / _token_count)
        for j = 0, d - 1 do
            f = f + 1
            _frame_index[f] = i - 1
            _frame_features[f] = _fpos[f]
            _frame_features[_n_frames + f] = tpos
            _frame_features[2 * _n_frames + f] = d == 1 and 0.0 or to_f32(j / (d - 1))
        end
    end
