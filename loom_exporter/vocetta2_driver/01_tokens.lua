    -- Upstream refuses an utterance past the duration net's training length (`max_tokens`, BOS and EOS
    -- included) rather than extrapolate its length hint, and a raw id outside the table has no row.
    local _tokens = inputs.tokens
    local _n = #_tokens
    if _n == 0 then error("vocetta2: no phoneme ids to speak") end
    if _n > MAX_TOKENS then
        error(string.format("vocetta2: %d phoneme ids; this model takes at most %d per call (BOS and EOS "
            .. "included). Split the text at its sentences and synthesise each.", _n, MAX_TOKENS))
    end
    for i = 1, _n do
        local id = _tokens[i]
        if id < 0 or id >= VOCAB then
            error(string.format("vocetta2: phoneme id %d is outside this model's %d-symbol table", id, VOCAB))
        end
    end

    -- `torch.linspace(0, 1, n)` in float32, value for value: ATen fills the first half up from 0 and the
    -- second half down from 1, by a float32 step, each value ONE rounding of a fused multiply-add (see
    -- sanotts_driver/01_tokens.lua, where this was checked against torch for every n below 3000).
    -- Upstream's `n > 1 else zeros(1)` is the `count == 1` branch.
    local function linspace01(count)
        local out = {}
        if count == 1 then out[1] = 0.0 return out end
        local step = to_f32(1.0 / (count - 1))
        local half = math.floor(count / 2)
        for i = 0, count - 1 do
            if i < half then out[i + 1] = to_f32(step * i)
            else out[i + 1] = to_f32(1.0 - step * (count - 1 - i)) end
        end
        return out
    end

    -- `[1, 3, n]`, channel-major: position, length hint, ones. The hint is `math.log1p(n) /
    -- math.log1p(max_tokens)` in Python doubles, rounded to float32 once by `torch.full`.
    local _dur_features = {}
    local _pos = linspace01(_n)
    local _hint = to_f32(math.log(1.0 + _n) / math.log(1.0 + MAX_TOKENS))
    for i = 1, _n do
        _dur_features[i] = _pos[i]
        _dur_features[_n + i] = _hint
        _dur_features[2 * _n + i] = 1.0
    end
