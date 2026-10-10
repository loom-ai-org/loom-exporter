    local _ids = inputs.tokens
    local _n = #_ids
    if _n == 0 then error("sanotts: no phoneme ids to speak") end

    -- `torch.linspace(0, 1, n)` in float32, value for value: ATen fills the first half up from 0 and the
    -- second half down from 1, by a float32 step, each value ONE rounding of a fused multiply-add --
    -- rounding the product first is 1 ulp off at almost every length (checked against torch for every
    -- n below 3000). In doubles the product of a float32 step and an index this size is exact, so one
    -- `to_f32` of the whole expression is that single rounding.
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

    -- An id outside a net's vocabulary becomes schwa, as upstream's runtimes do; the two nets' vocabularies
    -- can differ, so each gets its own copy.
    local function clamped(vocab)
        local fallback = FALLBACK_ID < vocab and FALLBACK_ID or 0
        local out = {}
        for i = 1, _n do
            local id = _ids[i]
            out[i] = (id >= 0 and id < vocab) and id or fallback
        end
        return out
    end
    local _dur_tokens, _ac_tokens = clamped(DUR_VOCAB), clamped(AC_VOCAB)

    -- `[1, 3, n]`, channel-major: position, length hint `log1p(n) / log1p(max_tokens)`, valid (all 1).
    local _dur_features = {}
    local _pos = linspace01(_n)
    -- A float32 tensor over a Python float: the scalar is rounded to float32 before the divide.
    local _hint = to_f32(to_f32(math.log(1.0 + _n)) / to_f32(math.log(1.0 + MAX_TOKENS)))
    for i = 1, _n do
        _dur_features[i] = _pos[i]
        _dur_features[_n + i] = _hint
        _dur_features[2 * _n + i] = 1.0
    end
