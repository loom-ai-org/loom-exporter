    -- `[1, NOISE_CH, T]`, channel-major: one `torch.randn(NOISE_CH * T)` reshaped, as upstream draws it.
    local _seed = inputs.seed
    if _seed == nil or _seed == 0 then _seed = DEFAULT_SEED end
    local _noise = aten_randn(_seed % 4294967296, NOISE_CH * _n_frames)
