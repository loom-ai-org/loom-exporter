    -- `[1, NOISE_CH, T]`, channel-major. Upstream feeds zeros unless `noise_scale` is named, and then
    -- `torch.randn(NOISE_CH, T) * noise_scale` from the global stream; here that draw is ATen's stream
    -- seeded by `seed`, so a seed renders the same twice. `aten_randn` starts at 16 values.
    local _noise_scale = inputs.noise_scale or 0.0
    local _noise = {}
    local _count = NOISE_CH * _n_frames
    if _noise_scale == 0 then
        for i = 1, _count do _noise[i] = 0.0 end
    else
        local _draw = aten_randn((inputs.seed or 0) % 4294967296, math.max(_count, 16))
        for i = 1, _count do _noise[i] = to_f32(_draw[i] * to_f32(_noise_scale)) end
    end
