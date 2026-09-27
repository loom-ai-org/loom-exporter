-- VoxCPM2's helpers. Top level, so every fragment below can call them.

-- `solve_euler`'s `(t, dt)` per step, flat, for a step count the export did not tabulate: the swayed
-- linspace, then `t -= dt` and `dt = t - t_span[step + 1]`, accumulated as the reference accumulates
-- them. In doubles, so a few ulps from the reference's f32 -- the default count ships as the
-- `euler_schedule` driver weight instead, which the export computed with the reference's own ops.
local function voxcpm_schedule(n, sway)
    local span = {}
    for i = 0, n do
        local t = 1 - i / n
        span[i + 1] = t + sway * (math.cos(math.pi / 2 * t) - 1 + t)
    end
    local rows = {}
    local t, dt = span[1], span[1] - span[2]
    for step = 1, n do
        rows[2 * step - 1], rows[2 * step] = t, dt
        t = t - dt
        if step < n then dt = t - span[step + 2] end
    end
    return rows
end
