-- Chatterbox's helpers. Top level, so every fragment below can call them.

-- A voice tensor: the caller's when it passes one, else the built-in voice the export shipped as a
-- driver weight (`conds.pt`). Read through `t3_step_embed`, which is only a name here -- driver weights
-- live in the one namespace every module reads.
local function chatterbox_voice(override, name)
    if override ~= nil then return override end
    return loom.get_weight('t3_step_embed', name)
end

-- `1 - cos(pi/2 * t)` over `n_steps + 1` points of a linspace: `CausalConditionalCFM.forward`'s cosine
-- schedule, which is why the sampler takes the CALLER's times.
local function chatterbox_cosine_times(n_steps)
    local times = {}
    for step = 0, n_steps do
        times[step + 1] = 1 - math.cos(step / n_steps * 0.5 * math.pi)
    end
    return times
end

local function chatterbox_zeros(n)
    local z = {}
    for i = 1, n do z[i] = 0.0 end
    return z
end
