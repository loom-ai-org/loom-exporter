-- The flow head of the small 2026-10 models: `drifting` (`head(noise)`) or `pointwise`
-- (`noise + head(noise)`), which the graph decides. No time condition and one step by construction, so
-- `n_steps` is accepted and ignored. Top level and ONE of two files the export ships
-- (00_flow_head_lsd.lua is the other): see there.
local function pocket_flow_head(x, std, n_steps)
    loom.run_subgraph_and_retain('flow_head', {n_tokens = 1, n_past = 0},
        {c = {from = 'lm', index = 1}, x = x, x_scale = {std}})
    return loom.get_output('flow_head', 1)
end
