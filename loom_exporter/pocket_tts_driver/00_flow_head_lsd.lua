-- The flow head of the full-size models: LSD, two time conditions. Top level and ONE of two files the
-- export ships (00_flow_head_one_step.lua is the other), so the driver's only `flow_head` call matches
-- the graph it runs -- the export checks every call site's inputs against its topology, both ways.
--
-- `lsd_decode`: `n_steps` updates from s = i/n to t = (i+1)/n. The first scales the unit draw by
-- sqrt(temperature); the graph does both products, in f32, as the reference does.
local function pocket_flow_head(x, std, n_steps)
    local scale = std
    for i = 0, n_steps - 1 do
        loom.run_subgraph_and_retain('flow_head', {n_tokens = 1, n_past = 0},
            {c = {from = 'lm', index = 1}, s = {i / n_steps}, t = {(i + 1) / n_steps},
             x = x, x_scale = {scale}, n_steps = {n_steps}})
        x = loom.get_output('flow_head', 1)
        scale = 1.0
    end
    return x
end
