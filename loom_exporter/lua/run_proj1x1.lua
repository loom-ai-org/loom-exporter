-- One 1x1 projection over a feature sequence a previous module retained, and the result stays
-- retained too: the caller gets a module NAME, not a table.
--
-- **Both ends used to cross for nothing.** The input came from `run_resblk_stack` as Lua rows and was
-- converted straight back to layout A here; the output went to the vocoder untouched. Neither is a
-- value the host reads -- `output_store.h`'s rule -- so both are references now, and the whole
-- F0/N branch runs from the shared BiLSTM to the vocoder without a table.
local function run_proj1x1(name, from_module)
    local shape = loom.output_shape(from_module, 1)
    loom.run_subgraph_and_retain(name, {n_tokens = shape[1], n_past = 0},
                                  {x = {from = from_module}})
    return name
end
