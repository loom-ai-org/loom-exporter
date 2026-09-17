-- Runs a 3-block AdainResBlk1d stack (F0Ntrain's F0/N branches), threading style through each block,
-- and returns the module NAME holding the last block's output.
--
-- **Nothing crosses the boundary here at all.** The blocks used to chain through Lua -- each output
-- read into a table and written straight back as the next block's input, through two conversions that
-- are exact inverses -- and that went first (ADR-031). What remained was the caller's rows coming IN,
-- because a BiLSTM's two directions were interleaved host-side; they are not any more, so this takes a
-- module name at both ends. `loom.output_shape` supplies the row count, since a block may change it.
local function run_resblk_stack(name_prefix, from_module, style)
    -- `shape[1]`, not `shape[2]`: these blocks declare layout A (`flat[c * T + t]`), so the TIME axis
    -- is the fastest one (ne[0]) and the channel count is ne[1]. The engine's own shape check caught
    -- this the first time round, which is that check earning its keep: marshalling through Lua compared
    -- element counts only, and [66,512] and [512,512] are not the same tensor even though a Lua table
    -- of 33792 numbers cannot tell you so.
    local shape = loom.output_shape(from_module, 1)
    -- The name is written INTO each call rather than bound to a local first, which is not style: the
    -- `drives` check reads the suffix literals out of the topology-name argument (see
    -- `lua_library.drives_mismatches`), so a local there would take the only evidence it has.
    loom.run_subgraph_and_retain(name_prefix .. "_block0", {n_tokens = shape[1], n_past = 0},
                                  {x = {from = from_module}, style = style})
    local prev = name_prefix .. "_block0"
    for i = 1, 2 do
        local block_shape = loom.output_shape(prev, 1)
        loom.run_subgraph_and_retain(name_prefix .. "_block" .. i, {n_tokens = block_shape[1], n_past = 0},
                                      {x = {from = prev}, style = style})
        prev = name_prefix .. "_block" .. i
    end
    return prev
end
