-- Runs a 3-block AdainResBlk1d stack (F0Ntrain's F0/N branches), threading style through each block.
--
-- **The blocks chain through the engine, not through Lua.** This used to read each block's output into
-- a Lua table and write it straight back as the next block's input -- and the two conversions around
-- that round trip are exact inverses (`from_layout_a(to_layout_a(rows)) == rows`), so the whole thing
-- was two full T x dim rebuilds per edge for a value nobody looked at. Each block retains instead and
-- the next names it; `loom.output_shape` supplies the row count, since a block may change it.
--
-- **It returns a module NAME, not rows.** Its only consumer is `run_proj1x1`, which is a graph -- so
-- converting the last block's output back into Lua here would be building a table for nobody. The one
-- conversion left is the caller's rows coming IN, whose producer interleaves two LSTM directions
-- host-side and so is genuinely a Lua value.
local function run_resblk_stack(name_prefix, x_rows, style)
    local T = #x_rows
    local dim = #x_rows[1]
    -- The name is written INTO each call rather than bound to a local first, which is not style: the
    -- `drives` check reads the suffix literals out of the topology-name argument (see
    -- `lua_library.drives_mismatches`), so a local there would take the only evidence it has.
    loom.run_subgraph_and_retain(name_prefix .. "_block0", {n_tokens = T, n_past = 0},
                                  {x = to_layout_a(x_rows, T, dim), style = style})
    local prev = name_prefix .. "_block0"
    for i = 1, 2 do
        -- `shape[1]`, not `shape[2]`: layout A is `flat[c * T + t]`, so the TIME axis is the fastest
        -- one (ne[0]) and the channel count is ne[1] -- the same reading the caller's rows get.
        -- The engine's own shape check caught this the first time round, which is that check earning
        -- its keep: marshalling through Lua compared element counts only, and [66,512] and [512,512]
        -- are not the same tensor even though a Lua table of 33792 numbers cannot tell you so.
        local shape = loom.output_shape(prev, 1)
        loom.run_subgraph_and_retain(name_prefix .. "_block" .. i, {n_tokens = shape[1], n_past = 0},
                                      {x = {from = prev}, style = style})
        prev = name_prefix .. "_block" .. i
    end
    return prev
end
