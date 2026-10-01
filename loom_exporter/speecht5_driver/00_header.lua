-- SpeechT5's helpers. Top level, so the fragment below can call them.

-- The encoder's relative position index, `[n, n]` row-major: `clip(i - j, -max, max - 1) + max`, which
-- is `SpeechT5RelativePositionalEncoding`'s `pos_seq` before it indexes `pe_k`. The graph gathers
-- `pe_k` rows by it; handing over the index rather than the gathered `[n, n, 64]` rows keeps the
-- array n^2 integers.
local function speecht5_relative_index(n, max_rel)
    local index = {}
    for i = 0, n - 1 do
        for j = 0, n - 1 do
            local d = i - j
            if d < -max_rel then d = -max_rel elseif d > max_rel - 1 then d = max_rel - 1 end
            index[i * n + j + 1] = d + max_rel
        end
    end
    return index
end

-- One prenet dropout mask: `units` values of {0, 1}, each 1 with probability 1/2 -- the
-- `torch.bernoulli(p=0.5)` draw of `_consistent_dropout`. A caller's pinned masks (`masks`, one
-- `[2, units]` pair per step, row-major) replace the draw; `row` is 0-based over steps * 2.
local function speecht5_draw_mask(units, pinned, row)
    local mask = {}
    if pinned ~= nil then
        local base = row * units
        if pinned[base + units] == nil then
            error(string.format('speecht5: the pinned masks end before step %d', math.floor(row / 2)))
        end
        for i = 1, units do mask[i] = pinned[base + i] end
    else
        local u = loom.uniform_array(units)
        for i = 1, units do mask[i] = (u[i] < 0.5) and 1 or 0 end
    end
    return mask
end
