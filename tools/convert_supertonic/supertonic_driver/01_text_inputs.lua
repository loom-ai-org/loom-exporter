    -- --- The text bucket, the padded ids, and the mask that says how much of the axis is real ---
    -- Every text-touching topology was traced at a FIXED text width (see supertonic_export.py's
    -- module docstring for the two independent reasons it cannot be dynamic), so the graph's
    -- `txt_ids` input is always exactly that wide. It is traced at SEVERAL widths, though, and this
    -- picks the smallest that fits (BACKLOG.md P4.6a) -- which is what stops a short sentence paying
    -- for the longest one this export can take. The HOST passes only the ids it has.
    --
    -- `txt_msk` is a real graph input (P4.6). It was synthesized inside the trace as all-ones until
    -- then, which was correct only because every id was real; the real modules genuinely READ it --
    -- `x = x * txt_msk`, `attn_mask = txt_msk^T * txt_msk`, and `txt_len = txt_msk:sum()` for the
    -- fractional RoPE -- so padding without saying so would attend to padding as text and recover a
    -- text length of the whole axis.
    local n_txt = #inputs.txt_ids
    local T_TEXT = nil
    for i = 1, #TEXT_BUCKETS do
        if TEXT_BUCKETS[i] >= n_txt then
            T_TEXT = TEXT_BUCKETS[i]
            break
        end
    end
    if T_TEXT == nil then
        -- TEXT_BUCKETS is ascending, so falling off the end means the text exceeds the largest one.
        -- Refusing is the point: truncating to the ceiling would synthesize perfectly plausible audio
        -- of the wrong words, which is far worse than not synthesizing at all.
        error("supertonic: " .. n_txt .. " txt_ids exceeds this export's T_TEXT of "
              .. TEXT_BUCKETS[#TEXT_BUCKETS])
    end

    local txt_ids, txt_msk = {}, {}
    for i = 1, T_TEXT do
        -- The pad ID is arbitrary and measured to be so: `x = x * txt_msk` zeroes the embedding of
        -- every padded position before anything reads it, so ids 0, 1 and 162 give bit-identical
        -- output (BACKLOG.md P4.6). PAD_ID is the vocabulary's one unused row anyway, so a dump of
        -- the padded ids is unambiguous to a reader.
        txt_ids[i] = inputs.txt_ids[i] or PAD_ID
        txt_msk[i] = i <= n_txt and 1.0 or 0.0
    end
