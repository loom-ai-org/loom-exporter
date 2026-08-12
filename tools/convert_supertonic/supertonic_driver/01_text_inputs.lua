    -- --- The text axis, padded to T_TEXT, and the mask that says how much of it is real ---
    -- Every text-touching topology was traced at a FIXED T_TEXT (see supertonic_export.py's module
    -- docstring for the two independent reasons), so the graph's `txt_ids` input is always exactly
    -- that wide. The HOST passes only the ids it has; padding them is this driver's job, which is
    -- what lets `infer` take any text up to T_TEXT rather than exactly T_TEXT of it.
    --
    -- `txt_msk` is a real graph input (P4.6). It was synthesized inside the trace as all-ones until
    -- then, which was correct only because T_TEXT ids were all real; the real modules genuinely READ
    -- it -- `x = x * txt_msk`, `attn_mask = txt_msk^T * txt_msk`, and `txt_len = txt_msk:sum()` for
    -- the fractional RoPE -- so padding it without saying so would attend to padding as text and
    -- recover a text length of T_TEXT.
    local n_txt = #inputs.txt_ids
    if n_txt > T_TEXT then
        error("supertonic: " .. n_txt .. " txt_ids exceeds this export's T_TEXT of " .. T_TEXT)
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
