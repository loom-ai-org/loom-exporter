-- Runs one BiLSTM instance over a `seq_len x input_dim` sequence and RETAINS the interleaved
-- `[h_fwd | h_bwd]` result, returning the module name that holds it. ggml has no LSTM op, so the
-- recurrence is genuinely outside the graph -- but outside the graph is not the same as across the
-- boundary, and as of ADR-031's follow-up neither end of this crosses.
--
-- **The input half of every gate is one product, before the sweep** (ADR-137). `<ns>_input` computes
-- `W_ih x + b` for every timestep of both directions at once (two outputs, forward then backward), and
-- each direction's cell adds only `W_hh h` to its row. It was one matrix-vector product per timestep per
-- direction: 70 ms of a 755 ms Paradee synthesis on a Pi 4, 0.98 s of Kokoro's 18.2 s. The sweep reads a
-- different sequence for each direction (`loom.run_bi_recurrent_pair_and_retain`), whose width is the
-- gate block's, `4 * hidden_dim`.
--
-- `seq` is a sequence to marshal or a `{from = "module"}` reference; `seq_len`/`input_dim` are stated
-- rather than measured because a reference has no `#`. `layout` is the consumer's convention, "rows"
-- (default, one row per timestep) or "layout_a" (time on the fastest axis, what the conv-family
-- topologies declare) -- see the binding.
local function run_bi_lstm(namespace_, seq, seq_len, input_dim, hidden_dim, layout)
    loom.run_subgraph_and_retain(namespace_ .. "_input", {n_tokens = seq_len, n_past = 0}, {sequence = seq})
    loom.run_bi_recurrent_pair_and_retain(namespace_ .. "_fwd", namespace_ .. "_bwd",
                                           {from = namespace_ .. "_input", index = 1},
                                           {from = namespace_ .. "_input", index = 2},
                                           seq_len, 4 * hidden_dim, hidden_dim, layout)
    -- The forward cell's store is where the whole interleaved sequence lives, so the name a caller
    -- threads onward is that module's. Returned rather than composed by the caller: which of the two
    -- holds it is this function's business.
    return namespace_ .. "_fwd"
end
