-- Runs one BiLSTM instance over a `seq_len x input_dim` sequence and RETAINS the interleaved
-- `[h_fwd | h_bwd]` result, returning the module name that holds it. ggml has no LSTM op, so the
-- recurrence is genuinely outside the graph -- but outside the graph is not the same as across the
-- boundary, and as of ADR-031's follow-up neither end of this crosses.
--
-- **One call, not 4T, and no interleave in Lua.** This stepped the cells from Lua once: `layer_input`,
-- `h_prev` and `c_prev` out and `h_new`/`c_new` back, per timestep, per direction. `loom.run_recurrent`
-- moved the sweep into C++ with the carry in `std::vector<float>`, which left exactly one crossing --
-- the two directions' outputs, pulled back so this function could build `[h_fwd | h_bwd]` rows for its
-- consumers. That was the last one: `loom.run_bi_recurrent_and_retain` runs both directions and writes
-- both halves of every row into one store slot, so a BiLSTM's output reaches the next graph as a name.
--
-- `seq` is a sequence to marshal or a `{from = "module"}` reference; `seq_len`/`input_dim` are stated
-- rather than measured because a reference has no `#`. `layout` is the consumer's convention, "rows"
-- (default, one row per timestep) or "layout_a" (time on the fastest axis, what the conv-family
-- topologies declare) -- see the binding.
local function run_bi_lstm(namespace_, seq, seq_len, input_dim, hidden_dim, layout)
    loom.run_bi_recurrent_and_retain(namespace_ .. "_fwd", namespace_ .. "_bwd", seq, seq_len,
                                      input_dim, hidden_dim, layout)
    -- The forward cell's store is where the whole interleaved sequence lives, so the name a caller
    -- threads onward is that module's. Returned rather than composed by the caller: which of the two
    -- holds it is this function's business.
    return namespace_ .. "_fwd"
end
