-- Runs one BiLSTM instance over a T x input_dim sequence, driving the per-timestep cell topologies
-- (`<ns>_fwd`/`_bwd`) the exporter's `recurrent.py` generates. ggml has no LSTM op, so the recurrence
-- is genuinely outside the graph -- but outside the graph is not the same as across the boundary.
--
-- **Two calls, not 4T.** This used to step the cells from Lua, which put `layer_input`, `h_prev` and
-- `c_prev` across the boundary and pulled `h_new`/`c_new` back, per timestep, per direction: the h/c
-- carry alone was four hidden-wide tables per step for a value no host ever looks at.
-- `loom.run_recurrent` walks the whole sweep in C++ with the carry in `std::vector<float>`, so what
-- crosses is the input sequence and each direction's output -- the two things Lua genuinely needs,
-- because the concatenation below is host math (`output_store.h`'s rule).
--
-- `reverse = true` is the backward direction and it is NOT a pre-reversed sequence: the binding walks
-- the same forward-ordered array from the far end and writes each result to its own real time index,
-- which is what the Lua loop did and what `BiLstmStepper` does.
local function run_bi_lstm(namespace_, seq, hidden_dim)
    local T = #seq
    local input_dim = #seq[1]
    local flat = {}
    for t = 1, T do
        local row = seq[t]
        local base = (t - 1) * input_dim
        for k = 1, input_dim do flat[base + k] = row[k] end
    end

    local fwd = loom.run_recurrent(namespace_ .. "_fwd", flat, T, input_dim, hidden_dim, false)
    local bwd = loom.run_recurrent(namespace_ .. "_bwd", flat, T, input_dim, hidden_dim, true)

    -- `[h_fwd | h_bwd]` per timestep, which is what every consumer of a BiLSTM here expects and the
    -- one part of this that has to be a Lua table.
    local out = {}
    for t = 1, T do
        local row = {}
        local base = (t - 1) * hidden_dim
        for i = 1, hidden_dim do
            row[i] = fwd[base + i]
            row[hidden_dim + i] = bwd[base + i]
        end
        out[t] = row
    end
    return out
end
