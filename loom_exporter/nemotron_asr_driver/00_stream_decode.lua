-- ===== The live transducer decode (loom.cpp ADR-109, ADR-110): greedy RNN-T over a call's rows =====
--
-- The template's greedy loop over `n_frames` rows of `phase`'s retained output from row `row0`, its
-- emissions appended to `out`. Each emission is final, so a live stream returns them as they come. What
-- the loop carries from one call to the next is state: the last label, whether the prediction network
-- must run again (`stale`), whether it has run at all (`primed`), and each LSTM layer's h and c
-- (`pred1`.., kept on the device). Shared by Nemotron 3.5's and the unified model's streams.

local function transducer_stream_decode(phase, row0, n_frames, out)
    local last_label, stale, primed = loom.state('last_label'), loom.state('stale'), loom.state('primed')
    local zeros = nil
    for t = 0, n_frames - 1 do
        local frame = {from = phase, row = row0 + t, rows = 1}
        local symbols = 0
        while symbols < S_MAX_SYMBOLS_PER_STEP do
            if stale == 1 then
                loom.run_subgraph_and_retain('embed', {n_tokens = 0, n_past = 0}, {last_label = {last_label}})
                local layer_input = {from = 'embed'}
                for l = 1, S_N_PRED_LAYERS do
                    local cell = 'pred_lstm_l' .. (l - 1) .. '_fwd'
                    local h_prev, c_prev
                    if primed == 1 then
                        h_prev, c_prev = {state = 'pred' .. l, index = 1}, {state = 'pred' .. l, index = 2}
                    else
                        if zeros == nil then
                            zeros = {}
                            for i = 1, S_PRED_HIDDEN do zeros[i] = 0.0 end
                        end
                        h_prev, c_prev = zeros, zeros
                    end
                    loom.run_subgraph_and_retain(cell, {n_tokens = 0, n_past = 0},
                                                  {layer_input = layer_input, h_prev = h_prev, c_prev = c_prev})
                    loom.keep_state('pred' .. l, cell)
                    layer_input = {from = cell, index = 1}
                end
                primed, stale = 1, 0
            end
            loom.run_subgraph_and_retain('joint', {n_tokens = 0, n_past = 0},
                                          {encoder_frame = frame,
                                           decoder_out = {state = 'pred' .. S_N_PRED_LAYERS, index = 1}})
            local k = loom.argmax_row('joint', 0)
            symbols = symbols + 1
            -- A blank advances a frame; a symbol stays on it, at most S_MAX_SYMBOLS_PER_STEP times.
            if k == S_BLANK_ID then break end
            out[#out + 1] = k
            last_label, stale = k, 1
        end
    end
    loom.set_state('last_label', last_label)
    loom.set_state('stale', stale)
    loom.set_state('primed', primed)
end

-- The decode's state, fresh: no label yet, the prediction network not run.
local function transducer_stream_open()
    loom.set_state('last_label', S_BLANK_ID)
    loom.set_state('stale', 1)
    loom.set_state('primed', 0)
end
