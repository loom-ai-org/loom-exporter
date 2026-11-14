-- ===== The live transducer decode (loom.cpp ADR-109, ADR-110, ADR-122): greedy RNN-T or TDT over a call's rows =====
--
-- The template's greedy loop over `n_frames` rows of `phase`'s retained output from row `row0`, its
-- emissions appended to `out`. Each emission is final, so a live stream returns them as they come. What
-- the loop carries from one call to the next is state: the last label, whether the prediction network
-- must run again (`stale`), whether it has run at all (`primed`), each LSTM layer's h and c (`pred1`..,
-- kept on the device), and -- TDT only -- `time_jump`, how far past the call's last row the last duration
-- reached. Shared by Nemotron 3.5's, the unified model's and Parakeet's streams.
--
-- The loop is NeMo's label-looping greedy decode with `prev_batched_state` (what its streaming script
-- decodes each chunk with), written frame by frame as `02_decode.lua` is. The two agree: a blank moves on
-- by its duration (at least one frame); a symbol is emitted and moves on by its own, which for RNN-T and
-- for a zero duration is none; `S_MAX_SYMBOLS_PER_STEP` symbols on one frame force a move of one. A
-- duration that lands past the last row is NeMo's `time_jumps`: the next call starts that far into its
-- rows (`time_indices - encoder_output_length`, carried). RNN-T never jumps, so its `time_jump` stays 0.
-- NeMo counts the symbols on a frame per CALL (`last_nb_timestamp` lives in that call's hypotheses), so
-- the count starts afresh with each call, which it does here because a call's frames are all new.

local function transducer_stream_decode(phase, row0, n_frames, out)
    local last_label, stale, primed = loom.state('last_label'), loom.state('stale'), loom.state('primed')
    local zeros = nil
    local t = loom.state('time_jump')
    while t < n_frames do
        local frame = {from = phase, row = row0 + t, rows = 1}
        local symbols = 0
        local skip = 0
        while true do
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
            -- RNN-T: a blank moves one frame, a symbol none. TDT: the duration head says, a blank's at least 1.
            skip = (k == S_BLANK_ID) and 1 or 0
            if S_N_DURATIONS > 0 then
                local dur = loom.get_output('joint', 2)
                local best = 1
                for i = 2, S_N_DURATIONS do
                    if dur[i] > dur[best] then best = i end
                end
                skip = S_DURATIONS[best]
                if k == S_BLANK_ID and skip == 0 then skip = 1 end
            end
            if k ~= S_BLANK_ID then
                out[#out + 1] = k
                last_label, stale = k, 1
                symbols = symbols + 1
            end
            if skip > 0 then break end
            -- A symbol that stays on its frame: at most S_MAX_SYMBOLS_PER_STEP of them, then one frame on.
            if symbols >= S_MAX_SYMBOLS_PER_STEP then
                skip = 1
                break
            end
        end
        t = t + skip
    end
    loom.set_state('time_jump', t - n_frames)
    loom.set_state('last_label', last_label)
    loom.set_state('stale', stale)
    loom.set_state('primed', primed)
end

-- The decode's state, fresh: no label yet, the prediction network not run, no jump carried.
local function transducer_stream_open()
    loom.set_state('last_label', S_BLANK_ID)
    loom.set_state('stale', 1)
    loom.set_state('primed', 0)
    loom.set_state('time_jump', 0)
end
