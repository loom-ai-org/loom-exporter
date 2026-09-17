    -- The encoder output STAYS IN THE ENGINE. The joint consumes one frame per call and
    -- `{from = 'encoder', row = t, rows = 1}` copies exactly that, backend-side -- so this driver never
    -- sees the [n_embd, n_frames] tensor at all, where it used to marshal every frame of it to slice
    -- one row at a time. `loom.output_shape` is how it learns the frame count without the data.
    local _enc_shape = loom.output_shape('encoder', 1)
    local n_embd, n_frames = _enc_shape[1], _enc_shape[2]

    local tokens = {}
    local last_label = BLANK_ID
    -- The prediction network's state lives in each cell's own retained outputs from the second pass on;
    -- these zeros are the first pass's `h_prev`/`c_prev` and the only ones that ever cross. `primed`
    -- flips once, after the first full stack pass, because every layer runs on every pass.
    local _zeros = {}
    for i = 1, PRED_HIDDEN do _zeros[i] = 0.0 end
    local primed = false

    -- The prediction network runs once per EMITTED TOKEN, not once per frame: its output is a pure
    -- function of (last_label, h, c), and all three change only on emission. `top_h == nil` means "must
    -- recompute", so a blank costs one joint call and nothing else -- and most frames of real audio are
    -- blank. Same restructuring the C++ decoder got before it was retired; the discarded recompute it
    -- replaces could not have differed, which is why this is equivalence and not an approximation.
    local top_h = nil
    local t = 0
    while t < n_frames do
        -- Row `t` of the encoder's retained output, named rather than copied through Lua.
        local frame = {from = 'encoder', row = t, rows = 1}

        local symbols = 0
        local advanced = false
        while symbols < MAX_SYMBOLS_PER_STEP do
            if top_h == nil then
                loom.run_subgraph_and_retain('embed', {n_tokens = 0, n_past = 0},
                                              {last_label = {last_label}})
                local layer_input = {from = 'embed'}
                for l = 1, N_PRED_LAYERS do
                    local cell = 'pred_lstm_l' .. (l - 1) .. '_fwd'
                    -- Each cell's own previous h/c, read out of its own store: the read happens while
                    -- the inputs are filled, before the graph that overwrites the store runs, so a
                    -- cell feeding itself is ordered and not aliased.
                    loom.run_subgraph_and_retain(cell, {n_tokens = 0, n_past = 0},
                                                  {layer_input = layer_input,
                                                   h_prev = primed and {from = cell, index = 1} or _zeros,
                                                   c_prev = primed and {from = cell, index = 2} or _zeros})
                    layer_input = {from = cell, index = 1}
                end
                primed = true
                -- The top layer's h, still by reference: the joint is its only reader, and a blank step
                -- reuses this without the cell running again -- which is exactly when a retained output
                -- is still the right one to name.
                top_h = layer_input
            end

            -- Retained, so the 8198-wide joint output never becomes a Lua table: the token head is
            -- reduced engine-side and only the handful of duration logits are marshalled.
            loom.run_subgraph_and_retain('joint', {n_tokens = 0, n_past = 0},
                                          {encoder_frame = frame, decoder_out = top_h})
            local k = loom.argmax_row('joint', 0)

            -- Plain RNN-T has no duration head at all, and every blank advances exactly one frame.
            local skip = 1
            if N_DURATIONS > 0 then
                local dur = loom.get_output('joint', 2)
                local best = 1
                for i = 2, N_DURATIONS do
                    if dur[i] > dur[best] then best = i end
                end
                skip = DURATIONS[best]
            end

            if k ~= BLANK_ID then
                tokens[#tokens + 1] = k
                last_label = k
                top_h = nil          -- last_label moved, so the cached prediction no longer applies
                if N_DURATIONS == 0 then skip = 0 end
            elseif skip == 0 then
                skip = 1             -- a blank must advance, or decoding spins on one frame forever
            end

            symbols = symbols + 1
            t = t + skip
            if skip > 0 then
                advanced = true
                break
            end
        end
        if not advanced then
            -- Defensive bound, not part of the TDT algorithm itself: guards a model that keeps emitting
            -- duration-0 non-blanks forever. Same fallback the C++ decoder carried.
            t = t + 1
        end
    end
