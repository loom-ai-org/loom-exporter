    -- The encoder output stays in the engine: the joint names one row of it per call.
    local n_frames = loom.output_shape('encoder', 1)[2]

    local tokens = {}
    -- `[-1] * (context_size - 1) + [blank]`, the reference's start. The decoder embeds a negative label
    -- as zeros, so the -1s are "no label yet" rather than a token.
    local context = {}
    for i = 1, CONTEXT_SIZE - 1 do context[i] = -1 end
    context[CONTEXT_SIZE] = BLANK_ID
    loom.run_subgraph_and_retain('decoder', {n_tokens = 0, n_past = 0}, {context = context})

    for t = 0, n_frames - 1 do
        loom.run_subgraph_and_retain('joint', {n_tokens = 0, n_past = 0},
                                      {encoder_frame = {from = 'encoder', row = t, rows = 1},
                                       decoder_out = {from = 'decoder'}})
        local k = loom.argmax_row('joint', 0)
        -- The decoder is a pure function of the context, which only an emission moves, so a blank frame
        -- costs one joint call and the retained decoder output is still the right one.
        if k ~= BLANK_ID then
            tokens[#tokens + 1] = k
            for i = 1, CONTEXT_SIZE - 1 do context[i] = context[i + 1] end
            context[CONTEXT_SIZE] = k
            loom.run_subgraph_and_retain('decoder', {n_tokens = 0, n_past = 0}, {context = context})
        end
    end
