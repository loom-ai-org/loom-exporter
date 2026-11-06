    -- The encoder output stays in the engine: the joint names a block of its rows per call.
    local n_frames = loom.output_shape('encoder', 1)[2]

    local tokens = {}
    -- `[-1] * (context_size - 1) + [blank]`, the reference's start. The decoder embeds a negative label
    -- as zeros, so the -1s are "no label yet" rather than a token.
    local context = {}
    for i = 1, CONTEXT_SIZE - 1 do context[i] = -1 end
    context[CONTEXT_SIZE] = BLANK_ID
    loom.run_subgraph_and_retain('decoder', {n_tokens = 0, n_past = 0}, {context = context})

    -- The joint runs on up to JOINT_BLOCK frames against the one decoder output (P4.53): its weight is
    -- read once per call instead of once per frame, and it returns one greedy id per row, so a call
    -- downloads JOINT_BLOCK ids rather than 11661 logits a frame. The decoder is a pure function of the
    -- context, which only an emission moves, so every frame up to the block's first emission saw the
    -- decoder output the frame-by-frame loop would have given it. The frames after an emission are
    -- discarded and the next block starts at the frame after it, so the emissions are the reference's
    -- (one symbol per frame at most).
    local t = 0
    while t < n_frames do
        local rows = math.min(JOINT_BLOCK, n_frames - t)
        loom.run_subgraph_and_retain('joint', {n_enc_frames = rows, n_tokens = 0, n_past = 0},
                                      {encoder_frame = {from = 'encoder', row = t, rows = rows},
                                       decoder_out = {from = 'decoder'}})
        local ks = loom.get_output('joint', 1)   -- one greedy id per row, reduced in the graph
        local emitted = nil
        for i = 1, rows do
            if ks[i] ~= BLANK_ID then emitted = i break end
        end
        if emitted == nil then
            t = t + rows
        else
            local k = ks[emitted]
            tokens[#tokens + 1] = k
            for i = 1, CONTEXT_SIZE - 1 do context[i] = context[i + 1] end
            context[CONTEXT_SIZE] = k
            loom.run_subgraph_and_retain('decoder', {n_tokens = 0, n_past = 0}, {context = context})
            t = t + emitted
        end
    end
