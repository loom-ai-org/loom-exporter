    -- ===== Zipformer streaming: the clip's fbank once, then one encoder call per chunk, decoding as it goes =====
    --
    -- upstream's `_infer_jit_streaming` + `DecodeStream` for one stream, greedy, at most one symbol per frame.
    -- Four topologies (zipformer_streaming_export.py): `fbank`, `encoder_chunk`, `decoder`, `joint`. The
    -- encoder's state is #STATE_SIZES tensors that each call returns and the next one reads as retained
    -- outputs, so it never leaves the engine.
    --
    -- Split into init / feed / finish so a live door can call the same three with chunks as they arrive;
    -- the one-pass door below is init, a feed per chunk of the whole clip, finish.
    --
    -- Inputs: `waveform` (16 kHz mono); `return_encoder` for the encoder rows instead of the token ids.

    local function _init()
        -- `DecodeStream`'s greedy start is `[blank] * context_size`, not greedy_search's `[-1, ..., blank]`.
        local s = {tokens = {}, context = {}, first = true}
        for i = 1, CONTEXT_SIZE do s.context[i] = BLANK_ID end
        loom.run_subgraph_and_retain('decoder', {n_tokens = 0, n_past = 0}, {context = s.context})
        return s
    end

    -- One encoder call on TAIL feature rows of `fbank` from `row`, then the joint over its frames.
    local function _feed(s, row)
        local inputs = {features = {from = 'fbank', row = row, rows = TAIL}}
        for k = 1, #STATE_SIZES do
            if s.first then
                -- The reference's `get_init_states`: every cache empty and nothing processed.
                local zeros = {}
                for i = 1, STATE_SIZES[k] do zeros[i] = 0.0 end
                inputs['state_' .. k] = zeros
            else
                inputs['state_' .. k] = {from = 'encoder_chunk', index = k + 1}
            end
        end
        loom.run_subgraph_and_retain('encoder_chunk', {n_tokens = 0, n_past = 0}, inputs)
        s.first = false
        -- `return_encoder` hands back every chunk's encoder rows instead of decoding them: the tensor the
        -- gate grades against upstream's streamed encoder, state carry included.
        if s.encoder then
            for _, v in ipairs(loom.get_output('encoder_chunk', 1)) do s.encoder[#s.encoder + 1] = v end
            return
        end

        -- The joint on blocks of up to JOINT_BLOCK frames against one decoder output, as the non-streaming
        -- driver does (P4.53): the frames after a block's first emission are discarded and the next block
        -- starts after it, so the emissions are the frame-by-frame loop's.
        local n_frames = loom.output_shape('encoder_chunk', 1)[2]
        local t = 0
        while t < n_frames do
            local rows = math.min(JOINT_BLOCK, n_frames - t)
            loom.run_subgraph_and_retain('joint', {n_enc_frames = rows, n_tokens = 0, n_past = 0},
                                          {encoder_frame = {from = 'encoder_chunk', row = t, rows = rows},
                                           decoder_out = {from = 'decoder'}})
            local ks = loom.get_output('joint', 1)
            local emitted = nil
            for i = 1, rows do
                if ks[i] ~= BLANK_ID then emitted = i break end
            end
            if emitted == nil then
                t = t + rows
            else
                local k = ks[emitted]
                s.tokens[#s.tokens + 1] = k
                for i = 1, CONTEXT_SIZE - 1 do s.context[i] = s.context[i + 1] end
                s.context[CONTEXT_SIZE] = k
                loom.run_subgraph_and_retain('decoder', {n_tokens = 0, n_past = 0}, {context = s.context})
                t = t + emitted
            end
        end
    end

    local function _finish(s)
        return s.tokens
    end

    -- The whole clip's features, then FEATURE_TAIL rows of log(1e-10): PAD_LENGTH of them are upstream's
    -- `set_features` padding, and the rest let the last chunk read TAIL rows as upstream pads it to.
    loom.run_subgraph_and_retain('fbank', {n_samples = #waveform, n_past = 0}, {waveform = waveform})
    local _n_features = loom.output_shape('fbank', 1)[2] - FEATURE_TAIL + PAD_LENGTH
    local _stream = _init()
    if inputs.return_encoder then _stream.encoder = {} end
    -- `get_feature_frames` advances by CHUNK_FEATURES and stops once it has passed the padded clip.
    local _row = 0
    repeat
        _feed(_stream, _row)
        _row = _row + CHUNK_FEATURES
    until _row >= _n_features
    tokens = _stream.encoder or _finish(_stream)
