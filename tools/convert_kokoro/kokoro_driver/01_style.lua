    -- Speaking rate, defaulted here rather than declared by the export: 1.0 is the IDENTITY, not a
    -- property of this checkpoint. A number the model has an opinion about (a sampler's step count, a
    -- voice) is declared in the file; a number whose neutral value is the same for every model belongs
    -- in the driver, where omitting it means "unchanged" instead of failing on a nil arithmetic.
    inputs.speed = inputs.speed or 1.0

    loom.seed_rng(inputs.seed)

    local T_text = #inputs.input_ids

    -- --- The voice: the caller's, or this export's default ---
    -- `ref_s` has always been an input and a different vector has always selected a different voice.
    -- What changed is that omitting it now works: the checkpoint's own style travels in the GGUF as one
    -- tensor and `loom.get_weight` reads it back. Before that a published model could not speak at all
    -- without the original checkpoint repo, and a caller who passed nothing got a nil index here rather
    -- than an explanation. Same fix as Supertonic's (BACKLOG.md P4.6b).
    --
    -- Read through "text_encoder_cnn" because `loom.get_weight`'s first argument is any REGISTERED
    -- module and every module shares one GgufModel -- so which one is named does not matter beyond it
    -- existing in every Kokoro export, which this one does.
    --
    -- Read ONCE, into a local: the slices below index it repeatedly and a get_weight per access would
    -- marshal 256 floats out of the model each time.
    local ref_s = inputs.ref_s or loom.get_weight("text_encoder_cnn", "loom.default_style.ref_s")

    local s_decoder = array_slice(ref_s, 1, STYLE_DIM)
    local s_predictor = array_slice(ref_s, STYLE_DIM + 1, STYLE_DIM)
