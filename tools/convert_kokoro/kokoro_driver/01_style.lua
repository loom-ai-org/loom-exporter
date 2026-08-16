    -- Speaking rate, defaulted here rather than declared by the export: 1.0 is the IDENTITY, not a
    -- property of this checkpoint. A number the model has an opinion about (a sampler's step count, a
    -- voice) is declared in the file; a number whose neutral value is the same for every model belongs
    -- in the driver, where omitting it means "unchanged" instead of failing on a nil arithmetic.
    inputs.speed = inputs.speed or 1.0

    loom.seed_rng(inputs.seed)

    local T_text = #inputs.input_ids

    -- --- The voice: the caller's, or this export's default ---
    -- `ref_s` has always been an input and a different vector has always selected a different voice.
    -- What changed is that omitting it now works: an upstream voice PACK travels in the GGUF as one
    -- tensor and `loom.get_weight` reads it back. Before that a published model could not speak at all
    -- without the original checkpoint repo, and a caller who passed nothing got a nil index here rather
    -- than an explanation. Same fix as Supertonic's (BACKLOG.md P4.6b).
    --
    -- Read through "text_encoder_cnn" because `loom.get_weight`'s first argument is any REGISTERED
    -- module and every module shares one GgufModel -- so which one is named does not matter beyond it
    -- existing in every Kokoro export, which this one does.
    --
    -- A PACK, not a vector, and the row is chosen HERE because the choice is length-dependent:
    -- upstream's `KPipeline.__call__` is `pack[len(ps)-1]`, one style per phoneme count. `input_ids`
    -- arrives wrapped (leading and trailing 0, see the header), so `len(ps)` is `T_text - 2` and the
    -- 0-based row is `T_text - 3`; +1 again for Lua's 1-based slice. Clamped rather than trusted, so a
    -- caller who passes unwrapped ids or a string past the pack's end gets the nearest real voice
    -- instead of a nil index deep inside `array_slice`.
    --
    -- Read the ROW ONCE, into a local: the two slices below index it repeatedly and a get_weight per
    -- access would marshal the whole pack out of the model each time.
    local ref_s = inputs.ref_s
    if not ref_s then
        local pack = loom.get_weight("text_encoder_cnn", "loom.default_style.ref_s")
        local rows = #pack / (2 * STYLE_DIM)
        local row = T_text - 3
        if row < 0 then row = 0 elseif row > rows - 1 then row = rows - 1 end
        ref_s = array_slice(pack, row * 2 * STYLE_DIM + 1, 2 * STYLE_DIM)
    end

    local s_decoder = array_slice(ref_s, 1, STYLE_DIM)
    local s_predictor = array_slice(ref_s, STYLE_DIM + 1, STYLE_DIM)
