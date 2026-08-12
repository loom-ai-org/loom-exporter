    -- --- The voice style: the caller's, or this export's default ---
    -- `style_ttl`/`style_dp` are OPTIONAL inputs (BACKLOG.md P4.6b). They have always been inputs, and
    -- passing a different pair has always selected a different voice -- what changed is that omitting
    -- them now works, because the checkpoint's own default style travels in the GGUF as two tensors
    -- and `loom.get_weight` reads them back. Before that, every caller had to have obtained a style
    -- from the original checkpoint repo, which a published model file is not shipped with.
    --
    -- Read through "decoder" because `loom.get_weight`'s first argument is any REGISTERED module and
    -- every module shares one GgufModel -- and "decoder" is the one topology whose name does not carry
    -- a text bucket, so it is the only one that is spelled the same whatever the text length.
    --
    -- Read once per call, not per CFM step: `style_ttl` is 12800 floats and the sampler is handed it
    -- as a plain Lua array, so a `get_weight` inside the loop would marshal it n_steps times.
    local style_ttl = inputs.style_ttl or loom.get_weight("decoder", "loom.default_style.ttl")
    local style_dp = inputs.style_dp or loom.get_weight("decoder", "loom.default_style.dp")
