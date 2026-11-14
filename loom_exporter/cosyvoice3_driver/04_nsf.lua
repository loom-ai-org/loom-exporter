    -- The sine source's one live draw: a UNIFORM [0, 1) value per harmonic per output sample,
    -- harmonic-major -- `SineGen2.sine_waves`, which the reference draws once with `torch.rand` at
    -- construction and slices. A caller may pass it, which is what makes the waveform comparable to a
    -- reference at all; otherwise it is the per-frame draw made with the flow's noise (02_flow_plan.lua),
    -- laid out harmonic-major.
    local nsf_noise = inputs.nsf_noise or cosyvoice3_nsf_from_blocks(nsf_blocks, n_gen_frames)
