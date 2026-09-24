    -- The sine source's one live draw: a UNIFORM [0, 1) value per harmonic per output sample,
    -- harmonic-major -- `SineGen2.sine_waves`, which the reference draws once with `torch.rand` at
    -- construction and slices. A caller may pass it, which is what makes the waveform comparable to a
    -- reference at all; otherwise the engine's stream supplies it.
    local nsf_noise = inputs.nsf_noise or loom.uniform_array(NSF_HARMONICS * SAMPLES_PER_FRAME * n_gen_frames)
