    -- The NSF source's two draws: a phase per harmonic (the fundamental's pinned to 0) uniform on
    -- [-pi, pi), and a unit Gaussian per harmonic per output sample. `SineGen.forward` draws them in
    -- that order. A caller may pass both, which is what makes the waveform comparable to a reference
    -- at all; otherwise the engine's stream supplies them.
    local nsf_phase = inputs.nsf_phase
    if nsf_phase == nil then
        local _u = loom.uniform_array(NSF_HARMONICS)
        nsf_phase = {0.0}
        for _h = 2, NSF_HARMONICS do nsf_phase[_h] = (2.0 * _u[_h] - 1.0) * math.pi end
    end
    local nsf_noise = inputs.nsf_noise or loom.gaussian_array(NSF_HARMONICS * SAMPLES_PER_FRAME * n_gen_frames)
