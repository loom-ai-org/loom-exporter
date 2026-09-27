
    -- --- The reference clip, normalised the way `infer_batch_process` normalises it. ---
    loom.seed_rng(opt_scalar(inputs.seed, 42))

    local rms = 0.0
    for i = 1, #waveform do rms = rms + waveform[i] * waveform[i] end
    rms = math.sqrt(rms / #waveform)

    -- Scale UP only. A clip already at or above the target is left alone, which is the reference's own
    -- `if rms < target_rms` and not a clamp: loud references are not attenuated.
    local rms_gain = (rms > 0.0 and rms < TARGET_RMS) and (TARGET_RMS / rms) or 1.0
    local ref_wave = {}
    for i = 1, #waveform do ref_wave[i] = waveform[i] * rms_gain end
