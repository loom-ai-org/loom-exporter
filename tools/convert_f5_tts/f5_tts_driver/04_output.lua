
    -- --- Undo the reference's own gain, so the answer is at the caller's loudness. ---
    local waveform_out = {}
    for i = 1, #wave_raw do waveform_out[i] = wave_raw[i] / rms_gain end
