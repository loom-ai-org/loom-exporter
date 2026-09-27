    -- The generated frames: the integrated grid with the prompt's rows sliced off, frame-major, which
    -- is a contiguous suffix and the vocoder's own input layout.
    local _mel_full = loom.get_output('estimator', 1)
    local mel_tail = {}
    local _skip = n_prompt_frames * N_MEL
    for _i = 1, n_gen_frames * N_MEL do mel_tail[_i] = _mel_full[_skip + _i] end
