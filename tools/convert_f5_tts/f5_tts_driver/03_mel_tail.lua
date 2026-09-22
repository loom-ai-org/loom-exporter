
    -- --- The frames to vocode: the reference's own `where(cond_mask, cond, out)`, then its slice. ---
    -- The integrated state is still in the estimator's own store (`loom.run_ode_and_retain`), and this
    -- is its only reader. It has to CROSS to be sliced -- a retained reference is the whole tensor or
    -- nothing, and the offset is a value only the host knows. Once, not per step, which is the
    -- distinction ADR-031 draws.
    --
    -- **Both arrays here are FRAME-major** (`N_MEL` contiguous floats per frame), which is the
    -- estimator's own layout and, since the layout fix, the vocoder's declared input layout too. It
    -- was not: `Vocos.decode`'s convention is channel-major, so this slice used to be handed to a
    -- graph that read it transposed -- which is still a plausible spectrogram, so nothing raised and
    -- the model said "(chimes ringing)" (loom.cpp Retro-052). The conversion lives in the vocoder's
    -- own graph now; nothing here reindexes.
    --
    -- The two lengths differ by exactly one frame and the frame is real. `sample()` overwrites the
    -- first `cond_len` rows of its answer with the CONDITIONING mel and the caller then slices from
    -- `ref_frames` -- which is one smaller, because a centred STFT yields `#samples//hop + 1` frames.
    -- So the first row handed to the vocoder is conditioning, not generation, and taking it from the
    -- integrated state instead costs 6.3 in that row's mel.
    local mel_full = loom.get_output('estimator', 1)
    local mel_tail = {}
    local overlap = (cond_len - ref_frames) * N_MEL
    for i = 1, overlap do
        mel_tail[i] = step_cond[ref_frames * N_MEL + i]
    end
    for i = overlap + 1, n_gen * N_MEL do
        mel_tail[i] = mel_full[ref_frames * N_MEL + i]
    end
