
    -- --- The frame grid everything downstream is sized by. ---
    -- `cond_shape` is the mel topology's own ne = [n_mel, cond_len, 1, 1]; the FRAME count is ne[2],
    -- and it is `#samples//hop + 1`, one more than the `ref_audio_len` the output slice uses. Those
    -- two being different numbers is the reference's own arithmetic (`sample()` takes its conditioning
    -- length from `cond.shape[1]` and the caller slices from `n_samples//hop`), and reading either as
    -- the other costs one frame of conditioning -- which 32 Euler steps amplify into an audible
    -- difference rather than a rounding one.
    local cond_len = cond_shape[2]
    local ref_frames = math.floor(#waveform / HOP_LENGTH)

    -- The duration estimate, `infer_batch_process`'s: the reference's frames-per-character rate,
    -- applied to the text to speak. It is a RATE and nothing in the model enforces it -- which is why
    -- `duration` overrides it outright.
    local n_frames
    local duration = opt_scalar(inputs.duration, nil)
    if duration then
        n_frames = math.floor(duration)
    else
        local n_ref_text = opt_scalar(inputs.n_ref_text, nil)
        if not n_ref_text then
            error("f5-tts: pass either `duration` (total frames) or `n_ref_text` (how many of " ..
                  "text_ids are the reference transcript) -- the frame count is estimated from the " ..
                  "reference's own characters-per-frame rate and there is nothing in the ids that " ..
                  "marks where the transcript ends")
        end
        local n_gen_text = #text_ids - n_ref_text
        local speed = opt_scalar(inputs.speed, DEFAULT_SPEED)
        n_frames = ref_frames + math.floor(ref_frames / n_ref_text * n_gen_text / speed)
    end
    -- At least one frame past the longer of the two things that have to fit, the reference's own
    -- `maximum(maximum(text_len, lens) + 1, duration)`.
    local floor_frames = math.max(#text_ids, cond_len) + 1
    if n_frames < floor_frames then n_frames = floor_frames end

    local n_gen = n_frames - ref_frames
    local ids, uncond_ids, keep = f5_text_arrays(text_ids, n_frames)
    local step_cond, zero_cond = f5_step_cond(cond_mel, cond_len, n_frames, N_MEL)
    local times = sway_times(opt_scalar(inputs.n_steps, DEFAULT_STEPS),
                             opt_scalar(inputs.sway_coef, DEFAULT_SWAY))
    local cfg_scale = opt_scalar(inputs.cfg_scale, DEFAULT_CFG)
