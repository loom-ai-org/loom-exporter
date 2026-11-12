    -- ===== The LM: text -> FSQ speech tokens, under `ras_sampling`. =====
    --
    -- `inputs.tokens` is the text's Qwen2 BPE ids, no framing. The prefill is `Qwen2LM.inference`'s
    -- `[sos, prompt text + text, task_id, prompt speech tokens]`, spelled as three parallel id arrays:
    -- a row is a SPEECH row (sos, task_id, the prompt's tokens -- `speech_embedding`) where
    -- `speech_mask` is 1 and a TEXT row (`embed_tokens`) where it is 0. The unused id is 0.
    --
    -- `inputs.speech_tokens` skips the LM: the flow and vocoder then run on the CALLER's tokens, which
    -- is how the gate checks those two phases on the reference's own tokens (VoxCPM2's
    -- `teacher_patches`). `inputs.return_tokens` returns the LM's raw draws, silence and all, instead of
    -- a waveform -- the exact arm of the sampled decode.
    local speech_tokens
    if inputs.speech_tokens then
        speech_tokens = inputs.speech_tokens
    else
        -- The loop the live stream runs too (00_stream.lua), to its end.
        if inputs.seed then loom.seed_rng(inputs.seed) end
        local _st = cosyvoice3_lm_begin(inputs, inputs.tokens)
        while _st.lm_ended == 0 do cosyvoice3_lm_step(_st) end
        speech_tokens = _st.lm_kept
        if #speech_tokens == 0 then
            error('cosyvoice3: the LM produced no speech tokens (the first draw was a stop id)')
        end
        if inputs.return_tokens then return _st.lm_history end
    end
