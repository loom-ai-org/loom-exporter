    -- ===== Chatterbox: text -> T3 speech tokens (25 Hz) -> S3Gen's mel -> HiFT's 24 kHz. =====
    --
    -- Inputs: `tokens` (the text's BPE ids, no framing), and optionally the voice (`speaker_emb`,
    -- `cond_prompt_tokens`, `prompt_token`, `prompt_feat`, `embedding`; the built-in one otherwise), T3's
    -- knobs (`exaggeration`, `cfg_weight`, `temperature` -- 0 is guided greedy -- `min_p`, `top_p`,
    -- `repetition_penalty`, `max_new_tokens`, `seed`), the flow's (`n_steps`, `flow_cfg_rate`), the draws
    -- (`noise`, `nsf_phase`, `nsf_noise`) and `return_mel`. Returns the waveform.
    --
    -- One pass is the live stream (00_stream.lua) opened, fed to its end and finished inside this call,
    -- against the fresh state the engine gives every one-pass call -- so the one-pass gate is the live
    -- path's gate too. The whole turn in one feed, one flow call and one vocoder call, unless the caller asks
    -- for hops (`piece_tokens`), which is the live cut run in one call.
    if inputs.piece_tokens == nil then inputs.piece_tokens = 0 end
    stream_open(inputs)
    wave = {}
    while loom.state('done') == 0 do
        local _piece = stream_feed({})
        local _base = #wave
        for _i = 1, #_piece do wave[_base + _i] = _piece[_i] end
    end
    stream_finish({})
