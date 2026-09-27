    -- ===== S3Gen's inputs: one token sequence, one in-fill grid. =====
    --
    -- The voice's prompt tokens come FIRST and the conformer sees one sequence, exactly as
    -- `CausalMaskedDiffWithXvec.inference` concatenates them. The mel grid is twice as long, and its
    -- first `n_prompt_frames` rows are the voice's own mel: `cond` carries them and the ODE in-fills
    -- the rest. Both conditioning arrays are FRAME-major, the layout every graph here agrees on.
    local _prompt_tokens = chatterbox_voice(inputs.prompt_token, 'voice.prompt_token')
    local _prompt_feat = chatterbox_voice(inputs.prompt_feat, 'voice.prompt_feat')
    local flow_embedding = chatterbox_voice(inputs.embedding, 'voice.embedding')
    local flow_tokens = {}
    for _i = 1, #_prompt_tokens do flow_tokens[_i] = _prompt_tokens[_i] end
    for _i = 1, #speech_tokens do flow_tokens[#flow_tokens + 1] = speech_tokens[_i] end
    local n_codes = #flow_tokens
    local n_frames = TOKEN_MEL_RATIO * n_codes
    local n_prompt_frames = #_prompt_feat / N_MEL
    local n_gen_frames = n_frames - n_prompt_frames
    local step_cond = chatterbox_zeros(n_frames * N_MEL)
    for _i = 1, #_prompt_feat do step_cond[_i] = _prompt_feat[_i] end
    local zero_cond = chatterbox_zeros(n_frames * N_MEL)
    local zero_spks = chatterbox_zeros(N_MEL)
    -- `n_steps`, the name every host passes a step count under (loom-py's `Text2Speech` reads the
    -- contract's `tts.default_steps` and forwards it as that).
    local times = chatterbox_cosine_times(inputs.n_steps or FLOW_STEPS)
    local flow_cfg = inputs.flow_cfg_rate or FLOW_CFG_RATE
