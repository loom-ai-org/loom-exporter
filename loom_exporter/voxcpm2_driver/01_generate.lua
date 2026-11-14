    -- ===== VoxCPM2: text -> patches of 4 x 64-d latents, one per step -> 48 kHz. =====
    --
    -- `VoxCPM2Model._inference` in zero-shot mode, through the loop the live stream runs (00_stream.lua:
    -- `voxcpm_begin`, then `voxcpm_step` to the end), so the one-pass gate is the live loop's gate too.
    -- What one pass adds is `generate`'s `retry_badcase` and the decoder in one call.
    if inputs.seed then loom.seed_rng(inputs.seed) end
    if #tokens == 0 then error('voxcpm2: no text ids') end
    -- `retry_badcase`: a run that spends its whole ratio budget is retried with fresh draws, up to three
    -- times, and the last attempt stands. A caller's own `max_frames` or pinned noise is a measurement,
    -- not a generation, and is never retried.
    local _attempts = (inputs.max_frames or inputs.noise) and 1 or 3
    local _patch = PATCH_SIZE * LATENT_DIM
    local _latents, _n_patches
    for _attempt = 1, _attempts do
        local _st = voxcpm_begin(inputs)
        _latents, _n_patches = {}, 0
        while _st.ended == 0 do
            local _x = voxcpm_step(_st)
            for _i = 1, _patch do _latents[_n_patches * _patch + _i] = _x[_i] end
            _n_patches = _n_patches + 1
        end
        -- `retry_badcase_ratio_threshold`: as long as the text times the ratio is a badcase.
        if _n_patches < #tokens * BADCASE_RATIO then break end
    end

    -- The AudioVAE, every latent in one call: `[4 * n_patches, 64]`, frame-major. `return_patches`
    -- hands back the latents themselves instead: the loop's own state, which is what an oracle compares
    -- step by step (a waveform smears one patch's error over the decoder's receptive field).
    if inputs.return_patches then
        wave = _latents
    else
        wave = loom.run_subgraph('vae_decode', {n_codes = PATCH_SIZE * _n_patches, n_past = 0},
            {latents = _latents})
    end
