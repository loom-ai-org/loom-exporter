    -- ===== VoxCPM2: text -> patches of 4 x 64-d latents, one per step -> 48 kHz. =====
    --
    -- `VoxCPM2Model._inference` in zero-shot mode. `inputs.tokens` is the text's ids as
    -- `loom::VoxCpmVocab::encode` returns them (no BOS); `<|audio_start|>` is appended here, as the
    -- reference appends it after tokenizing.
    if inputs.seed then loom.seed_rng(inputs.seed) end
    local _n_text = #inputs.tokens
    if _n_text == 0 then error('voxcpm2: no text ids') end
    local _ids = {}
    for _i = 1, _n_text do _ids[_i] = inputs.tokens[_i] end
    _ids[_n_text + 1] = AUDIO_START_TOKEN
    local _n = _n_text + 1
    local _cfg = inputs.cfg or DEFAULT_CFG
    local _steps = inputs.n_timesteps or DEFAULT_TIMESTEPS
    local _schedule = (_steps == DEFAULT_TIMESTEPS) and loom.get_weight('dit_step', 'euler_schedule')
                      or voxcpm_schedule(_steps, SWAY_COEF)
    -- CFG-Zero*'s zero-init: the first steps' velocity is ZERO, so `x - dt * 0` leaves the draw as it
    -- is and the step costs nothing. Skipped, not run.
    local _zero_init = math.max(1, math.floor((_steps + 1) * ZERO_INIT_FRACTION))
    local _patch = PATCH_SIZE * LATENT_DIM
    -- `_generate`'s budget: `ratio * len(target ids) + 10` patches, and the cache's room.
    local _budget = inputs.max_frames or math.min(math.floor(_n_text * BADCASE_RATIO + 10), MAX_LEN)
    _budget = math.min(_budget, LM_MAX_POSITIONS - _n)
    if _budget < 1 then error(string.format('voxcpm2: %d ids leave no room in a %d-position cache', _n,
                                            LM_MAX_POSITIONS)) end
    -- A caller's pinned draws (`inputs.noise`, one PATCH-major `[4][64]` block per step taken, in the
    -- order the reference draws them) and teacher patches (`inputs.teacher_patches`, one per patch
    -- kept), which feed the reference's history back in place of this loop's own (loom.cpp
    -- Retro-055): the loop is a feedback system, and two correct implementations drift apart along it.
    local _forced = inputs.teacher_patches
    local _draw = 0
    -- `generate`'s `retry_badcase`: a run that spends its whole ratio budget is retried with fresh
    -- draws, up to three times, and the last attempt stands. A caller's own `max_frames` or pinned
    -- noise is a measurement, not a generation, and is never retried.
    local _attempts = (inputs.max_frames or inputs.noise) and 1 or 3
    local _latents, _n_patches

    for _attempt = 1, _attempts do
        -- The prefill: the text's rows, with every feature patch zero (the prompt has no audio), which
        -- the feature encoder still encodes because the reference does -- `audio_mask` then zeroes it.
        local _zeros, _ones, _nil = {}, {}, {}
        for _i = 1, _n * _patch do _zeros[_i] = 0 end
        for _i = 1, _n do _ones[_i], _nil[_i] = 1, 0 end
        loom.run_subgraph_and_retain('feat_encode', {n_tokens = _n, n_past = 0}, {patches = _zeros})
        loom.run_subgraph_and_retain('base_lm', {n_tokens = _n, n_past = 0},
            {text_ids = _ids, feat_embed = {from = 'feat_encode'}, text_mask = _ones, audio_mask = _nil,
             position_ids = loom.range(0, _n), attention_mask = loom.causal_mask(_n, 0)})
        loom.run_subgraph_and_retain('residual_lm', {n_tokens = _n, n_past = 0},
            {enc = {from = 'base_lm', index = 1}, feat_embed = {from = 'feat_encode'}, audio_mask = _nil,
             attention_mask = loom.causal_mask(_n, 0)})
        local _n_past = _n

        -- `prefix_feat_cond` starts as the prompt's last patch, which is zeros here.
        local _cond = {}
        for _i = 1, _patch do _cond[_i] = 0 end
        _latents, _n_patches = {}, 0
        for _step = 0, _budget - 1 do
            -- One patch: a unit draw, integrated from t = 1 to 0 by the guided DiT.
            local _x
            if inputs.noise ~= nil then
                _x = {}
                for _i = 1, _patch do _x[_i] = inputs.noise[_draw * _patch + _i] end
                if _x[_patch] == nil then error('voxcpm2: inputs.noise ran out at step ' .. _step) end
            else
                _x = loom.gaussian_array(_patch)
            end
            _draw = _draw + 1
            for _k = _zero_init + 1, _steps do
                loom.run_subgraph_and_retain('dit_step', {n_tokens = 1, n_past = 0},
                    {lm_hidden = {from = 'base_lm', index = 2}, residual_hidden = {from = 'residual_lm'},
                     cond = _cond, x = _x, t = {_schedule[2 * _k - 1]}, dt = {_schedule[2 * _k]},
                     cfg = {_cfg}})
                _x = loom.get_output('dit_step', 1)
            end
            for _i = 1, _patch do _latents[_n_patches * _patch + _i] = _x[_i] end
            _n_patches = _n_patches + 1

            -- What the NEXT step is conditioned on: this patch, or the teacher's.
            local _fed = _x
            if _forced ~= nil and _forced[_n_patches * _patch] ~= nil then
                _fed = {}
                for _i = 1, _patch do _fed[_i] = _forced[(_n_patches - 1) * _patch + _i] end
            end
            loom.run_subgraph_and_retain('feat_encode', {n_tokens = 1, n_past = 0}, {patches = _fed})
            _cond = _fed

            -- The stop head on the row this patch was generated from; the reference's argmax, so a
            -- tie is "go on". Ignored until the patch after `min_len`.
            local _stop = loom.get_output('base_lm', 3)
            if _step > MIN_LEN and _stop[2] > _stop[1] then break end
            if _step == _budget - 1 then break end

            loom.run_subgraph_and_retain('base_lm', {n_tokens = 1, n_past = _n_past},
                {text_ids = {0}, feat_embed = {from = 'feat_encode'}, text_mask = {0}, audio_mask = {1},
                 position_ids = {_n_past}, attention_mask = loom.causal_mask(1, _n_past)})
            loom.run_subgraph_and_retain('residual_lm', {n_tokens = 1, n_past = _n_past},
                {enc = {from = 'base_lm', index = 1}, feat_embed = {from = 'feat_encode'}, audio_mask = {1},
                 attention_mask = loom.causal_mask(1, _n_past)})
            _n_past = _n_past + 1
        end
        -- `retry_badcase_ratio_threshold`: as long as the text times the ratio is a badcase.
        if _n_patches < _n_text * BADCASE_RATIO then break end
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
