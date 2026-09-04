    -- The encoder frame count. One frame per input byte -- this encoder neither subsamples nor pads --
    -- so the caller's own byte count is it, and it is bound as an axis on every decoder call because
    -- the cross-attention K/V carry it as their own dynamic dimension (dia_export.py's
    -- `declared_axes`). It is the second symbol this topology resolves; `n_tokens` is the first.
    local _n_enc = #inputs.tokens

    -- One decoder call's inputs. The cross-attention K/V are a `cross_kv` module's retained outputs,
    -- bound by index -- output `2*layer + 1` is that layer's K and `+ 2` its V, the interleaving
    -- `_DiaCrossKvWrapper` returns and `cross_kv_input_names` names. Built in a loop because there are
    -- two per layer and the count is a property of the checkpoint.
    --
    -- `_kv` names WHICH cross_kv module, because classifier-free guidance has two: the conditional
    -- projection of the caller's text and the unconditional projection of an all-zero prompt. Every
    -- other input is shared -- the two streams are fed the same codes at the same positions, exactly
    -- as `transformers` batches them.
    local function _decoder_inputs(_frame, _n_past, _kv)
        local _in = {
            codes = _frame,
            position_ids = loom.range(_n_past, 1),
            attention_mask = loom.causal_mask(1, _n_past),
        }
        for _l = 0, N_LAYERS - 1 do
            _in["xk_" .. _l] = {from = _kv, index = 2 * _l + 1}
            _in["xv_" .. _l] = {from = _kv, index = 2 * _l + 2}
        end
        return _in
    end

    -- The decoding knobs: the caller's, else the checkpoint's own, which the export read out of its
    -- `generation_config.json`. `temperature = 0` is greedy and is what a caller passes to reproduce
    -- a reference exactly; this checkpoint's own default is 1.8, so the two are genuinely different
    -- models of what "run this file" means and the file's answer is the default.
    -- **A sampling driver needs a seed, and this is the only place a caller can set one.** Without
    -- it the same sentence gives different audio every call with no way to reproduce either, which is
    -- the state every other TTS driver here already refuses to be in. Named only -- an unseeded call
    -- keeps whatever stream the bridge is on, so a caller who wants variety simply does not pass one.
    if inputs.seed then loom.seed_rng(inputs.seed) end

    local _temperature = inputs.temperature or TEMPERATURE
    local _top_k = inputs.top_k or TOP_K
    local _top_p = inputs.top_p or TOP_P

    -- **Classifier-free guidance, and the arithmetic of the scale is Dia's.** Its own processor
    -- combines as `cond + g * (cond - uncond)`, centred on the CONDITIONAL logits, where the general
    -- form `loom.sample_row` implements is centred on the unconditional ones. The two are the same
    -- family: `cond + g*(cond - uncond)` is `uncond + (g + 1)*(cond - uncond)`. So the file declares
    -- the checkpoint's `g` and the `+ 1` happens here, next to the model whose convention it is.
    --
    -- `g <= 1` means off, which is `transformers`' own condition for not installing the processor at
    -- all -- and off is a real mode rather than a degenerate one, because it halves the work.
    local _guidance = inputs.guidance_scale or GUIDANCE_SCALE
    local _use_cfg = _guidance > 1.0

    -- The unconditional half: the same encoder over an all-zero prompt of the same length, projected
    -- into its own cross-attention K/V. `torch.zeros_like(inputs)` is exactly what
    -- `DiaGenerationMixin._prepare_model_inputs` builds, and the LENGTH matching matters -- both
    -- streams' cross-attention is over `n_enc_frames` frames, so a shorter unconditional prompt would
    -- be a different axis rather than a weaker condition.
    --
    -- It runs through the SAME encoder module, which is safe for the one reason worth stating: the
    -- encoder's retained output is consumed by `cross_kv` immediately and never read again, so the
    -- second pass overwriting the first costs nothing. The cross-attention K/V are the opposite --
    -- they are read at every step for the rest of the generation -- which is why THOSE are two
    -- modules and this is one.
    local _uncond_kv = nil
    if _use_cfg then
        local _blank = {}
        for _i = 1, _n_enc do _blank[_i] = 0 end
        loom.run_subgraph_and_retain('encoder', {n_tokens = _n_enc, n_past = 0}, {tokens = _blank})
        loom.run_subgraph_and_retain('cross_kv_uncond', {n_enc_frames = _n_enc, n_past = 0},
                                      {xa = {from = 'encoder'}})
        _uncond_kv = 'cross_kv_uncond'
    end

    -- Everything `loom.sample_row` needs that does not change from step to step. `guidance.top_k` is
    -- NOT `top_k`: it is `DiaClassifierFreeGuidanceLogitsProcessor`'s `guidance_top_k`, which uses the
    -- GUIDED logits to pick a shortlist and then draws from the CONDITIONAL ones restricted to it.
    -- `transformers` passes `generation_config.top_k` into that slot, so the same number lands in two
    -- roles, and both are spelled here rather than one being assumed from the other.
    local function _sample_opts(_lo, _hi, _greedy)
        local _o = {lo = _lo, hi = _hi,
                    temperature = _greedy and 0.0 or _temperature,
                    top_k = _greedy and 0 or _top_k,
                    top_p = _greedy and 1.0 or _top_p}
        if _use_cfg then
            _o.guidance = {module = 'decoder_uncond', scale = _guidance + 1.0, top_k = _top_k}
        end
        return _o
    end

    -- The DELAYED sequence, flat and frame-major: N_CHANNELS entries per row. Row 0 is the all-BOS
    -- frame the model conditions on -- `DiaProcessor` builds exactly this scaffold and calls the rest
    -- of it padding, so a driver that starts here needs no processor.
    local _rows = {}
    local _step = {}
    for _c = 1, N_CHANNELS do
        _rows[_c] = BOS
        _step[_c] = BOS
    end

    -- Rows already committed, which is also the 0-based index of the row about to be written.
    local _row = 1
    local _n_past = 0
    -- The row on which channel 0 said EOS, or -1 while it has not. Channel 0 is the only channel whose
    -- EOS means anything: it leads the delay pattern, so when it stops every other channel is still
    -- emitting frames that belong BEFORE it.
    local _eos_row = -1
    -- **The ceiling is a count of AUDIO FRAMES, and reaching it FORCES an EOS rather than breaking.**
    -- Both halves of that matter. A loop that simply stopped would return frames whose higher channels
    -- were never generated -- channel k's contribution to frame t lives MAX_DELAY rows further on --
    -- so the last MAX_DELAY frames would be short, and in Lua a short frame is not an error but a
    -- silently smaller array. Forcing EOS instead makes the ceiling take the SAME path a natural stop
    -- takes: the tail below runs, and every returned frame is complete on all nine channels.
    --
    -- It is also what `transformers` does. `DiaEOSDelayPatternLogitsProcessor` carries a
    -- `reached_max_len` clause beside its "channel 0 said EOS" one and feeds both into the same
    -- forcing, so a generation that runs out of budget is indistinguishable downstream from one that
    -- finished. A driver that truncated instead would disagree with the reference on the last frames
    -- of every capped generation.
    --
    -- The default leaves exactly MAX_DELAY rows of tail inside the decoder's own position budget.
    local _max_frames = inputs.max_new_tokens or (MAX_CODES - 1 - MAX_DELAY)

    while true do
        local _axes = {n_tokens = 1, n_past = _n_past, n_enc_frames = _n_enc}
        loom.run_subgraph_and_retain('decoder', _axes, _decoder_inputs(_step, _n_past, 'cross_kv'))
        if _use_cfg then
            -- The same codes, the same positions, its own KV cache (`kv_cache_scope: "private"` on
            -- the emitted topology) and the unconditional cross-attention. Second, so that the
            -- conditional module's retained output is the one `sample_row` names below.
            loom.run_subgraph_and_retain('decoder_uncond', _axes,
                                         _decoder_inputs(_step, _n_past, _uncond_kv))
        end
        _n_past = _n_past + 1

        -- **A RESTRICTED draw per channel, not one reduction over all nine.** The decoder graph
        -- slices its own last row before the head (`_DiaDecoderWrapper`), so its output is [1028, 9] --
        -- nine rows an unrestricted reduction would happily take whole-row answers from. That would be
        -- wrong: this model's four highest ids are control tokens (eos/pad/bos and one unused), and
        -- `DiaEOSChannelFilterLogitsProcessor` bans them per channel rather than globally. Only
        -- channel 0 may say EOS; no channel may say PAD or BOS. So the window is [0, EOS] for channel
        -- 0 and [0, EOS) for the rest.
        --
        -- **Channel 0 takes two calls, and that is the processor's other two clauses becoming live.**
        -- They read: force EOS when it is already the highest logit, suppress it when it is not --
        -- both no-ops under an argmax, which had already decided the same question, and both real
        -- under sampling, where an EOS that is merely likely would otherwise end the utterance early
        -- and one that is certain could still be missed. Spelled here as: ask which of [0, EOS] wins
        -- (temperature 0, which is that same argmax), and if it is EOS take it, otherwise draw from
        -- [0, EOS) with EOS excluded. That is exactly the two clauses, without needing the engine to
        -- express `-inf`.
        local _next = {}
        if loom.sample_row('decoder', 0, _sample_opts(0, EOS + 1, true)) == EOS then
            _next[1] = EOS
        else
            _next[1] = loom.sample_row('decoder', 0, _sample_opts(0, EOS, false))
        end
        for _c = 2, N_CHANNELS do
            _next[_c] = loom.sample_row('decoder', _c - 1, _sample_opts(0, EOS, false))
        end

        -- The delay scaffold. A channel that has not reached its own offset yet is not predicting an
        -- audio frame at all, so whatever it produced is discarded and BOS is fed back in its place.
        for _c = 1, N_CHANNELS do
            if _row <= DELAY_PATTERN[_c] then _next[_c] = BOS end
        end

        -- The ceiling, applied before the EOS is read so that both routes to stopping converge on one
        -- piece of state. `_row` is the row about to be written, so `_row > _max_frames` is the first
        -- row that would be a frame past the budget.
        if _eos_row < 0 and _row > _max_frames then _next[1] = EOS end
        if _eos_row < 0 and _next[1] == EOS then _eos_row = _row end

        -- Once channel 0 has stopped, every other channel's ending is DETERMINED rather than predicted:
        -- channel k says EOS exactly `delay[k]` rows later and pads after that
        -- (`DiaEOSDelayPatternLogitsProcessor`). Forcing it here matters even though none of these
        -- positions is read back as audio -- the loop below stops at row `_eos_row - 1 + delay[k]` for
        -- channel k, strictly before the forced one -- because they are fed back in, and the rows that
        -- ARE read depend on them through the KV cache.
        if _eos_row >= 0 then
            local _since = _row - _eos_row
            for _c = 1, N_CHANNELS do
                if _since > DELAY_PATTERN[_c] then
                    _next[_c] = PAD
                elseif _since == DELAY_PATTERN[_c] then
                    _next[_c] = EOS
                end
            end
        end

        for _c = 1, N_CHANNELS do
            _rows[_row * N_CHANNELS + _c] = _next[_c]
            _step[_c] = _next[_c]
        end
        _row = _row + 1

        if _eos_row >= 0 and _row > _eos_row + MAX_DELAY then break end
        -- A backstop, not the stop: the forced EOS above ends every generation well inside this. It is
        -- here because the loop writes into the KV cache, whose capacity is MAX_CODES, and a loop that
        -- could reach it would corrupt rather than fail.
        if _row >= MAX_CODES then break end
    end

    -- Undo the delay: audio frame t's channel k was emitted at row t + delay[k]. Frames start at 1 --
    -- row 0 is the BOS scaffold, never audio -- and where they STOP depends on which way the loop
    -- ended, which is the one asymmetry in this driver worth spelling out:
    --
    --   * channel 0 said EOS at `_eos_row`, so the last complete frame is the one before it, and the
    --     loop deliberately kept going for MAX_DELAY more rows to have somewhere to read the higher
    --     channels of that frame FROM. This is `[start_of_generation_idx : end_of_generation_idx]`,
    --     the window `DiaProcessor.batch_decode` cuts by counting BOS and PAD rows in channel 0.
    --   * the MAX_CODES backstop fired, which the forced EOS above makes unreachable in practice. The
    --     branch stays because it is the honest answer if it ever does: nothing was generated past
    --     `_row - 1`, so the last frame every channel can supply is `_row - 1 - MAX_DELAY`.
    --
    -- Getting the second case wrong is not a short answer, it is a SILENT one: `_rows[...]` past the
    -- end is `nil`, and `_codes[#_codes + 1] = nil` appends nothing at all, so the returned array is
    -- simply a few codes shorter with no frame boundary to notice it at. Measured, before the ceiling
    -- forced an EOS, on a synthetic checkpoint that never emits one: 186 codes came back where 189
    -- were due. That is what the `error()` below exists for.
    local _last = (_eos_row >= 0) and (_eos_row - 1) or (_row - 1 - MAX_DELAY)
    if _last < 0 then _last = 0 end
    local _codes = {}
    for _t = 1, _last do
        for _c = 1, N_CHANNELS do
            local _v = _rows[(_t + DELAY_PATTERN[_c]) * N_CHANNELS + _c]
            -- The bound above is derived, not clamped -- `build_indices` clamps, which would turn a
            -- broken stop condition into a duplicated frame. This says so instead, because the failure
            -- it guards is the invisible one described above.
            if _v == nil then
                error("dia driver: no row " .. (_t + DELAY_PATTERN[_c]) .. " for channel " .. _c ..
                      " (frames=" .. _last .. ", rows=" .. _row .. ", eos_row=" .. _eos_row .. ")")
            end
            _codes[#_codes + 1] = _v
        end
    end
