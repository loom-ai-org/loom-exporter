    -- The encoder frame count. One frame per input byte -- this encoder neither subsamples nor pads --
    -- so the caller's own byte count is it, and it is bound as an axis on every decoder call because
    -- the cross-attention K/V carry it as their own dynamic dimension (dia_export.py's
    -- `declared_axes`). It is the second symbol this topology resolves; `n_tokens` is the first.
    local _n_enc = #inputs.tokens

    -- One decoder call's inputs. The cross-attention K/V are `cross_kv`'s retained outputs, bound by
    -- index -- output `2*layer + 1` is that layer's K and `+ 2` its V, the interleaving
    -- `_DiaCrossKvWrapper` returns and `cross_kv_input_names` names. Built in a loop because there are
    -- two per layer and the count is a property of the checkpoint.
    local function _decoder_inputs(_frame, _n_past)
        local _in = {
            codes = _frame,
            position_ids = loom.range(_n_past, 1),
            attention_mask = loom.causal_mask(1, _n_past),
        }
        for _l = 0, N_LAYERS - 1 do
            _in["xk_" .. _l] = {from = 'cross_kv', index = 2 * _l + 1}
            _in["xv_" .. _l] = {from = 'cross_kv', index = 2 * _l + 2}
        end
        return _in
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
        loom.run_subgraph_and_retain('decoder',
                                     {n_tokens = 1, n_past = _n_past, n_enc_frames = _n_enc},
                                     _decoder_inputs(_step, _n_past))
        _n_past = _n_past + 1

        -- **A RESTRICTED argmax per channel, not one `loom.argmax_rows` over all nine.** The decoder
        -- graph slices its own last row before the head (`_DiaDecoderWrapper`), so its output is
        -- [1028, 9] -- nine rows the unrestricted reduction would happily take whole-row argmaxes of.
        -- That would be wrong: this model's four highest ids are control tokens (eos/pad/bos and one
        -- unused), and `DiaEOSChannelFilterLogitsProcessor` bans them per channel rather than globally.
        -- Only channel 0 may say EOS; no channel may say PAD or BOS. So the window is [0, EOS] for
        -- channel 0 and [0, EOS) for the rest, which is exactly `loom.argmax_row_range` -- the same
        -- restricted reduction whisper_driver uses to detect a language, and no new engine primitive.
        --
        -- The processor's other two clauses (force EOS when it is already the top logit, suppress it
        -- when it is not) are no-ops under a greedy argmax: both describe how SAMPLING must treat a
        -- token the argmax has already decided about. They become live the day this loop samples.
        local _next = {}
        _next[1] = loom.argmax_row_range('decoder', 0, 0, EOS + 1)
        for _c = 2, N_CHANNELS do
            _next[_c] = loom.argmax_row_range('decoder', _c - 1, 0, EOS)
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
