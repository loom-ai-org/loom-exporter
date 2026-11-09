-- ===== Qwen3-TTS's talker, live (loom.cpp ADR-102): one frame of codes per `stream_feed`. =====
--
-- `stream_open` builds the prompt, prefills the talker and draws frame 0; each `stream_feed` feeds the
-- previous frame back (its embedding plus one row of the text schedule, through one talker step) and
-- draws the next. The work is the one-pass loop's, cut where one iteration's feedback meets the next
-- one's draw, so the draws come in the same order and `infer` -- this stream opened, fed to its end and
-- finished in one call -- gives the codes it always gave.
--
-- Between calls everything is the stream's: the prompt's text schedule (`prompt`, kept on the device,
-- since every step reads a row of it), the last frame, the ids drawn so far (the repetition penalty's
-- history), the step and the knobs; the talker's history is the model's KV cache, leased or owned
-- (ADR-099). `piece` counts the draws and `done` is set at EOS or the frame budget.

-- The draw half of one step of the loop: codebook 0 from the talker's last row, then the code
-- predictor's fifteen. Returns the frame's sixteen codes, or {} at EOS.
local function qwen3_draw(step)
    -- **`hi` is how `min_new_tokens` is expressed**, and it costs nothing because the export already
    -- trimmed the head to the drawable ids: EOS is its last row, so excluding it is a one-shorter window
    -- rather than a mask.
    local _hi = EOS_INDEX
    if step >= MIN_NEW_TOKENS then _hi = EOS_INDEX + 1 end

    -- **The repetition penalty is not optional here, and greedy is not an exception.** `transformers`
    -- applies it as a processor rather than a warper, so it moves an argmax too; a greedy decode without
    -- it never emits EOS and runs to `max_new_tokens`. `generated` is the driver's own history, which is
    -- the half of the knob the checkpoint cannot state.
    local _generated = loom.state('generated') or {}
    local _first = loom.sample_row('talker', 0, {
        temperature = loom.state('temperature'), top_k = loom.state('top_k'), top_p = loom.state('top_p'),
        lo = 0, hi = _hi,
        repetition_penalty = loom.state('repetition_penalty'), penalized = _generated})
    loom.set_state('piece', loom.state('piece') + 1)
    loom.set_state('step', step)
    if _first == EOS_INDEX then
        loom.set_state('done', 1)
        return {}
    end
    _generated[#_generated + 1] = _first
    loom.set_state('generated', _generated)

    -- The code predictor, conditioned on the talker's hidden state and on codebook 0.
    --
    -- **It is not KV-cached, and it re-reads its own prefix every step.** One KvCache serves a whole
    -- model with one per-layer width, and these two phases' K/V geometry does not agree -- so rather
    -- than force it, the smaller model stops needing a cache. It never exceeds sixteen positions, so the
    -- whole frame costs 135 row-forwards of a five-layer stack, against one 28-layer talker step beside
    -- it. What it buys is that the driver still passes only integers: the prefix is rebuilt in-graph
    -- from `_prefix`, one longer each step.
    local _hidden = {from = 'talker', index = 2}
    loom.run_subgraph_and_retain('predictor_prefill', {n_tokens = 2, n_past = 0},
        {first_id = {_first}, talker_hidden = _hidden,
         position_ids = loom.range(0, 2),
         attention_mask = loom.causal_mask(2, 0)})

    -- **Group `g` is the window `[g*V, (g+1)*V)` of one 15-way-concatenated head**, which is what keeps
    -- fifteen draws on one head. A windowed draw returns the ABSOLUTE index, so the number that comes
    -- back is already the merged embedding table's row -- `_prefix` holds those unchanged, and the real
    -- codes are recovered by subtracting the offset once, at the end.
    local _sub_temperature, _sub_top_k = loom.state('sub_temperature'), loom.state('sub_top_k')
    local _sub_top_p = loom.state('sub_top_p')
    local _frame, _prefix = {_first}, {}
    local _module = 'predictor_prefill'
    for _g = 0, N_GROUPS - 2 do
        local _row = loom.sample_row(_module, 0, {
            temperature = _sub_temperature, top_k = _sub_top_k, top_p = _sub_top_p,
            lo = _g * CODEBOOK_SIZE, hi = (_g + 1) * CODEBOOK_SIZE})
        _frame[#_frame + 1] = _row
        if _g < N_GROUPS - 2 then
            _prefix[#_prefix + 1] = _row
            local _n = #_prefix + 2
            loom.run_subgraph_and_retain('predictor_steps', {n_tokens = _n, n_past = 0},
                {first_id = {_first}, talker_hidden = _hidden, rows = _prefix,
                 position_ids = loom.range(0, _n),
                 attention_mask = loom.causal_mask(_n, 0)})
            _module = 'predictor_steps'
        end
    end

    local _codes = {_first}
    for _g = 0, N_GROUPS - 2 do
        _codes[#_codes + 1] = _frame[_g + 2] - _g * CODEBOOK_SIZE
    end
    -- The rows the next step feeds back: the merged table's, not the codes.
    loom.set_state('frame', _frame)
    if step + 1 >= loom.state('max_new') then loom.set_state('done', 1) end
    return _codes
end

function stream_open(inputs)
    -- The template. The talker only ever saw `<|im_start|>assistant\n{text}<|im_end|>\n` and, for
    -- the target, a second `<|im_start|>assistant\n` -- and everything below strips `TEXT_HEAD` /
    -- `TEXT_TAIL` ids off the ends as if they were that scaffolding. `text2codes(text)` sends the
    -- sentence alone, so ids that do not open with the template head are wrapped in it here. Ids
    -- that already do are the caller's own rendering and pass through unchanged; a sentence cannot
    -- open with `<|im_start|>`, which only the special-token string encodes to.
    local function _templated(ids, tail)
        if ids == nil then return nil end
        local wrapped = #ids >= #TEMPLATE_HEAD
        for _i = 1, #TEMPLATE_HEAD do
            if wrapped and ids[_i] ~= TEMPLATE_HEAD[_i] then wrapped = false end
        end
        if wrapped then return ids end
        local out = {}
        for _i = 1, #TEMPLATE_HEAD do out[#out + 1] = TEMPLATE_HEAD[_i] end
        for _i = 1, #ids do out[#out + 1] = ids[_i] end
        for _i = 1, #tail do out[#out + 1] = tail[_i] end
        return out
    end
    inputs.tokens = _templated(inputs.tokens, TEXT_TEMPLATE_TAIL)
    inputs.ref_tokens = _templated(inputs.ref_tokens, REF_TEMPLATE_TAIL)

    -- The voice. Either the caller's own x-vector -- 1024 floats it got from a previous call, or
    -- from anywhere else -- or one extracted here from a reference clip at 24 kHz. A topology input
    -- takes an array or a reference indifferently, so the two paths differ only in this assignment.
    local _spk
    if inputs.x_vector then
        _spk = inputs.x_vector
    else
        loom.run_subgraph_and_retain('speaker_encoder',
            {n_samples = #inputs.waveform, n_past = 0}, {waveform = inputs.waveform})
        _spk = {from = 'speaker_encoder'}
    end

    -- This model is streaming: only the first text token is in the prompt, and every later one is
    -- added to a generated frame's embedding, one per step. `schedule` is that sequence with
    -- `tts_pad` appended, so "use trailing[step] while there is one, else pad" becomes an index
    -- rather than a branch.
    local _language = inputs.language_id or DEFAULT_LANGUAGE_ID
    local _prompt

    -- THE PROMPT, in one of two modes. `x_vector_only` is ten rows and constant; ICL is nine rows
    -- plus the REFERENCE utterance replayed -- its transcript on the text stream, its codec frames on
    -- the codec stream, summed row for row -- which is how the model is shown a voice rather than
    -- told about one. Both modes carry the x-vector; ICL is the nested arm, not the alternative.
    --
    -- **The reference picks the shorter of the two streams and slices the other by its length.** Both
    -- lengths are known here -- one is a token count, the other a frame count -- so this driver does
    -- that arithmetic and the graph does none of it: `_icl_text` comes out exactly as long as the
    -- replay and `_schedule` is already the remainder.
    local _prefill_len, _rows
    if inputs.ref_code or inputs.ref_audio then
        -- **The reference's own codes, drawn here when the caller passes audio rather than codes.**
        -- This is the codec's ENCODER, four phases of it, living in the talker's file because the
        -- ICL prompt is its only caller (`_RefEncodeWrapper` says why at more length).
        local _ref_code = inputs.ref_code
        if not _ref_code then
            -- **Whole frames only, and that is the export's correctness condition rather than a
            -- convenience.** Every convolution in the encode stack pads by a length-derived amount
            -- that coremltools will not trace; trimmed to a multiple of the frame stride, that
            -- amount is exactly zero at all fifteen of them. The reference keeps a PARTIAL final
            -- frame (it ceil-divides); this drops it, which is at most 79 ms off the end of a
            -- reference clip and is the difference between codes that match the reference on every
            -- frame and codes that match it on all but the last.
            local _samples = math.floor(#inputs.ref_audio / ENCODE_STRIDE) * ENCODE_STRIDE
            if _samples < ENCODE_STRIDE then
                error('ref_audio is shorter than one codec frame (' .. ENCODE_STRIDE ..
                      ' samples at ' .. REF_SAMPLE_RATE .. ' Hz)')
            end
            local _clip = {}
            for _i = 1, _samples do _clip[_i] = inputs.ref_audio[_i] end
            local _conv_rows = _samples / CONV_STRIDE
            local _frames = _samples / ENCODE_STRIDE

            -- Causal, NOT the 250-wide window the codec's config declares -- the reference's own
            -- `create_causal_mask` does not apply it on this path, and applying it moved 67 of 2192
            -- ids at 137 frames. `encoder_attention_mask` carries the finding.
            loom.run_subgraph_and_retain('ref_encode', {n_enc_frames = _conv_rows, n_past = 0},
                {waveform = _clip, attention_mask = loom.causal_mask(_conv_rows, 0)})

            -- One id per frame per codebook, drawn stage by stage: the graph scores, the driver
            -- reduces, the ids come back in as the next stage's subtraction. `subtract = 0` is how
            -- the first draw of each chain skips a subtraction it has nothing for.
            local _zeros = {}
            for _i = 1, _frames do _zeros[_i] = 0 end
            local _first_range = loom.range(0, REF_CODEBOOK)

            loom.run_subgraph_and_retain('rvq_project_semantic', {n_codes = _frames, n_past = 0},
                {rows = {from = 'ref_encode'}})
            loom.run_subgraph_and_retain('rvq_step', {n_codes = _frames, n_past = 0},
                {rows = {from = 'rvq_project_semantic'}, prev_ids = _zeros,
                 prev_codebook = _first_range, next_codebook = _first_range, subtract = {0.0}})
            local _drawn = {loom.argmax_rows('rvq_step')}

            loom.run_subgraph_and_retain('rvq_project_acoustic', {n_codes = _frames, n_past = 0},
                {rows = {from = 'ref_encode'}})
            local _rows = {from = 'rvq_project_acoustic'}
            local _prev = _zeros
            local _subtract = 0.0
            for _stage = 1, N_GROUPS - 1 do
                local _prev_range = loom.range(math.max(_stage - 1, 1) * REF_CODEBOOK, REF_CODEBOOK)
                loom.run_subgraph_and_retain('rvq_step', {n_codes = _frames, n_past = 0},
                    {rows = _rows, prev_ids = _prev, prev_codebook = _prev_range,
                     next_codebook = loom.range(_stage * REF_CODEBOOK, REF_CODEBOOK),
                     subtract = {_subtract}})
                _prev = loom.argmax_rows('rvq_step')
                _drawn[#_drawn + 1] = _prev
                -- `rvq_step`'s second output is the residual it just produced, and it stays in the
                -- engine: the next stage reads it by index rather than through Lua.
                _rows = {from = 'rvq_step', index = 2}
                _subtract = 1.0
            end

            -- Frame-major, which is the layout every codec in this tree passes codes in.
            _ref_code = {}
            for _f = 1, _frames do
                for _g = 1, N_GROUPS do
                    _ref_code[#_ref_code + 1] = _drawn[_g][_f]
                end
            end
        end

        -- `math.floor`, because Lua's `/` is float division and these two numbers become an axis
        -- extent and a loop bound.
        local _n_ref = math.floor(#_ref_code / N_GROUPS)
        local _n_replay = _n_ref + 1

        -- The interleaved id stream: the reference transcript, the target text, then `tts_eos`.
        -- Both are stripped of their own template, which is `ref_ids[:, 3:-2]` and
        -- `input_id[:, 3:-5]` in the reference implementation.
        local _stream = {}
        for _i = REF_HEAD + 1, #inputs.ref_tokens - REF_TAIL do
            _stream[#_stream + 1] = inputs.ref_tokens[_i]
        end
        for _i = TEXT_HEAD + 1, #inputs.tokens - TEXT_TAIL do
            _stream[#_stream + 1] = inputs.tokens[_i]
        end
        _stream[#_stream + 1] = TTS_EOS_ID

        -- Cut it where the replay ends: what fits is the prompt's text half, what is left is the
        -- schedule. A stream shorter than the replay is padded, which is the reference's other arm.
        local _icl_text, _schedule = {}, {}
        for _i = 1, _n_replay do
            _icl_text[_i] = _stream[_i] or TTS_PAD_ID
        end
        for _i = _n_replay + 1, #_stream do
            _schedule[#_schedule + 1] = _stream[_i]
        end
        if #_schedule == 0 then _schedule[1] = TTS_PAD_ID end

        -- The codes, offset into the merged predictor table exactly as a drawn frame's are, with a
        -- leading column of zeros where `codec_bos` goes -- `bos_mask` selects it, so the replay
        -- needs no concatenation and no second symbol for the sake of one row.
        --
        -- **GROUP-major, transposed here from the frame-major layout every codec in this tree passes
        -- codes in.** A group has to be a contiguous run for `ggml_get_rows`: sliced out of a
        -- frame-major array it is a strided view, and a strided view aborts the process inside ggml
        -- rather than raising. One transpose here buys a legal gather in sixteen places.
        local _codes_in, _mask = {}, {}
        _mask[1] = 1.0
        for _f = 1, _n_ref do _mask[_f + 1] = 0.0 end
        for _g = 1, N_GROUPS do
            _codes_in[#_codes_in + 1] = 0            -- the `codec_bos` column
            local _offset = (_g - 2) * CODEBOOK_SIZE
            for _f = 0, _n_ref - 1 do
                local _code = _ref_code[_f * N_GROUPS + _g]
                if _g > 1 then _code = _code + _offset end
                _codes_in[#_codes_in + 1] = _code
            end
        end

        local _role = {}
        for _i = 1, TEXT_HEAD do _role[_i] = inputs.tokens[_i] end

        loom.run_subgraph_and_retain('prefill_embed_icl',
            {n_codes = _n_replay, n_tokens = #_schedule, n_past = 0},
            {role_ids = _role, icl_text_ids = _icl_text, ref_code = _codes_in, bos_mask = _mask,
             schedule_ids = _schedule, language_id = {_language}, x_vector = _spk})
        _prefill_len = ICL_HEAD_LEN + _n_replay
        _rows = #_schedule + 1
        _prompt = 'prefill_embed_icl'
    else
        loom.run_subgraph_and_retain('prefill_embed', {n_tokens = #inputs.tokens, n_past = 0},
            {text_ids = inputs.tokens, language_id = {_language}, x_vector = _spk})
        -- The template contributes three tokens before the text and five after, and the schedule
        -- drops the first text token (it is in the prompt) and appends `tts_eos` and `tts_pad`. So
        -- its row count is `#tokens - 9 + 2`, and the driver needs it only to clamp the index.
        _prefill_len = PREFILL_LEN
        _rows = #inputs.tokens - 7
        _prompt = 'prefill_embed'
    end

    loom.run_subgraph_and_retain('talker', {n_tokens = _prefill_len, n_past = 0},
        {inputs_embeds = {from = _prompt, index = 1},
         position_ids = loom.range(0, _prefill_len),
         attention_mask = loom.causal_mask(_prefill_len, 0)})

    -- Every step reads a row of the prompt's text schedule (output 2), so it is the stream's own rather
    -- than a retained output another stream's prompt would overwrite between two feeds.
    loom.keep_state('prompt', _prompt)
    if inputs.seed then loom.seed_rng(inputs.seed) end
    loom.set_state('temperature', inputs.temperature or TEMPERATURE)
    loom.set_state('top_k', inputs.top_k or TOP_K)
    loom.set_state('top_p', inputs.top_p or TOP_P)
    loom.set_state('repetition_penalty', inputs.repetition_penalty or REPETITION_PENALTY)
    loom.set_state('sub_temperature', inputs.subtalker_temperature or SUB_TEMPERATURE)
    loom.set_state('sub_top_k', inputs.subtalker_top_k or SUB_TOP_K)
    loom.set_state('sub_top_p', inputs.subtalker_top_p or SUB_TOP_P)
    loom.set_state('max_new', inputs.max_new_tokens or MAX_NEW_TOKENS)
    loom.set_state('prefill_len', _prefill_len)
    loom.set_state('rows', _rows)
    loom.set_state('piece', 0)
    loom.set_state('done', 0)
    if loom.state('max_new') <= 0 then
        loom.set_state('done', 1)
        return {}
    end
    return qwen3_draw(0)
end

function stream_feed(inputs)
    if loom.state('done') ~= 0 then return {} end
    local _step = loom.state('step') + 1
    local _prefill_len, _rows = loom.state('prefill_len'), loom.state('rows')
    -- The previous frame's sixteen embeddings summed, plus one row of the text schedule, is the talker's
    -- next input. Both operands stay engine-side: the codes are integers and the schedule is the
    -- stream's own copy of the prompt's output, indexed by a row number.
    local _schedule_row = _step - 1
    if _schedule_row > _rows - 1 then _schedule_row = _rows - 1 end
    loom.run_subgraph_and_retain('frame_embed', {n_tokens = _rows, n_past = 0},
        {codes = loom.state('frame'), schedule_row = {_schedule_row},
         schedule = {state = 'prompt', index = 2}})
    loom.run_subgraph_and_retain('talker', {n_tokens = 1, n_past = _prefill_len + _step - 1},
        {inputs_embeds = {from = 'frame_embed'},
         position_ids = loom.range(_prefill_len + _step - 1, 1),
         attention_mask = loom.causal_mask(1, _prefill_len + _step - 1)})
    return qwen3_draw(_step)
end

function stream_finish(inputs)
    return {}
end
