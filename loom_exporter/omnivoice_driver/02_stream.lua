-- ===== OmniVoice: generation, one text chunk at a time; the live door (loom.cpp ADR-131). =====
--
-- The reference generates a text whose estimated audio is under `audio_chunk_threshold` (30 s) in one
-- pass, and a longer one in `audio_chunk_duration` (15 s) text chunks -- each its own masked-diffusion
-- run, decoded on its own, the chunks cross-faded. A run cannot emit audio before its last unmasking
-- step, so a CHUNK is the smallest piece this model has, and the stream is that loop cut per chunk:
-- `stream_open` plans every chunk (texts, frame counts, draws), each `stream_feed` generates, decodes
-- and returns one. `infer` runs the same chunks and post-processes the joined audio once, as the
-- reference does; a stream post-processes each piece -- the level and the silence trim are per piece
-- there, which is the one way the two differ (ADR-131). A stream_open option `chunk_threshold` /
-- `chunk_duration` (seconds) lowers both to cut a long text into more, shorter pieces.
--
-- Between calls everything is the stream's state: the chunks' prompts as ids, the reference (codes,
-- transcript, level), every draw, the knobs, and -- with no reference -- chunk 0's codes, which every
-- later chunk is prompted with so the voice stays one voice.

-- A reference clip -> its codes, through `ref_encode`; defined with the encoder (04_encode.lua).
local omni_encode_reference

local function omni_f32_schedule(num_step, t_shift, total)
    -- `_get_time_steps` at float32 (torch.linspace's halfway kernel, then the shift elementwise), read
    -- back as Python floats; then `_generate_iterative`'s per-step counts in double.
    local steps = num_step + 1
    local step = to_f32(1.0 / num_step)
    local half = math.floor(steps / 2)
    local shift, shift_m1 = to_f32(t_shift), to_f32(t_shift - 1)
    local ts = {}
    for idx = 0, num_step do
        local t
        if idx < half then t = to_f32(step * idx) else t = to_f32(1.0 - to_f32(step * (steps - idx - 1))) end
        ts[idx + 1] = to_f32(to_f32(shift * t) / to_f32(1.0 + to_f32(shift_m1 * t)))
    end
    local sched, rem = {}, total
    for s = 1, num_step do
        local num
        if s == num_step then num = rem
        else num = math.min(math.ceil(total * (ts[s + 1] - ts[s])), rem) end
        sched[s] = num
        rem = rem - num
    end
    return sched
end

local function omni_slice_list(flat, offsets, i)
    local out = {}
    for k = offsets[i] + 1, offsets[i + 1] do out[#out + 1] = flat[k] end
    return out
end

-- The style prefix: `[<|denoise|>] <|lang_start|>lang<|lang_end|> <|instruct_start|>instruct<|instruct_end|>`.
local function omni_style_ids(denoise, language, instruct)
    local ids = {}
    if denoise then ids[#ids + 1] = DENOISE_ID end
    ids[#ids + 1] = LANG_START_ID
    omni_append(ids, omni_slice_list(LANG_TOKEN_IDS, LANG_TOKEN_OFFSETS, language + 1))
    ids[#ids + 1] = LANG_END_ID
    ids[#ids + 1] = INSTRUCT_START_ID
    if instruct ~= nil then omni_append(ids, loom.tokenize(instruct)) else omni_append(ids, NONE_IDS) end
    ids[#ids + 1] = INSTRUCT_END_ID
    return ids
end

-- One chunk's masked-diffusion run: `_generate_iterative` at batch 1. `prompt` is the conditional
-- sequence's ids before the target (style, text, reference codes as `ref`), `t` the target's frames.
-- Returns the codes, codebook-major (`codes[c * t + j + 1]`).
local function omni_generate(prefix, ref, n_ref, t, draws, draw_pos)
    local n_prefix = #prefix
    local L = n_prefix + n_ref + t
    local N = L + t
    local tokens = {}
    for k = 1, N_CODEBOOKS * t do tokens[k] = MASK_ID end
    local audio_mask, positions, segment = {}, {}, {}
    for p = 1, N do
        audio_mask[p] = (p > n_prefix) and 1 or 0
        positions[p] = (p <= L) and (p - 1) or (p - L - 1)
        segment[p] = (p <= L) and 1 or 0
    end
    local cond_rows, uncond_rows = {}, {}
    for j = 0, t - 1 do cond_rows[j + 1] = L - t + j uncond_rows[j + 1] = L + j end
    local temperature = loom.state('position_temperature')
    local knobs = {loom.state('guidance_scale'), loom.state('layer_penalty'),
                   temperature > 0 and temperature or 1.0, temperature > 0 and 1 or 0}
    local schedule = omni_f32_schedule(loom.state('num_step'), loom.state('t_shift'), N_CODEBOOKS * t)
    -- `stop_after` (a gate's): the codes after that many steps, still-masked cells left at MASK_ID.
    local stop_after = loom.state('stop_after') or #schedule
    for s = 1, math.min(#schedule, stop_after) do
        local k = schedule[s]
        if k > 0 then
            -- The ids, row c of the conditional sequence then its own target again (the unconditional
            -- half); text positions repeat the text id in every row, as the reference's input does.
            local ids = {}
            for c = 0, N_CODEBOOKS - 1 do
                for p = 1, n_prefix do ids[#ids + 1] = prefix[p] end
                for r = 1, n_ref do ids[#ids + 1] = ref[c * n_ref + r] end
                for _ = 1, 2 do for j = 1, t do ids[#ids + 1] = tokens[c * t + j] end end
            end
            local uniforms = {}
            for q = 1, N_CODEBOOKS * t do uniforms[q] = draws[draw_pos + q] end
            draw_pos = draw_pos + N_CODEBOOKS * t
            loom.run_subgraph_and_retain('lm_step', {n_tokens = N, n_codes = t, n_past = 0},
                {ids = ids, audio_mask = audio_mask, position_ids = positions, segment = segment,
                 cond_rows = cond_rows, uncond_rows = uncond_rows, uniforms = uniforms, knobs = knobs})
            local pred = loom.get_output('lm_step', 1)
            local score = loom.get_output('lm_step', 2)
            -- `masked_fill(already unmasked, -inf)` then `topk(k)` over the flattened 8 x t cells.
            local open = {}
            for q = 1, N_CODEBOOKS * t do
                if tokens[q] == MASK_ID then open[#open + 1] = q end
            end
            table.sort(open, function(a, b)
                if score[a] ~= score[b] then return score[a] > score[b] end
                return a < b
            end)
            for r = 1, math.min(k, #open) do tokens[open[r]] = pred[open[r]] end
        end
    end
    return tokens, draw_pos
end

local function omni_decode(codes, t)
    return loom.run_subgraph('codec_decode', {n_codes = t, n_past = 0}, {codes = codes})
end

local function omni_put(name, value, default)
    if value == nil then value = default end
    if value ~= nil then loom.set_state(name, value) end
end

function stream_open(inputs)
    if inputs.tokens == nil or #inputs.tokens == 0 then error('omnivoice: no text ids') end
    omni_put('num_step', inputs.n_steps or inputs.num_step, NUM_STEP)
    omni_put('guidance_scale', inputs.guidance_scale, GUIDANCE_SCALE)
    omni_put('t_shift', inputs.t_shift, T_SHIFT)
    omni_put('layer_penalty', inputs.layer_penalty, LAYER_PENALTY)
    omni_put('position_temperature', inputs.position_temperature, POSITION_TEMPERATURE)
    omni_put('postprocess', inputs.postprocess, 1)
    omni_put('preprocess_prompt', inputs.preprocess_prompt, 1)
    omni_put('return_codes', inputs.return_codes, 0)
    omni_put('stop_after', inputs.stop_after, nil)
    local denoise = (inputs.denoise or 1) ~= 0
    local speed = inputs.speed or 1.0
    local language = inputs.language or 0
    if language < 0 or language > #LANG_TOKEN_OFFSETS - 1 then error('omnivoice: language index out of range') end

    local text = omni_codepoints(loom.detokenize(inputs.tokens))
    local instruct = nil
    if inputs.instruct ~= nil and #inputs.instruct > 0 then
        local use_zh = false
        for _, c in ipairs(text) do if omni_is_cjk(c) then use_zh = true end end
        instruct = omni_resolve_instruct(loom.detokenize(inputs.instruct), use_zh)
    end

    -- The reference: codes `[8, R]` codebook-major and the transcript, from `ref_encode` (a clip) or
    -- handed in directly. `ref_rms` is the clip's level before any gain, -1 when there is none.
    local ref, n_ref, ref_text, ref_rms = {}, 0, nil, -1
    if inputs.waveform ~= nil then
        ref, n_ref, ref_rms = omni_encode_reference(inputs.waveform)
    elseif inputs.ref_codes ~= nil then
        ref, n_ref = inputs.ref_codes, math.floor(#inputs.ref_codes / N_CODEBOOKS)
        ref_rms = inputs.ref_rms or -1
    end
    if n_ref > 0 then
        if inputs.ref_text == nil or #inputs.ref_text == 0 then
            error('omnivoice: a reference needs its transcript (ref_text ids)')
        end
        ref_text = omni_codepoints(loom.detokenize(inputs.ref_text))
        if (inputs.preprocess_prompt or 1) ~= 0 then ref_text = omni_add_punctuation(ref_text) end
    end

    -- The length: a fixed `duration` in seconds, else the rule estimate scaled by `speed`.
    local est = omni_target_tokens(text, ref_text, n_ref > 0 and n_ref or nil,
                                   inputs.duration ~= nil and 1.0 or speed)
    local target, chunk_speed = est, speed
    if inputs.duration ~= nil then
        target = math.max(1, math.floor(inputs.duration * FRAME_RATE))
        chunk_speed = est / target
    end

    -- The chunk plan: one chunk under the threshold, else `chunk_text_punctuation` on the target
    -- text, each chunk's frames re-estimated (with chunk 0 as the reference when there is none).
    local threshold = math.floor((inputs.chunk_threshold or AUDIO_CHUNK_THRESHOLD) * FRAME_RATE)
    local chunks, lens = {text}, {target}
    if target > threshold then
        local per_char = target / #text
        local chunk_len = math.floor((inputs.chunk_duration or AUDIO_CHUNK_DURATION) * FRAME_RATE / per_char)
        chunks = omni_chunk_text(text, chunk_len, 3)
        lens = {}
        for i, c in ipairs(chunks) do
            if n_ref > 0 then
                lens[i] = omni_target_tokens(c, ref_text, n_ref, chunk_speed)
            elseif i == 1 then
                lens[i] = omni_target_tokens(c, nil, nil, chunk_speed)
            else
                lens[i] = omni_target_tokens(c, chunks[1], lens[1], chunk_speed)
            end
        end
    end

    -- Every chunk's prompt prefix as ids: style (`<|denoise|>` whenever the chunk has reference codes,
    -- which chunk 1+ of an unreferenced text does) and the combined text.
    local flat, offsets = {}, {0}
    for i, c in ipairs(chunks) do
        local referenced = n_ref > 0 or i > 1
        omni_append(flat, omni_style_ids(denoise and referenced, language, instruct))
        local rt = ref_text
        if n_ref == 0 and i > 1 then rt = chunks[1] end
        omni_append(flat, omni_text_ids(omni_combine_text(c, rt)))
        offsets[#offsets + 1] = #flat
    end

    -- Every draw, up front, in the reference's order: chunk by chunk, step by step, skipping the steps
    -- whose count is zero (the reference `continue`s before drawing).
    local n_draws = 0
    local num_step, t_shift = loom.state('num_step'), loom.state('t_shift')
    for _, t in ipairs(lens) do
        for _, k in ipairs(omni_f32_schedule(num_step, t_shift, N_CODEBOOKS * t)) do
            if k > 0 then n_draws = n_draws + N_CODEBOOKS * t end
        end
    end
    local draws = inputs.draws
    if draws ~= nil then
        if #draws < n_draws then
            error(('omnivoice: %d pinned draws given, this text needs %d'):format(#draws, n_draws))
        end
    else
        loom.seed_rng(inputs.seed or 0)
        draws = loom.uniform_array(n_draws)
    end

    loom.set_state('prefix', flat)
    loom.set_state('prefix_offsets', offsets)
    loom.set_state('lens', lens)
    loom.set_state('ref', ref)
    loom.set_state('n_ref', n_ref)
    loom.set_state('ref_rms', ref_rms)
    loom.set_state('draws', draws)
    loom.set_state('draw_pos', 0)
    loom.set_state('chunk', 0)
    loom.set_state('done', 0)
    return {}
end

-- Chunk `i` (0-based): its codes, generated against the right reference, then its raw waveform.
local function omni_chunk(i)
    local lens = loom.state('lens')
    local t = lens[i + 1]
    local prefix = omni_slice_list(loom.state('prefix'), loom.state('prefix_offsets'), i + 1)
    local ref, n_ref = loom.state('ref'), loom.state('n_ref')
    if n_ref == 0 and i > 0 then ref, n_ref = loom.state('chunk0_codes'), lens[1] end
    local codes, draw_pos = omni_generate(prefix, ref, n_ref, t, loom.state('draws'), loom.state('draw_pos'))
    loom.set_state('draw_pos', draw_pos)
    if i == 0 and loom.state('n_ref') == 0 then loom.set_state('chunk0_codes', codes) end
    loom.set_state('chunk', i + 1)
    if i + 1 >= #lens then loom.set_state('done', 1) end
    return codes, t
end

function stream_feed(inputs)
    local i = loom.state('chunk')
    local n = #loom.state('lens')
    if i >= n then return {} end
    local codes, t = omni_chunk(i)
    -- `return_codes` (a gate's) is an OPEN option: a door feeds with no inputs of its own.
    if loom.state('return_codes') ~= 0 then return codes end
    local wave = omni_decode(codes, t)
    if n > 1 then wave = omni_cross_fade_edges(wave, i == 0, i == n - 1) end
    local ref_rms = loom.state('ref_rms')
    if loom.state('n_ref') == 0 then ref_rms = nil end
    return omni_post_process(wave, ref_rms, loom.state('postprocess'), i == 0, i == n - 1)
end

function stream_finish(inputs)
    return {}
end
