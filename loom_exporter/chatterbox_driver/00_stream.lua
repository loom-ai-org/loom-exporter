-- ===== Chatterbox, live (loom.cpp ADR-117): a hop of speech tokens per feed, the flow re-run. =====
--
-- S3Gen is CosyVoice2's flow and HiFT, and keeps CosyVoice2's streaming hooks: `flow_inference(finalize =
-- False)` runs the encoder over every token so far and drops the mel of the last LOOKAHEAD_TOKENS, which
-- the pre-lookahead conv reads as context. Chatterbox ships no loop around them (`S3GenStreamer` is named in
-- a docstring and not shipped), so this is CosyVoice2's `tts(stream=True)` loop over them: each hop the flow
-- re-runs over the voice's prompt and every token drawn so far, the mel past what the stream already has is
-- kept, and HiFT re-runs over ALL the mel, handing out the samples no later frame can change -- all but the
-- last VOCODER_RIGHT_FRAMES frames', which the export finds by marking the mel's tail and seeing which
-- samples it reaches. The hop is STREAM_HOP_TOKENS, doubling up to STREAM_MAX_HOP_TOKENS (`piece_tokens`
-- fixes it; 0 is the whole turn in one flow call, which `infer` runs).
--
-- **So a stream is not `infer`'s audio.** Chatterbox's encoder and U-Net attend over the whole sequence
-- (`static_chunk_size` 0 in both, where CosyVoice2 trained a chunk mask and CosyVoice3 streams with one,
-- ADR-112), so a hop's mel is what the flow makes without the tokens after it. The gate holds the stream
-- to the same loop run in PyTorch, and HiFT's streaming to its one call over the stream's mel.
--
-- **A seed is the same take both ways** (ADR-112's amendment): T3's uniforms are one block drawn when the
-- loop begins (one per step of the budget, used or not), then the 9 harmonic phases, then a FRAME at a
-- time in frame order the flow's 80 gaussians and, for a generated frame, the sine source's 9 x 480
-- (`chatterbox_draw_frames`). `infer` makes the same calls for the whole grid at once, so the tokens are
-- the same, and so is every noise value a frame meets. A caller may pin each (`noise` frame-major for the
-- whole grid, `nsf_phase`, `nsf_noise` harmonic-major for the whole wave); a pinned one is not drawn.
--
-- The functions hold nothing between calls: the text, the voice, the knobs, T3's loop (step, `n_past`,
-- the penalised history, the kept tokens, the uniforms), the hop, the mel so far and the samples handed
-- out are the stream's state; T3's history is the model's KV cache, leased or owned (ADR-099). Each
-- step feeds the previous draw back BEFORE it samples, so no graph output is read across a call.

local function chatterbox_keep(name, value)
    if value ~= nil then loom.set_state(name, value) end
end

-- What a stream keeps across feeds, besides what it is opened with.
local CHATTERBOX_LOOP = {'step', 'n_past', 'budget', 'draws', 'history', 'kept', 'ended', 'offset', 'hop',
                         'mel', 'flow_noise', 'nsf_blocks', 'frames_drawn', 'emitted'}

-- The ones that start empty: an empty array is not something to rely on reading back as one.
local CHATTERBOX_ARRAYS = {kept = true, mel = true, flow_noise = true, nsf_blocks = true}

local function chatterbox_load()
    local st = {}
    for _, name in ipairs(CHATTERBOX_LOOP) do
        st[name] = loom.state(name)
        if st[name] == nil and CHATTERBOX_ARRAYS[name] then st[name] = {} end
    end
    return st
end

local function chatterbox_save(st)
    for _, name in ipairs(CHATTERBOX_LOOP) do chatterbox_keep(name, st[name]) end
end

-- One T3 step: the decoder brought up to the previous draw (the prefill on the first step), then one draw
-- from the GUIDED logits `t3_lm` returns, the repetition penalty over the history (which starts at BOS, as
-- `T3.inference` seeds `generated_ids`), temperature and min_p, from the step's own uniform. Sets `ended` at
-- the stop token or the budget; a speech token (below SPEECH_VOCAB) is appended to `kept`. Control ids are
-- fed back like any other draw -- the reference drops them only on the way to S3Gen.
--
-- Both guidance streams are ONE run of `t3_lm` over a batch of two (conditional, then unconditional:
-- `T3LMPhase`), which combines them as the reference does, `cond + w * (cond - uncond)`, with `w` its `cfg`
-- input; at `w = 0` that is the conditional stream's logits.
local function chatterbox_t3_step(st)
    if st.ended ~= 0 then return end
    local cfg = {loom.state('cfg_weight')}
    if st.n_past == 0 then
        local text = loom.state('text')
        local n_text = #text
        local prefill = N_COND_ROWS + n_text + N_BOS_ROWS
        loom.run_subgraph_and_retain('t3_prefill_embed', {n_tokens = n_text, n_past = 0},
            {speaker_emb = chatterbox_voice(loom.state('speaker_emb'), 'voice.speaker_emb'),
             prompt_tokens = chatterbox_voice(loom.state('cond_prompt_tokens'), 'voice.cond_prompt_tokens'),
             emotion_adv = {loom.state('exaggeration')}, text_ids = text})
        loom.run_subgraph_and_retain('t3_lm', {n_tokens = prefill, n_past = 0},
            {inputs_embeds = {from = 't3_prefill_embed'}, position_ids = loom.range(0, prefill),
             attention_mask = loom.causal_mask(prefill, 0), cfg = cfg})
        st.n_past = prefill
    else
        -- Draw k sits at speech position k: the prefill's BOS rows are position 0.
        loom.run_subgraph_and_retain('t3_step_embed', {n_tokens = 1, n_past = 0},
            {token = {st.history[#st.history]}, position = {st.step}})
        local n_past = st.n_past
        loom.run_subgraph_and_retain('t3_lm', {n_tokens = 1, n_past = n_past},
            {inputs_embeds = {from = 't3_step_embed'}, position_ids = {n_past},
             attention_mask = loom.causal_mask(1, n_past), cfg = cfg})
        st.n_past = n_past + 1
    end
    local token = loom.sample_row('t3_lm', -1,
        {temperature = loom.state('temperature'), min_p = loom.state('min_p'), top_p = loom.state('top_p'),
         repetition_penalty = loom.state('repetition_penalty'), penalized = st.history,
         uniform = st.draws[st.step + 1]})
    st.step = st.step + 1
    st.history[#st.history + 1] = token
    if token == STOP_SPEECH then
        st.ended = 1
        return
    end
    if token < SPEECH_VOCAB then st.kept[#st.kept + 1] = token end
    if st.step >= st.budget then st.ended = 1 end
end

-- The flow's noise and the sine source's for grid frames `st.frames_drawn + 1 .. n_frames`, a frame at a
-- time: 80 gaussians onto `st.flow_noise` (frame-major) unless the caller pinned `noise`, then, for a
-- generated frame (past the voice's `n_prompt` prompt frames), 9 x 480 gaussians onto `st.nsf_blocks`
-- (harmonic-major within the frame) unless the caller pinned `nsf_noise`. Per frame, so where a caller
-- cuts the grid does not change the order.
local function chatterbox_draw_frames(st, n_frames, n_prompt)
    local per_frame = NSF_HARMONICS * SAMPLES_PER_FRAME
    local noise_pinned, nsf_pinned = loom.state('noise') ~= nil, loom.state('nsf_noise') ~= nil
    local noise, blocks = st.flow_noise, st.nsf_blocks
    for f = st.frames_drawn + 1, n_frames do
        if not noise_pinned then
            local g = loom.gaussian_array(N_MEL)
            for i = 1, N_MEL do noise[#noise + 1] = g[i] end
        end
        if f > n_prompt and not nsf_pinned then
            local g = loom.gaussian_array(per_frame)
            for i = 1, per_frame do blocks[#blocks + 1] = g[i] end
        end
    end
    if n_frames > st.frames_drawn then st.frames_drawn = n_frames end
end

-- S3Gen's flow over the voice's prompt tokens and the first `n_tokens` kept; returns the mel of the
-- generated frames from token `offset` to the end of the grid, frame-major. `finalize` is the last hop:
-- every token, the encoder's zero look-ahead pad at the end, as the reference's `finalize=True`. Otherwise
-- the last LOOKAHEAD_TOKENS tokens are context only: the encoder runs over all of them and its rows past
-- the hop are dropped, which is `CausalMaskedDiffWithXvec.inference`'s `h[:, :-pre_lookahead_len * 2]`.
local function chatterbox_flow(st, n_tokens, offset, finalize)
    local prompt_tokens = chatterbox_voice(loom.state('prompt_token'), 'voice.prompt_token')
    local prompt_feat = chatterbox_voice(loom.state('prompt_feat'), 'voice.prompt_feat')
    local embedding = chatterbox_voice(loom.state('embedding'), 'voice.embedding')
    local tokens = {}
    for i = 1, #prompt_tokens do tokens[i] = prompt_tokens[i] end
    for i = 1, n_tokens do tokens[#tokens + 1] = st.kept[i] end
    loom.run_subgraph_and_retain('flow_encoder', {n_codes = #tokens, n_past = 0},
        {tokens = tokens, embedding = embedding})
    local n_frames = TOKEN_MEL_RATIO * (#tokens - (finalize and 0 or LOOKAHEAD_TOKENS))
    local mu_all = loom.get_output('flow_encoder', 1)
    local mu = {}
    for i = 1, n_frames * N_MEL do mu[i] = mu_all[i] end
    -- The voice's own mel in the grid's first rows; the ODE in-fills the rest.
    local cond = chatterbox_zeros(n_frames * N_MEL)
    for i = 1, #prompt_feat do cond[i] = prompt_feat[i] end
    chatterbox_draw_frames(st, n_frames, #prompt_feat / N_MEL)
    local source = loom.state('noise') or st.flow_noise
    if source[n_frames * N_MEL] == nil then error('chatterbox: inputs.noise is shorter than the mel grid') end
    local noise = {}
    for i = 1, n_frames * N_MEL do noise[i] = source[i] end
    -- Euler over S3Gen's causal U-Net on the cosine schedule, under classifier-free guidance: ONE estimator
    -- run per step over `solve_euler`'s batch of two (the unconditional half sees mu, spks and cond zeroed),
    -- combined in the graph with `cfg` as the guidance weight (`EstimatorPhase`).
    loom.run_ode_and_retain('estimator', {n_tokens = n_frames, n_past = 0},
        {mu = mu, spks = {from = 'flow_encoder', index = 2}, cond = cond, cfg = {loom.state('flow_cfg')}},
        {carried = 'x', time = 't', method = 'euler', times = chatterbox_cosine_times(loom.state('n_steps')),
         n_elems = n_frames * N_MEL, state = noise})
    local grid = loom.get_output('estimator', 1)
    local mel = {}
    for i = #prompt_feat + TOKEN_MEL_RATIO * offset * N_MEL + 1, n_frames * N_MEL do mel[#mel + 1] = grid[i] end
    return mel
end

-- HiFT over every generated frame so far; returns the samples past `st.emitted` that no later frame can
-- change: all of them on the last hop, else all but the last VOCODER_RIGHT_FRAMES frames'.
local function chatterbox_vocode(st, finalize)
    local m = #st.mel / N_MEL
    local n = m * SAMPLES_PER_FRAME
    -- The sine source's per-sample draw, harmonic-major for `m` frames: the caller's, sliced, or the
    -- stream's own per-frame blocks laid out that way.
    local nsf = {}
    local pinned = loom.state('nsf_noise')
    if pinned ~= nil then
        local total = #pinned / NSF_HARMONICS
        if total < n then error('chatterbox: inputs.nsf_noise is shorter than the waveform') end
        for h = 0, NSF_HARMONICS - 1 do
            for s = 1, n do nsf[h * n + s] = pinned[h * total + s] end
        end
    else
        local per_frame = NSF_HARMONICS * SAMPLES_PER_FRAME
        local blocks = st.nsf_blocks
        for f = 0, m - 1 do
            for h = 0, NSF_HARMONICS - 1 do
                local src, dst = f * per_frame + h * SAMPLES_PER_FRAME, h * n + f * SAMPLES_PER_FRAME
                for s = 1, SAMPLES_PER_FRAME do nsf[dst + s] = blocks[src + s] end
            end
        end
    end
    local wave = loom.run_subgraph('vocoder', {n_enc_frames = m, n_past = 0},
        {mel = st.mel, nsf_phase = loom.state('nsf_phase'), nsf_noise = nsf})
    local stop = finalize and #wave or math.max(st.emitted, (m - VOCODER_RIGHT_FRAMES) * SAMPLES_PER_FRAME)
    local out = {}
    for i = st.emitted + 1, stop do out[#out + 1] = wave[i] end
    st.emitted = stop
    return out
end

-- `inputs` is `infer`'s: `tokens` (the text's BPE ids, no framing -- the start/stop ids are added here, as
-- `ChatterboxTTS.generate` adds them after tokenizing), and optionally the voice, the sampling knobs, the
-- draws, `return_mel` (each piece's mel instead of samples) and `piece_tokens` (the hop: unset or -1 grows
-- from STREAM_HOP_TOKENS, 0 is the whole turn in one piece).
function stream_open(inputs)
    if inputs.seed then loom.seed_rng(inputs.seed) end
    if inputs.tokens == nil or #inputs.tokens == 0 then error('chatterbox: no text ids') end
    local text = {START_TEXT}
    for i = 1, #inputs.tokens do text[#text + 1] = inputs.tokens[i] end
    text[#text + 1] = STOP_TEXT
    loom.set_state('text', text)
    for _, key in ipairs({'speaker_emb', 'cond_prompt_tokens', 'prompt_token', 'prompt_feat', 'embedding',
                          'noise', 'nsf_noise', 'return_mel'}) do
        chatterbox_keep(key, inputs[key])
    end
    loom.set_state('exaggeration', inputs.exaggeration or DEFAULT_EXAGGERATION)
    loom.set_state('cfg_weight', inputs.cfg_weight or DEFAULT_CFG_WEIGHT)
    loom.set_state('temperature', inputs.temperature or DEFAULT_TEMPERATURE)
    loom.set_state('min_p', inputs.min_p or DEFAULT_MIN_P)
    loom.set_state('top_p', inputs.top_p or DEFAULT_TOP_P)
    loom.set_state('repetition_penalty', inputs.repetition_penalty or DEFAULT_REPETITION_PENALTY)
    -- `n_steps`, the name every host passes a step count under (the contract's `tts.default_steps`).
    loom.set_state('n_steps', inputs.n_steps or FLOW_STEPS)
    loom.set_state('flow_cfg', inputs.flow_cfg_rate or FLOW_CFG_RATE)
    -- -1 is the grown hop too, so `infer` can run the live cut in one call (Soprano's convention).
    local piece = inputs.piece_tokens
    if piece == -1 then piece = nil end
    if piece ~= nil and piece < 0 then error('chatterbox: piece_tokens is ' .. piece .. '; -1, 0 or more') end
    -- The draws, in the order the top of this file gives: T3's block, then the harmonic phases (the
    -- fundamental's pinned to 0, the rest uniform on [-pi, pi), `SineGen.forward`'s). A pinned one is not
    -- drawn.
    local prefill = N_COND_ROWS + #text + N_BOS_ROWS
    local budget = math.min(inputs.max_new_tokens or DEFAULT_MAX_NEW_TOKENS, T3_MAX_POSITIONS - prefill)
    if budget <= 0 then
        error('chatterbox: the prefill (' .. prefill .. ' rows) leaves no room in the ' .. T3_MAX_POSITIONS ..
              '-position cache')
    end
    local nsf_phase = inputs.nsf_phase
    local draws = loom.uniform_array(budget)
    if nsf_phase == nil then
        local u = loom.uniform_array(NSF_HARMONICS)
        nsf_phase = {0.0}
        for h = 2, NSF_HARMONICS do nsf_phase[h] = (2.0 * u[h] - 1.0) * math.pi end
    end
    loom.set_state('nsf_phase', nsf_phase)
    chatterbox_save({step = 0, n_past = 0, budget = budget, draws = draws, history = {START_SPEECH}, kept = {},
                     ended = 0, offset = 0, hop = piece or STREAM_HOP_TOKENS, mel = {}, flow_noise = {},
                     nsf_blocks = {}, frames_drawn = 0, emitted = 0})
    -- A fixed hop does not grow (nor is it capped); 0 never cuts.
    loom.set_state('hop_scale', (piece == nil) and STREAM_HOP_SCALE or 1)
    loom.set_state('whole', (piece == 0) and 1 or 0)
    loom.set_state('piece', 0)
    loom.set_state('n_pieces', 0)
    loom.set_state('done', 0)
    return {}
end

-- One hop: T3 drawn until the hop and its look-ahead are there (or it stops), the flow over every token so
-- far, the new mel appended, and the samples it settles (or, with `return_mel`, the hop's mel).
function stream_feed(inputs)
    if loom.state('done') ~= 0 then return {} end
    local st = chatterbox_load()
    local whole = loom.state('whole') ~= 0
    local hop = st.hop
    while st.ended == 0 and (whole or #st.kept - st.offset < hop + LOOKAHEAD_TOKENS) do
        chatterbox_t3_step(st)
    end
    if #st.kept == 0 then error('chatterbox: T3 produced no speech tokens (the first draw was the stop token)') end
    local finalize = whole or #st.kept - st.offset < hop + LOOKAHEAD_TOKENS
    local n_tokens = finalize and #st.kept or st.offset + hop + LOOKAHEAD_TOKENS
    local mel = chatterbox_flow(st, n_tokens, st.offset, finalize)
    for i = 1, #mel do st.mel[#st.mel + 1] = mel[i] end
    local out = (loom.state('return_mel') ~= nil) and mel or chatterbox_vocode(st, finalize)
    if finalize then
        loom.set_state('done', 1)
    else
        st.offset = st.offset + hop
        if loom.state('hop_scale') > 1 then st.hop = math.min(STREAM_MAX_HOP_TOKENS, hop * loom.state('hop_scale')) end
    end
    chatterbox_save(st)
    loom.set_state('piece', loom.state('piece') + 1)
    return out
end

function stream_finish(inputs)
    return {}
end
