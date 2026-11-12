-- ===== CosyVoice3, live (loom.cpp ADR-112): upstream's `tts(stream=True)`, a hop of tokens per feed. =====
--
-- Upstream streams by RE-RUNNING, not by carrying: each hop it runs the flow over the voice's prompt and
-- every token drawn so far, in the DiT's STREAMING mode (a block-causal mask of CHUNK_FRAMES frames, which
-- the checkpoint was trained with), keeps the mel past what it already has, and runs CausalHiFT over all
-- of the mel, handing out only the samples it had not. The cut is the token count, never the clock: the
-- first hop is STREAM_HOP_TOKENS plus whatever puts the prompt and it on a chunk boundary, then the hop
-- doubles up to STREAM_MAX_HOP_TOKENS, and every hop but the last waits for LOOKAHEAD_TOKENS more (the
-- flow's pre-lookahead conv reads them as context). This does the same, a hop per `stream_feed`.
--
-- **So a stream is NOT `infer`.** `infer` runs the flow in the non-streaming mode (full attention), as
-- upstream's non-streaming call does; the two mels differ by ~5 (dev box, a 9 s sentence). What a stream
-- joins to is `infer` with `streaming = 1`: at f64 upstream's re-run loop IS that one pass, mel for mel
-- (max|d| 0), because each hop ends on a chunk boundary, the position conv and the chunk mask are
-- causal, and the flow's noise is a fixed table sliced from the start. At f32 the ten guided Euler steps
-- amplify the re-run's different rounding to ~5e-3 on the mel, which CausalHiFT's sine phase (a cumsum
-- of F0 reaching ~1e6 rad) turns into O(1) sample differences: compare a stream's mel, not its wave.
-- The vocoder alone, streamed this way over one mel, is its one call within 1.1e-05.
--
-- The noise is the stream's own and kept: the flow's draw frame-major (a prefix of it serves every
-- re-run, as the reference's `rand_noise[:, :, :T]` does) and the sine source's per frame (9 x 480 a
-- frame); a caller pins both (`noise`, frame-major for the whole grid; `nsf_noise`, harmonic-major for the
-- whole wave) and the LM's draws (`draws`), which is how a stream is compared with a one pass at all. The
-- functions hold nothing between calls: the text's chunks, the knobs, the LM's loop (prompt rows, step,
-- `n_past`, RAS history, the silence run, the kept tokens), the hop, the mel so far and the samples
-- handed out are the stream's state; the LM's history is the model's KV cache, leased or owned (ADR-099).
-- Every step feeds the previous token BEFORE it samples, so no graph output is read across a call.

local function cosyvoice3_keep(name, value)
    if value ~= nil then loom.set_state(name, value) end
end

-- ---- The LM, one token per call: `infer` (01_lm.lua) and the stream run the same loop. ----

-- One chunk's prompt rows and budget (`Qwen2LM.inference`); runs nothing. The prefill is `[sos, prompt
-- text + text, task_id, prompt speech tokens]`, spelled as three parallel id arrays: a row is a SPEECH row
-- (sos, task_id, the prompt's tokens -- `speech_embedding`) where `mask` is 1 and a TEXT row
-- (`embed_tokens`) where it is 0. The unused id is 0.
function cosyvoice3_lm_begin(inputs, text)
    local prompt_text = cosyvoice3_voice(inputs.prompt_text, 'voice.prompt_text')
    local prompt_speech = cosyvoice3_voice(inputs.prompt_speech_tokens, 'voice.prompt_speech_tokens')
    local text_ids, speech_ids, mask = {0}, {SOS}, {1.0}
    local has_eop = false
    local function text_row(id)
        text_ids[#text_ids + 1], speech_ids[#speech_ids + 1], mask[#mask + 1] = id, 0, 0.0
        if id == END_OF_PROMPT then has_eop = true end
    end
    local function speech_row(id)
        text_ids[#text_ids + 1], speech_ids[#speech_ids + 1], mask[#mask + 1] = 0, id, 1.0
    end
    for i = 1, #prompt_text do text_row(prompt_text[i]) end
    for i = 1, #text do text_row(text[i]) end
    if not has_eop then
        -- `Qwen2LM.inference` asserts it; without it the LM has no boundary between instruction and text.
        error('cosyvoice3: no <|endofprompt|> (151646) in prompt text + text -- a voice\'s prompt text carries it')
    end
    speech_row(TASK_ID)
    for i = 1, #prompt_speech do speech_row(prompt_speech[i]) end
    local prefill = #text_ids
    -- The token budget is the TEXT's (not the prompt's): `min_len`/`max_len` in `Qwen2LM.inference`.
    local max_len = math.min(math.floor(#text * (inputs.max_token_text_ratio or MAX_TOKEN_TEXT_RATIO)),
                             LM_MAX_POSITIONS - prefill)
    if max_len <= 0 then
        error('cosyvoice3: the prefill (' .. prefill .. ' rows) leaves no room in the ' ..
              LM_MAX_POSITIONS .. '-position cache')
    end
    return {lm_text_ids = text_ids, lm_speech_ids = speech_ids, lm_mask = mask,
            lm_min_len = math.floor(#text * MIN_TOKEN_TEXT_RATIO), lm_max_len = max_len,
            lm_n_past = 0, lm_step = 0, lm_history = {}, lm_silent_run = 0, lm_kept = {}, lm_ended = 0}
end

-- One step: the LM brought up to the previous token (the prefill on the first step), then one draw under
-- `ras_sampling` -- a nucleus draw (top-k 25, top-p 0.8 over the WHOLE softmax); if that id is among the
-- last RAS_WIN drawn at least RAS_WIN * RAS_TAU times, a second draw from the full distribution with it
-- banned. Before `min_len` tokens the id `speech_token_size` (6561) is banned from both draws --
-- `sampling_ids`' `ignore_eos`, which bans that one id and not the other 199 stop ids. `draws` pins both
-- uniforms per step (two per step, used or not); absent, the engine's stream draws. Sets `lm_ended` at a
-- stop id or the budget; the token it keeps, if any, is appended to `lm_kept`.
function cosyvoice3_lm_step(st, draws)
    if st.lm_ended ~= 0 then return end
    if st.lm_n_past == 0 then
        local n = #st.lm_text_ids
        loom.run_subgraph_and_retain('lm', {n_tokens = n, n_past = 0},
            {text_ids = st.lm_text_ids, speech_ids = st.lm_speech_ids, speech_mask = st.lm_mask,
             position_ids = loom.range(0, n), attention_mask = loom.causal_mask(n, 0)})
        st.lm_n_past = n
    else
        local n_past = st.lm_n_past
        loom.run_subgraph_and_retain('lm', {n_tokens = 1, n_past = n_past},
            {text_ids = {0}, speech_ids = {st.lm_history[#st.lm_history]}, speech_mask = {1.0},
             position_ids = {n_past}, attention_mask = loom.causal_mask(1, n_past)})
        st.lm_n_past = n_past + 1
    end
    local step = st.lm_step
    st.lm_step = step + 1
    local banned = {}
    if step < st.lm_min_len then banned[1] = SOS end
    local u1, u2
    if draws then u1, u2 = draws[2 * step + 1], draws[2 * step + 2] end
    local token = loom.sample_row('lm', -1, {temperature = 1.0, top_k = RAS_TOP_K, top_p = RAS_TOP_P,
                                             top_p_mass = 'row', banned = banned, uniform = u1})
    local history = st.lm_history
    local rep = 0
    for j = math.max(1, #history - RAS_WIN + 1), #history do
        if history[j] == token then rep = rep + 1 end
    end
    if rep >= RAS_WIN * RAS_TAU then
        banned[#banned + 1] = token
        token = loom.sample_row('lm', -1, {temperature = 1.0, banned = banned, uniform = u2})
    end
    if token >= SOS then
        st.lm_ended = 1
        return
    end
    history[#history + 1] = token
    -- `llm_job`'s silence cap is on what reaches the FLOW; the LM is fed every token back.
    if COSYVOICE3_SILENT[token] then st.lm_silent_run = st.lm_silent_run + 1 else st.lm_silent_run = 0 end
    if st.lm_silent_run <= MAX_SILENT_RUN then st.lm_kept[#st.lm_kept + 1] = token end
    if st.lm_step >= st.lm_max_len then st.lm_ended = 1 end
end

-- ---- The flow and the vocoder over what a stream has so far. ----

-- Per frame of a grid: the DiT's chunk index in streaming mode, else 0 (one chunk: full attention).
function cosyvoice3_chunk_ids(n_frames, streaming)
    local ids = {}
    for i = 1, n_frames do ids[i] = streaming and math.floor((i - 1) / CHUNK_FRAMES) or 0 end
    return ids
end

-- What a stream keeps across feeds, besides the knobs it is opened with.
local COSYVOICE3_LOOP = {'lm_text_ids', 'lm_speech_ids', 'lm_mask', 'lm_min_len', 'lm_max_len', 'lm_n_past',
                         'lm_step', 'lm_history', 'lm_silent_run', 'lm_kept', 'lm_ended',
                         'chunk', 'offset', 'hop', 'mel', 'flow_noise', 'nsf_blocks', 'emitted'}

-- The ones that start empty: an empty array is not something to rely on reading back as one.
local COSYVOICE3_ARRAYS = {lm_history = true, lm_kept = true, mel = true, flow_noise = true, nsf_blocks = true}

local function cosyvoice3_load()
    local st = {}
    for _, name in ipairs(COSYVOICE3_LOOP) do
        st[name] = loom.state(name)
        if st[name] == nil and COSYVOICE3_ARRAYS[name] then st[name] = {} end
    end
    return st
end

local function cosyvoice3_save(st)
    for _, name in ipairs(COSYVOICE3_LOOP) do cosyvoice3_keep(name, st[name]) end
end

-- The flow over the voice's prompt tokens and the first `n_tokens` kept, in streaming mode; returns the
-- mel of the generated frames from token `offset` to the end of the grid, frame-major. `finalize` is the
-- last hop: every token, the encoder's zero look-ahead pad at the end, as the reference's `finalize=True`.
-- Otherwise the last LOOKAHEAD_TOKENS tokens are the look-ahead conv's context only: the encoder runs
-- over all of them and its rows past the hop are dropped, which is what `PreLookaheadLayer(x, context)`
-- computes (its first conv reads tokens i .. i + 3, its second is causal).
local function cosyvoice3_flow(st, n_tokens, offset, finalize)
    local prompt_tokens = cosyvoice3_voice(loom.state('prompt_speech_tokens'), 'voice.prompt_speech_tokens')
    local prompt_feat = cosyvoice3_voice(loom.state('prompt_feat'), 'voice.prompt_feat')
    local embedding = cosyvoice3_voice(loom.state('embedding'), 'voice.embedding')
    local tokens = {}
    for i = 1, #prompt_tokens do tokens[i] = prompt_tokens[i] end
    for i = 1, n_tokens do tokens[#tokens + 1] = st.lm_kept[i] end
    loom.run_subgraph_and_retain('flow_encoder', {n_codes = #tokens, n_past = 0},
        {tokens = tokens, embedding = embedding})
    local n_codes = #tokens - (finalize and 0 or LOOKAHEAD_TOKENS)
    local n_frames = TOKEN_MEL_RATIO * n_codes
    local mu_all = loom.get_output('flow_encoder', 1)
    local mu = {}
    for i = 1, n_frames * N_MEL do mu[i] = mu_all[i] end
    local cond = cosyvoice3_zeros(n_frames * N_MEL)
    for i = 1, #prompt_feat do cond[i] = prompt_feat[i] end
    -- The grid's noise: the caller's, or the stream's own, drawn frame-major as the grid first reaches
    -- each frame and kept, so every re-run starts from the same draw.
    local pinned = loom.state('noise')
    local noise = {}
    if pinned ~= nil then
        if pinned[n_frames * N_MEL] == nil then error('cosyvoice3: inputs.noise is shorter than the mel grid') end
        for i = 1, n_frames * N_MEL do noise[i] = pinned[i] end
    else
        local kept = st.flow_noise
        if #kept < n_frames * N_MEL then
            local more = loom.gaussian_array(n_frames * N_MEL - #kept)
            for i = 1, #more do kept[#kept + 1] = more[i] end
        end
        for i = 1, n_frames * N_MEL do noise[i] = kept[i] end
    end
    local ids = cosyvoice3_chunk_ids(n_frames, true)
    loom.run_ode_and_retain('estimator', {n_tokens = n_frames, n_past = 0},
        {mu = mu, spks = {from = 'flow_encoder', index = 2}, cond = cond, chunk_ids = ids,
         cfg = {loom.state('flow_cfg')}},
        {carried = 'x', time = 't', method = 'euler', times = cosyvoice3_cosine_times(loom.state('n_steps')),
         n_elems = n_frames * N_MEL, state = noise})
    local grid = loom.get_output('estimator', 1)
    local from = #prompt_feat + TOKEN_MEL_RATIO * offset * N_MEL
    local mel = {}
    for i = from + 1, n_frames * N_MEL do mel[#mel + 1] = grid[i] end
    return mel
end

-- CausalHiFT over every generated frame so far; returns the samples past `st.emitted` that no later frame
-- can change -- all of them on the last hop, else all but the last VOCODER_RIGHT_FRAMES frames' (the F0
-- predictor's and `conv_pre`'s look-right, and the iSTFT's last hop), which is what the reference's
-- `finalize=False` withholds.
local function cosyvoice3_vocode(st, finalize)
    local m = #st.mel / N_MEL
    local per_frame = NSF_HARMONICS * SAMPLES_PER_FRAME
    local n = m * SAMPLES_PER_FRAME
    -- The sine source's draw, harmonic-major for `m` frames: the caller's, sliced, or the stream's own,
    -- kept a frame at a time (harmonic-major within the frame) so a re-run reads the same values.
    local nsf = {}
    local pinned = loom.state('nsf_noise')
    if pinned ~= nil then
        local total = #pinned / NSF_HARMONICS
        if total < n then error('cosyvoice3: inputs.nsf_noise is shorter than the waveform') end
        for h = 0, NSF_HARMONICS - 1 do
            for s = 1, n do nsf[h * n + s] = pinned[h * total + s] end
        end
    else
        local blocks = st.nsf_blocks
        if #blocks < m * per_frame then
            local more = loom.uniform_array(m * per_frame - #blocks)
            for i = 1, #more do blocks[#blocks + 1] = more[i] end
        end
        for f = 0, m - 1 do
            for h = 0, NSF_HARMONICS - 1 do
                local src, dst = f * per_frame + h * SAMPLES_PER_FRAME, h * n + f * SAMPLES_PER_FRAME
                for s = 1, SAMPLES_PER_FRAME do nsf[dst + s] = blocks[src + s] end
            end
        end
    end
    local wave = loom.run_subgraph('vocoder', {n_enc_frames = m, n_past = 0}, {mel = st.mel, nsf_noise = nsf})
    local stop = finalize and #wave or math.max(st.emitted, (m - VOCODER_RIGHT_FRAMES) * SAMPLES_PER_FRAME)
    local out = {}
    for i = st.emitted + 1, stop do out[#out + 1] = wave[i] end
    st.emitted = stop
    return out
end

-- A chunk from the top: its LM prompt, and no tokens, mel or noise yet.
local function cosyvoice3_open_chunk(st, c)
    local ends, all = loom.state('chunk_ends'), loom.state('chunk_ids')
    local text = {}
    for i = (c == 1 and 0 or ends[c - 1]) + 1, ends[c] do text[#text + 1] = all[i] end
    local lm = cosyvoice3_lm_begin({prompt_text = loom.state('prompt_text'),
                                    prompt_speech_tokens = loom.state('prompt_speech_tokens'),
                                    max_token_text_ratio = loom.state('max_token_text_ratio')}, text)
    for k, v in pairs(lm) do st[k] = v end
    st.chunk, st.offset, st.hop, st.mel, st.flow_noise, st.nsf_blocks, st.emitted = c, 0, STREAM_HOP_TOKENS, {}, {}, {}, 0
end

function stream_open(inputs)
    if inputs.seed then loom.seed_rng(inputs.seed) end
    -- `loom::CosyVoice3Vocab::encode` opens every chunk with CHUNK_HEADER; ids with no header are one chunk.
    local chunks = {inputs.tokens}
    if inputs.tokens[1] == CHUNK_HEADER then chunks = cosyvoice3_split_chunks(inputs.tokens, CHUNK_HEADER) end
    if #chunks == 0 or #chunks[1] == 0 then error('cosyvoice3: no text ids') end
    if #chunks > 1 then
        for _, key in ipairs({'draws', 'noise', 'nsf_noise', 'return_mel'}) do
            if inputs[key] ~= nil then
                error('cosyvoice3: inputs.' .. key .. ' pins one generation, and this text is ' .. #chunks ..
                      ' chunks -- pass one sentence, or leave it out')
            end
        end
    end
    local ids, ends = {}, {}
    for c, chunk in ipairs(chunks) do
        for _, id in ipairs(chunk) do ids[#ids + 1] = id end
        ends[c] = #ids
    end
    loom.set_state('chunk_ids', ids)
    loom.set_state('chunk_ends', ends)
    for _, key in ipairs({'prompt_text', 'prompt_speech_tokens', 'prompt_feat', 'embedding', 'draws', 'noise',
                          'nsf_noise', 'max_token_text_ratio', 'return_mel'}) do
        cosyvoice3_keep(key, inputs[key])
    end
    loom.set_state('n_steps', inputs.n_steps or FLOW_STEPS)
    loom.set_state('flow_cfg', inputs.flow_cfg_rate or FLOW_CFG_RATE)
    local st = {}
    cosyvoice3_open_chunk(st, 1)
    cosyvoice3_save(st)
    loom.set_state('piece', 0)
    loom.set_state('n_pieces', 0)
    loom.set_state('done', 0)
    return {}
end

-- One hop: tokens drawn until the hop and its look-ahead are there (or the LM stops), the flow over all
-- of them, the new mel appended, and the samples it settles. `return_mel` hands back the hop's mel
-- instead: the stream's own flow output, which is what an oracle compares (see the top).
function stream_feed(inputs)
    if loom.state('done') ~= 0 then return {} end
    local st = cosyvoice3_load()
    local draws = loom.state('draws')
    local n_prompt = #cosyvoice3_voice(loom.state('prompt_speech_tokens'), 'voice.prompt_speech_tokens')
    -- `prompt_token_pad`: the first hop ends the prompt and itself on a chunk boundary.
    local hop = st.hop
    if st.offset == 0 then
        hop = hop + math.ceil(n_prompt / STREAM_HOP_TOKENS) * STREAM_HOP_TOKENS - n_prompt
    end
    while st.lm_ended == 0 and #st.lm_kept - st.offset < hop + LOOKAHEAD_TOKENS do
        cosyvoice3_lm_step(st, draws)
    end
    local finalize = #st.lm_kept - st.offset < hop + LOOKAHEAD_TOKENS
    if finalize and #st.lm_kept == 0 then error('cosyvoice3: the LM produced no speech tokens (the first draw was a stop id)') end
    local n_tokens = finalize and #st.lm_kept or st.offset + hop + LOOKAHEAD_TOKENS
    local mel = cosyvoice3_flow(st, n_tokens, st.offset, finalize)
    for i = 1, #mel do st.mel[#st.mel + 1] = mel[i] end
    local out = (loom.state('return_mel') ~= nil) and mel or cosyvoice3_vocode(st, finalize)
    if finalize then
        if st.chunk == #loom.state('chunk_ends') then
            loom.set_state('done', 1)
        else
            cosyvoice3_open_chunk(st, st.chunk + 1)
        end
    else
        st.offset = st.offset + hop
        st.hop = math.min(STREAM_MAX_HOP_TOKENS, st.hop * STREAM_HOP_SCALE)
    end
    cosyvoice3_save(st)
    loom.set_state('piece', loom.state('piece') + 1)
    return out
end

function stream_finish(inputs)
    return {}
end
