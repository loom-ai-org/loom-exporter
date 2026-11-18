-- ===== Paradee: phoneme ids in, 24 kHz out, one sentence per pass (loom-exporter paradee_export.py). =====
--
-- Input `tokens`: Kokoro-vocabulary ids, each sentence wrapped in a 0 at both ends (the text door's
-- phoneme table does that; a bare run is wrapped here). Upstream reads a text one SENTENCE at a time --
-- it cuts after `.`, `!`, `?` or `…` followed by whitespace, then phonemizes each part -- so the ids are cut
-- at the same place: a sentence mark followed by a space. Options: `voice` (0-based row of the file's
-- `loom.tts.voices`, default 0 = the first), `speed` (> 1 is faster), `seed` (sentence i draws its noise
-- with seed + i), `noise` (pinned draws, every sentence's end to end, for a reference comparison).
--
-- Per sentence: Kokoro's text side and prosody, exactly the calls of Kokoro's driver
-- (`kokoro_driver/02_duration_encoder.lua`, `03_frame_expansion.lua`, `04_f0n.lua`), with `asr_proj`
-- between the text encoder's BiLSTM and the frame expansion -- then the F0 curve read back for the sine
-- bank's phase, and `synth`. Inlined rather than reused as fragments because a fragment is a straight-line
-- block of the driver body and this runs once per sentence.

local PARADEE_SENTENCE_END = {[4] = true, [5] = true, [6] = true, [10] = true}   -- . ! ? …
local PARADEE_SPACE = 16
-- ALBERT has 512 positions (upstream's MAX_PHONEMES = 510 plus the two 0s).
local PARADEE_MAX_IDS = 512

local function paradee_sentences(ids)
    local out, current = {}, {}
    local function close()
        local any = false
        for _, id in ipairs(current) do if id ~= PARADEE_SPACE then any = true end end
        if any then
            local u = {0}
            for _, id in ipairs(current) do u[#u + 1] = id end
            u[#u + 1] = 0
            if #u > PARADEE_MAX_IDS then
                error(('paradee: a sentence of %d phonemes; the model reads at most %d -- split it'):format(
                    #u - 2, PARADEE_MAX_IDS - 2))
            end
            out[#out + 1] = u
        end
        current = {}
    end
    for k, raw in ipairs(ids) do
        local id = math.floor(raw + 0.5)
        if id == 0 then
            close()
        elseif id == PARADEE_SPACE and #current > 0 and PARADEE_SENTENCE_END[current[#current]] then
            close()
        else
            current[#current + 1] = id
        end
    end
    close()
    return out
end

-- The sine bank's running phase (`vhead.sine_bank`), in doubles as upstream's float64: frame j advances
-- f0[j] * FRAME_HOP / SAMPLE_RATE cycles, the running sum is kept modulo 1, and each frame also carries
-- the advance to the next frame's centre (none after the last). Row 0 is frame 0 again with no advance:
-- upstream clamps the first half frame's interpolation index to 0. Flat [frames + 1, 2], row-major.
local function paradee_phase(f0)
    local phase = {f0[1] * FRAME_HOP / SAMPLE_RATE % 1.0, 0.0}
    local running = 0.0
    for j = 1, #f0 do
        running = running + f0[j] * FRAME_HOP / SAMPLE_RATE
        local nxt = 0.0
        if j < #f0 then nxt = f0[j + 1] * FRAME_HOP / SAMPLE_RATE end
        phase[2 * j + 1] = running % 1.0
        phase[2 * j + 2] = nxt
    end
    return phase
end

local function paradee_speak(ids, s_predictor, s_decoder, speed, seed, pinned)
    local T_text = #ids
    loom.run_subgraph_and_retain("albert_bert_encoder", {n_tokens = T_text, n_past = 0}, {tokens = ids})

    -- DurationEncoder: the style concatenated, 3x (BiLSTM, AdaLayerNorm, concatenation), as Kokoro's.
    local d_channels = D_MODEL + STYLE_DIM
    loom.run_subgraph_and_retain("duration_style_concat", {n_tokens = T_text, n_past = 0},
                                  {x = {from = 'albert_bert_encoder'}, style = s_predictor})
    local d = "duration_style_concat"
    for i = 0, 2 do
        local lstm = run_bi_lstm("duration_lstm_" .. i, {from = d}, T_text, d_channels, HIDDEN_PER_DIR)
        loom.run_subgraph_and_retain("duration_adaln_" .. i, {n_tokens = T_text, n_past = 0},
                                      {x = {from = lstm}, style = s_predictor})
        loom.run_subgraph_and_retain("duration_style_concat", {n_tokens = T_text, n_past = 0},
                                      {x = {from = "duration_adaln_" .. i}, style = s_predictor})
        d = "duration_style_concat"
    end
    local top = run_bi_lstm("top_lstm", {from = d}, T_text, d_channels, HIDDEN_PER_DIR)
    local duration_logits = loom.run_subgraph("duration_proj", {n_tokens = T_text, n_past = 0},
                                               {x = {from = top}})
    local pred_dur = predict_durations(duration_logits, T_text, speed)
    local T_frames = array_sum(pred_dur)

    -- The prosody states and the text encoder's output, repeated in place by the durations; `asr_proj`
    -- runs per token first (a Linear-GELU-Linear commutes with a repeat).
    loom.expand_by_duration_and_retain(d, pred_dur)
    loom.run_subgraph_and_retain("text_encoder_cnn", {n_tokens = T_text, n_past = 0}, {tokens = ids})
    local te_channels = loom.output_shape("text_encoder_cnn", 1)[1]
    local asr_rows = run_bi_lstm("text_encoder_lstm", {from = "text_encoder_cnn"}, T_text, te_channels,
                                  HIDDEN_PER_DIR)
    loom.run_subgraph_and_retain("asr_proj", {n_tokens = T_text, n_past = 0}, {x = {from = asr_rows}})
    loom.expand_by_duration_and_retain("asr_proj", pred_dur, "layout_a")

    -- F0Ntrain, as Kokoro's.
    local shared_out = run_bi_lstm("f0n_shared_lstm", {from = d}, T_frames, D_MODEL + STYLE_DIM,
                                    HIDDEN_PER_DIR, "layout_a")
    local f0_feat = run_resblk_stack("f0n_f0", shared_out, s_predictor)
    local n_feat = run_resblk_stack("f0n_n", shared_out, s_predictor)
    local F0_curve = run_proj1x1("f0n_f0_proj", f0_feat)
    local N_curve = run_proj1x1("f0n_n_proj", n_feat)

    local f0 = loom.get_output(F0_curve)
    local samples = FRAME_HOP * #f0
    local noise
    if pinned ~= nil then
        if #pinned < samples then error('paradee: fewer pinned noise values than the sentence needs') end
        noise = {}
        for k = 1, samples do noise[k] = pinned[k] end
    else
        noise = aten_randn(seed % 4294967296, samples)
    end
    -- The 3 ms noise path shapes the same draw reversed (upstream's `nz.flip(-1)`).
    local noise_rev = {}
    for k = 1, samples do noise_rev[k] = noise[samples + 1 - k] end
    local wave = loom.run_subgraph("synth", {n_enc_frames = T_frames, n_past = 0},
        {asr = {from = "asr_proj"}, f0 = {from = F0_curve}, n = {from = N_curve}, style = s_decoder,
         phase = paradee_phase(f0), noise = noise, noise_rev = noise_rev})
    return wave, samples
end

-- The voice's two style rows, from the tables the file carries.
local function paradee_voice(voice)
    local v = math.floor((voice or 0) + 0.5)
    if v < 0 or v >= N_VOICES then
        error(('paradee: voice %d; this file has %d (0 to %d, loom.tts.voices)'):format(v, N_VOICES, N_VOICES - 1))
    end
    local text = loom.get_weight("text_encoder_cnn", "loom.paradee.text_style")
    local dec = loom.get_weight("text_encoder_cnn", "loom.paradee.decoder_style")
    return array_slice(text, v * STYLE_DIM + 1, STYLE_DIM), array_slice(dec, v * DECODER_STYLE_DIM + 1, DECODER_STYLE_DIM)
end

-- ===== The live door: one sentence per `stream_feed`. =====
function stream_open(inputs)
    if inputs.tokens == nil or #inputs.tokens == 0 then error('paradee: no phoneme ids') end
    local sentences = paradee_sentences(inputs.tokens)
    local flat, offsets = {}, {0}
    for _, u in ipairs(sentences) do
        for k = 1, #u do flat[#flat + 1] = u[k] end
        offsets[#offsets + 1] = #flat
    end
    local s_predictor, s_decoder = paradee_voice(inputs.voice)
    loom.set_state('ids', flat)
    loom.set_state('offsets', offsets)
    loom.set_state('s_predictor', s_predictor)
    loom.set_state('s_decoder', s_decoder)
    loom.set_state('speed', inputs.speed or 1.0)
    loom.set_state('seed', inputs.seed or 0)
    if inputs.noise ~= nil then loom.set_state('noise', inputs.noise) end
    loom.set_state('noise_pos', 0)
    loom.set_state('next', 0)
    loom.set_state('done', #sentences == 0 and 1 or 0)
    return {}
end

local function paradee_next()
    local i = loom.state('next')
    local offsets, ids = loom.state('offsets'), loom.state('ids')
    local u = {}
    for k = offsets[i + 1] + 1, offsets[i + 2] do u[#u + 1] = ids[k] end
    local pinned = loom.state('noise')
    local rest = nil
    if pinned ~= nil then
        rest = {}
        for k = loom.state('noise_pos') + 1, #pinned do rest[#rest + 1] = pinned[k] end
    end
    local wave, samples = paradee_speak(u, loom.state('s_predictor'), loom.state('s_decoder'),
                                        loom.state('speed'), loom.state('seed') + i, rest)
    if pinned ~= nil then loom.set_state('noise_pos', loom.state('noise_pos') + samples) end
    loom.set_state('next', i + 1)
    if i + 2 >= #offsets then loom.set_state('done', 1) end
    return wave
end

function stream_feed(inputs)
    if loom.state('done') ~= 0 and loom.state('next') >= #loom.state('offsets') - 1 then return {} end
    return paradee_next()
end

function stream_finish(inputs)
    return {}
end
