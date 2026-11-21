-- ===== kale-tts-1-tiny: text in (the file's own front end), 24 kHz out, one sentence per pass
-- (loom-exporter kale_tts_export.py). =====
--
-- Input `tokens`: what the file's tokenizer (kind `kale_tts`) writes for a text -- the record count, then one
-- record per sentence (a sentence past MAX_IDS - 2 phonemes is cut at a space into several, each with the whole
-- sentence's words, as upstream's runtime does):
--
--   n, ids[n], widx[n], n_words, n_sub, sub[n_sub], sub_word[n_sub]
--
-- `ids` are Kokoro-vocabulary phoneme ids without the 0 pads, `widx` each one's word (0-based), `sub` the Ettin
-- sub-tokens of the previous sentence and this one (`[CLS] ... [SEP]`), and `sub_word` each sub-token's word in
-- this sentence or -1 (the previous sentence, the two specials). Options: `voice` (0-based row of the file's
-- `loom.tts.voices`, default 0), `speed` (> 1 is faster), `seed` (sentence i draws its noise with seed + i),
-- `noise` (pinned draws, every sentence's end to end, for a reference comparison).
--
-- Per sentence: Ettin, the text side (K20Text), then Kokoro's prosody exactly as Paradee's driver runs it,
-- the end clamp on the F0 read back, the text encoder, and Paradee's synth.

local function kale_records(tokens)
    local at = 1
    local function take()
        local v = tokens[at]
        if v == nil then error('kale: the token stream ends inside a record (it is the kale_tts tokenizer\'s layout)') end
        at = at + 1
        return math.floor(v + 0.5)
    end
    local out = {}
    local count = take()
    for _ = 1, count do
        local r = {ids = {}, widx = {}, sub = {}, sub_word = {}}
        local n = take()
        if n < 1 or n > MAX_IDS - 2 then
            error(('kale: a record of %d phonemes; the model reads 1 to %d'):format(n, MAX_IDS - 2))
        end
        for k = 1, n do r.ids[k] = take() end
        for k = 1, n do r.widx[k] = take() end
        r.n_words = take()
        local s = take()
        for k = 1, s do r.sub[k] = take() end
        for k = 1, s do r.sub_word[k] = take() end
        out[#out + 1] = r
    end
    if at <= #tokens then error('kale: values after the last record') end
    return out
end

-- `wavg[widx]`, [n + 2, n_sub] row-major: each phoneme's row averages its word's sub-tokens (upstream's
-- `ettinInputs`), the two pads taking the first and the last phoneme's word.
local function kale_wtok(r)
    local W = math.max(r.n_words, 1)
    local S = #r.sub
    local count = {}
    for w = 0, W - 1 do count[w] = 0 end
    for k = 1, S do
        local w = r.sub_word[k]
        if w >= 0 then count[w] = count[w] + 1 end
    end
    local rows = {r.widx[1]}
    for k = 1, #r.widx do rows[#rows + 1] = r.widx[k] end
    rows[#rows + 1] = r.widx[#r.widx]
    local out = {}
    for _, w in ipairs(rows) do
        for k = 1, S do
            local v = 0.0
            if r.sub_word[k] == w and count[w] > 0 then v = 1.0 / count[w] end
            out[#out + 1] = v
        end
    end
    return out
end

-- Upstream's end clamp (`textside.clamp_onnx` with the release's per-voice inputs), then the unvoicing floor:
-- in the last `tail` frames up to the last voiced one (F0 > thr), with at least 5 voiced, voiced F0 is held at
-- or above `floor`; every frame under `unvoice` becomes 0.
local function kale_clamp(f0, c)
    local floor, tail, thr, unvoice = c[1], c[2], c[3], c[4]
    local count, last = 0, 0
    for j = 1, #f0 do
        if f0[j] > thr then count = count + 1; last = j - 1 end
    end
    local out = {}
    for j = 1, #f0 do
        local v = f0[j]
        local i = j - 1
        if count >= 5 and v > thr and i >= last - tail and i <= last and v < floor then v = floor end
        if v < unvoice then v = 0.0 end
        out[j] = v
    end
    return out
end

-- The sine bank's running phase in doubles, as Paradee's driver builds it (upstream's float64 cumsum).
local function kale_phase(f0)
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

local function kale_speak(r, v, speed, seed, pinned)
    local ids = {0}
    for k = 1, #r.ids do ids[#ids + 1] = r.ids[k] end
    ids[#ids + 1] = 0
    local T_text = #ids
    -- The voice's pack row for this sentence's phoneme count (`pack[min(n, 510) - 1]`), its prosody half.
    local row = math.min(#r.ids, PACK_ROWS) - 1
    local s_predictor = array_slice(v.packs, (v.index * PACK_ROWS + row) * STYLE_DIM + 1, STYLE_DIM)

    loom.run_subgraph_and_retain("ettin", {n_tokens = #r.sub, n_past = 0}, {tokens = r.sub})
    loom.run_subgraph_and_retain("text_side", {n_tokens = T_text, n_enc_frames = #r.sub, n_past = 0},
                                  {tokens = ids, wtok = kale_wtok(r), h = {from = 'ettin'}})

    local d_channels = D_MODEL + STYLE_DIM
    loom.run_subgraph_and_retain("duration_style_concat", {n_tokens = T_text, n_past = 0},
                                  {x = {from = 'text_side'}, style = s_predictor})
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

    loom.expand_by_duration_and_retain(d, pred_dur)
    loom.run_subgraph_and_retain("text_encoder_cnn", {n_tokens = T_text, n_past = 0}, {tokens = ids})
    local te_channels = loom.output_shape("text_encoder_cnn", 1)[1]
    local asr_rows = run_bi_lstm("text_encoder_lstm", {from = "text_encoder_cnn"}, T_text, te_channels,
                                  HIDDEN_PER_DIR)
    loom.expand_by_duration_and_retain(asr_rows, pred_dur, "layout_a")

    local shared_out = run_bi_lstm("f0n_shared_lstm", {from = d}, T_frames, D_MODEL + STYLE_DIM,
                                    HIDDEN_PER_DIR, "layout_a")
    local f0_feat = run_resblk_stack("f0n_f0", shared_out, s_predictor)
    local n_feat = run_resblk_stack("f0n_n", shared_out, s_predictor)
    local F0_curve = run_proj1x1("f0n_f0_proj", f0_feat)
    local N_curve = run_proj1x1("f0n_n_proj", n_feat)

    local f0 = kale_clamp(loom.get_output(F0_curve), v.clamp)
    local samples = FRAME_HOP * #f0
    local noise
    if pinned ~= nil then
        if #pinned < samples then error('kale: fewer pinned noise values than the sentence needs') end
        noise = {}
        for k = 1, samples do noise[k] = pinned[k] end
    else
        noise = aten_randn(seed % 4294967296, samples)
    end
    local noise_rev = {}
    for k = 1, samples do noise_rev[k] = noise[samples + 1 - k] end
    local wave = loom.run_subgraph("synth", {n_enc_frames = T_frames, n_past = 0},
        {asr = {from = asr_rows}, f0 = f0, n = {from = N_curve}, style = v.sound,
         phase = kale_phase(f0), noise = noise, noise_rev = noise_rev})
    return wave, samples
end

local function kale_voice(voice)
    local v = math.floor((voice or 0) + 0.5)
    if v < 0 or v >= N_VOICES then
        error(('kale: voice %d; this file has %d (0 to %d, loom.tts.voices)'):format(v, N_VOICES, N_VOICES - 1))
    end
    local sound = loom.get_weight("text_encoder_cnn", "loom.kale.voice_sound")
    local clamp = loom.get_weight("text_encoder_cnn", "loom.kale.voice_clamp")
    return {index = v, packs = loom.get_weight("text_encoder_cnn", "loom.kale.prosody_packs"),
            sound = array_slice(sound, v * DECODER_STYLE_DIM + 1, DECODER_STYLE_DIM),
            clamp = array_slice(clamp, v * N_CLAMP + 1, N_CLAMP)}
end

-- ===== The live door: one sentence per `stream_feed`. =====
function stream_open(inputs)
    if inputs.tokens == nil or #inputs.tokens == 0 then error('kale: no tokens') end
    local records = kale_records(inputs.tokens)
    loom.set_state('tokens', inputs.tokens)
    loom.set_state('n_records', #records)
    loom.set_state('voice', inputs.voice or 0)
    loom.set_state('speed', inputs.speed or 1.0)
    loom.set_state('seed', inputs.seed or 0)
    if inputs.noise ~= nil then loom.set_state('noise', inputs.noise) end
    loom.set_state('noise_pos', 0)
    loom.set_state('next', 0)
    loom.set_state('done', #records == 0 and 1 or 0)
    return {}
end

local function kale_next()
    local i = loom.state('next')
    local r = kale_records(loom.state('tokens'))[i + 1]
    local pinned = loom.state('noise')
    local rest = nil
    if pinned ~= nil then
        rest = {}
        for k = loom.state('noise_pos') + 1, #pinned do rest[#rest + 1] = pinned[k] end
    end
    local wave, samples = kale_speak(r, kale_voice(loom.state('voice')), loom.state('speed'),
                                     loom.state('seed') + i, rest)
    if pinned ~= nil then loom.set_state('noise_pos', loom.state('noise_pos') + samples) end
    loom.set_state('next', i + 1)
    if i + 1 >= loom.state('n_records') then loom.set_state('done', 1) end
    return wave
end

function stream_feed(inputs)
    if loom.state('done') ~= 0 then return {} end
    return kale_next()
end

function stream_finish(inputs)
    return {}
end
