local IS_PHONE, IS_VOWEL = {}, {}
for i, flag in ipairs(IS_PHONE_LIST) do IS_PHONE[i - 1] = flag == 1 end
for i, flag in ipairs(IS_VOWEL_LIST) do IS_VOWEL[i - 1] = flag == 1 end

-- ===== tiny-voca: phonemes in, 24 kHz out, one utterance per graph pass (loom.cpp ADR-133). =====
--
-- Input `tokens`: ids from this file's phoneme table, which holds every symbol of upstream's inventory
-- three times -- plain (id k), with primary stress (VOCAB + k: `ˈiː`) and with secondary (2 VOCAB + k:
-- `ˌiː`) -- so a dizione string ("<bos> h ə l ˈoʊ ! <eos>") encodes by longest match and the stress
-- rides in the id. One or more utterances, each `<bos> ... <eos>` (a bare run is wrapped); the reference
-- (`speak_text`) phonemizes and speaks a text one SENTENCE at a time, and the host's door does the same
-- cut. The last utterance ending on a phone gets a full stop first (`finished`). Options: `speaker` (0 F2,
-- 1 M2), `speed`, `pitch_shift`, `pitch_range`, `seed`, `noise` (pinned draws, every utterance's end to
-- end).

local function voca_utterances(ids)
    local out, current = {}, nil
    for _, raw in ipairs(ids) do
        local id, stress = raw, 0
        if id >= 2 * VOCAB then id, stress = id - 2 * VOCAB, 2 elseif id >= VOCAB then id, stress = id - VOCAB, 1 end
        if id == BOS_ID then
            current = {ids = {BOS_ID}, stress = {0}}
        elseif id == EOS_ID then
            if current == nil then current = {ids = {BOS_ID}, stress = {0}} end
            current.ids[#current.ids + 1], current.stress[#current.stress + 1] = EOS_ID, 0
            out[#out + 1] = current
            current = nil
        elseif id ~= PAD_ID then
            if current == nil then current = {ids = {BOS_ID}, stress = {0}} end
            current.ids[#current.ids + 1], current.stress[#current.stress + 1] = id, stress
        end
    end
    if current ~= nil then
        current.ids[#current.ids + 1], current.stress[#current.stress + 1] = EOS_ID, 0
        out[#out + 1] = current
    end
    -- `finished`: a text ending without a mark gets a full stop, which only the last sentence carries.
    local last = out[#out]
    if last ~= nil and IS_PHONE[last.ids[#last.ids - 1]] then
        table.insert(last.ids, #last.ids, PERIOD_ID)
        table.insert(last.stress, #last.stress, 0)
    end
    for _, u in ipairs(out) do
        local any = false
        for _, id in ipairs(u.ids) do if IS_PHONE[id] or id == UNK_ID then any = true end end
        if not any then error('tiny-voca: an utterance has nothing to pronounce') end
        if #u.ids > MAX_TOKENS then
            error(('tiny-voca: an utterance of %d tokens; the model reads at most %d -- split the text into '
                   .. 'sentences first'):format(#u.ids, MAX_TOKENS))
        end
    end
    return out
end

local function voca_range(n)
    local r = {}
    for i = 1, n do r[i] = i - 1 end
    return r
end

-- The band-limited pulse train (`tiny_voca.model.pulse_train`) in double precision, sample by sample:
-- log-linear F0 between frame knots, the phase's running sum, the closed-form sum of the harmonics below
-- HARMONIC_LIMIT * sr with the top one faded in, at unit power. Rounded to float32 at the end, as
-- upstream's `.float()`.
local function voca_pulses(f0)
    local frames = #f0
    local knots = {}
    for j = 1, frames do knots[j] = math.log(math.max(f0[j], 1.0)) end
    knots[frames + 1] = knots[frames]
    local out = {}
    local acc = 0.0
    local two_pi = 2 * math.pi
    for s = 0, frames * HOP - 1 do
        local position = s / HOP
        local low = math.floor(position)
        local w = position - low
        local frequency = math.exp(knots[low + 1] * (1 - w) + knots[low + 2] * w)
        acc = acc + frequency / SAMPLE_RATE
        local theta = two_pi * acc
        theta = theta - two_pi * math.floor(theta / two_pi)
        local harmonics = math.max(HARMONIC_LIMIT * SAMPLE_RATE / frequency, 1.0)
        local whole = math.floor(harmonics)
        local denominator = 2 * math.sin(theta / 2)
        local core
        if math.abs(denominator) < 1e-7 then core = whole
        else core = math.sin((whole + 0.5) * theta) / denominator - 0.5 end
        local wave = core + (harmonics - whole) * math.cos((whole + 1) * theta)
        out[s + 1] = to_f32(wave * math.sqrt(2.0 / harmonics))
    end
    return out
end

-- One utterance, `TinyVoca.synthesize` with its defaults (pitch_jitter 0, voiced_vowels on). `pinned` is
-- the pinned draws still unread (this utterance takes the first `frames * hop`), or nil to draw them as
-- the reference does: `randn(frames * hop)` seeded `seed`.
local function voca_speak(u, voice, speed, pitch_shift, pitch_range, seed, pinned)
    local n = #u.ids
    loom.run_subgraph_and_retain('encode', {n_tokens = n, n_past = 0},
        {tokens = u.ids, stress = u.stress, voice = {voice}, positions = voca_range(n)})
    local d = loom.get_output('encode', 2)                       -- [N, 2], row-major
    -- Durations: round(expm1(clamp(d0, 0, log1p 400)) / speed), at least 1; zero for a token that is no
    -- phone and whose presence logit is not positive. float32 throughout, round half to even.
    local durations, total = {}, 0
    local cap, sp = to_f32(math.log(401)), to_f32(speed)
    for i = 1, n do
        local x = math.min(math.max(d[2 * i - 1], 0), cap)
        local frames = round_half_to_even(to_f32(to_f32(math.exp(x) - 1) / sp))
        if frames < 1 then frames = 1 end
        if not (IS_PHONE[u.ids[i]] or d[2 * i] > 0) then frames = 0 end
        durations[i] = frames
        total = total + frames
    end
    if total < 1 then error('tiny-voca: the utterance has no frames') end
    if total > MAX_FRAMES then error('tiny-voca: the utterance needs more than ' .. MAX_FRAMES .. ' frames; split it') end
    -- Each frame's owner token and its position inside it (`TinyVoca.expand`).
    local owner, features, is_vowel = {}, {}, {}
    local t = 0
    for i = 1, n do
        local owned = math.max(durations[i], 1)
        local tail = to_f32(to_f32(math.log(1 + owned)) / 4)
        for k = 0, durations[i] - 1 do
            t = t + 1
            owner[t] = i - 1
            local relative = to_f32(to_f32(k + 0.5) / owned)
            features[3 * t - 2], features[3 * t - 1], features[3 * t] = relative, to_f32(1 - relative), tail
            is_vowel[t] = IS_VOWEL[u.ids[i]] or false
        end
    end
    local positions = voca_range(total)
    loom.run_subgraph_and_retain('prosody', {n_enc_frames = total, n_tokens = n, n_past = 0},
        {hidden = {from = 'encode', index = 1}, owner = owner, features = features, voice = {voice},
         positions = positions})
    local p = loom.get_output('prosody', 2)                      -- [T, 3]
    -- A vowel is voiced by definition; the pitch range around the voiced mean; the hard conditioning.
    for j = 1, total do
        if is_vowel[j] and p[3 * j - 1] < 2.0 then p[3 * j - 1] = 2.0 end
    end
    if pitch_range ~= 1.0 then
        local sum, count = 0.0, 0
        for j = 1, total do if p[3 * j - 1] > 0 then sum, count = sum + p[3 * j - 2], count + 1 end end
        local mean = to_f32(sum / math.max(count, 1))
        local r = to_f32(pitch_range)
        for j = 1, total do p[3 * j - 2] = to_f32(mean + to_f32(to_f32(p[3 * j - 2] - mean) * r)) end
    end
    local condition, f0 = {}, {}
    for j = 1, total do
        local pitch = p[3 * j - 2]
        condition[3 * j - 2] = math.min(math.max(pitch, -2.0), 1.5)
        condition[3 * j - 1] = p[3 * j - 1] > 0 and 1.0 or 0.0
        condition[3 * j] = math.min(math.max(p[3 * j], -4.5), 3.0)
        local hz = to_f32(PITCH_REFERENCE_HZ * to_f32(math.exp(to_f32(pitch + pitch_shift))))
        f0[j] = math.min(math.max(hz, F0_MIN), F0_MAX)
    end
    loom.run_subgraph_and_retain('acoustic', {n_enc_frames = total, n_past = 0},
        {states = {from = 'prosody', index = 1}, condition = condition, positions = positions})
    -- `SourceFilter._subframes`: sub-frame j uses frame j // steps; past the end, the last sub-frame.
    local function subframes(steps)
        local sub, own, nxt = {}, {}, {}
        for j = 0, steps * total + 2 do
            local s = math.min(j, steps * total - 1)
            sub[j + 1] = s
            own[j + 1] = math.floor(s / steps)
            nxt[j + 1] = math.min(math.floor((s + 1) / steps), total - 1)
        end
        return sub, own, nxt
    end
    local h_sub, here, following = subframes(HOP / SYNTH_HOP)
    local n_sub, n_owner = subframes(HOP / NOISE_HOP)
    local noise
    if pinned ~= nil then
        if #pinned < total * HOP then error('tiny-voca: fewer pinned noise values than the utterance needs') end
        noise = {}
        for k = 1, total * HOP do noise[k] = pinned[k] end
    else
        noise = aten_randn(seed % 4294967296, total * HOP)
    end
    local wave = loom.run_subgraph('synth', {n_enc_frames = total, n_samples = total * HOP, n_past = 0},
        {h = {from = 'acoustic', index = 1}, n = {from = 'acoustic', index = 2}, here = here, following = following,
         h_sub = h_sub, n_owner = n_owner, n_sub = n_sub, pulses = voca_pulses(f0), noise = noise})
    return wave, total
end

-- `limit_peak`: scaled down as a whole when the loudest sample is above PEAK.
local function voca_limit(wave)
    local loudest = 0.0
    for i = 1, #wave do local a = math.abs(wave[i]) if a > loudest then loudest = a end end
    if loudest > PEAK then
        local k = to_f32(PEAK / loudest)
        for i = 1, #wave do wave[i] = to_f32(wave[i] * k) end
    end
    return wave
end

-- ===== The live door: one utterance (sentence) per `stream_feed`. Each piece is peak-limited on its
-- own, where `infer` limits the joined audio once (ADR-133). =====
function stream_open(inputs)
    if inputs.tokens == nil or #inputs.tokens == 0 then error('tiny-voca: no phoneme ids') end
    local utterances = voca_utterances(inputs.tokens)
    local flat, offsets, stress = {}, {0}, {}
    for _, u in ipairs(utterances) do
        for k = 1, #u.ids do flat[#flat + 1], stress[#stress + 1] = u.ids[k], u.stress[k] end
        offsets[#offsets + 1] = #flat
    end
    loom.set_state('ids', flat)
    loom.set_state('stress', stress)
    loom.set_state('offsets', offsets)
    loom.set_state('voice', inputs.speaker or 0)
    loom.set_state('speed', inputs.speed or 1.0)
    loom.set_state('pitch_shift', inputs.pitch_shift or 0.0)
    loom.set_state('pitch_range', inputs.pitch_range or 1.0)
    loom.set_state('seed', inputs.seed or 0)
    if inputs.noise ~= nil then loom.set_state('noise', inputs.noise) end
    loom.set_state('noise_pos', 0)
    loom.set_state('next', 0)
    loom.set_state('done', #utterances == 0 and 1 or 0)
    return {}
end

local function voca_next()
    local i = loom.state('next')
    local offsets, ids, stress = loom.state('offsets'), loom.state('ids'), loom.state('stress')
    local u = {ids = {}, stress = {}}
    for k = offsets[i + 1] + 1, offsets[i + 2] do u.ids[#u.ids + 1], u.stress[#u.stress + 1] = ids[k], stress[k] end
    -- Pinned draws are every utterance's end to end; this one's length is known only once its durations
    -- are, so it is handed what is left and takes its own.
    local pinned = loom.state('noise')
    local rest = nil
    if pinned ~= nil then
        rest = {}
        for k = loom.state('noise_pos') + 1, #pinned do rest[#rest + 1] = pinned[k] end
    end
    local wave, frames = voca_speak(u, loom.state('voice'), loom.state('speed'), loom.state('pitch_shift'),
                                    loom.state('pitch_range'), loom.state('seed') + i, rest)
    if pinned ~= nil then loom.set_state('noise_pos', loom.state('noise_pos') + frames * HOP) end
    loom.set_state('next', i + 1)
    if i + 2 >= #offsets then loom.set_state('done', 1) end
    return wave
end

function stream_feed(inputs)
    if loom.state('done') ~= 0 and loom.state('next') >= #loom.state('offsets') - 1 then return {} end
    return voca_limit(voca_next())
end

function stream_finish(inputs)
    return {}
end
