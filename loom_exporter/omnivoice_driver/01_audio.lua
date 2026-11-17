-- ===== OmniVoice: the reference's waveform post-processing, ported (omnivoice/utils/audio.py). =====
--
-- `remove_silence` goes through pydub, so this ports pydub too: the waveform becomes 16-bit integers
-- (`(x * 32768).clip(-32768, 32767).astype(int16)`, truncating), silence is found on millisecond slices
-- by `audioop.rms` -- the integer square root of the mean square -- against -50 dBFS, and the result
-- comes back as `int16 / 32768`. Every quantity below is that arithmetic, so the post-processed audio
-- is the reference's sample for sample, not merely close: the int16 round trip is part of its output.

local OMNI_RMS_SILENT = 10 ^ (-50 / 20) * 32768      -- db_to_float(-50) * max_possible_amplitude

local function omni_round_half_even(x)
    local f = math.floor(x)
    local d = x - f
    if d < 0.5 then return f elseif d > 0.5 then return f + 1 elseif f % 2 == 0 then return f end
    return f + 1
end

-- An AudioSegment: `samples` (int16 values as Lua numbers), `first`/`last` the live range, and the
-- prefix sums of squares (exact: every partial sum is an integer below 2^53).
local function omni_segment(samples, first, last)
    return {samples = samples, first = first, last = last}
end

local function omni_seg_frames(seg) return seg.last - seg.first + 1 end
local function omni_seg_len_ms(seg) return omni_round_half_even(1000 * omni_seg_frames(seg) / SAMPLE_RATE) end

local function omni_seg_prefix(seg)
    if seg.prefix == nil then
        local p, acc = {}, 0
        p[0] = 0
        for i = seg.first, seg.last do
            local v = seg.samples[i]
            acc = acc + v * v
            p[i - seg.first + 1] = acc
        end
        seg.prefix = p
    end
    return seg.prefix
end

-- `seg[start:end]` in ms -> frame offsets into the segment, pydub's `_parse_position` (truncating).
local function omni_ms_frames(ms) return math.floor(ms * SAMPLE_RATE / 1000.0) end

local function omni_slice_rms(seg, start_ms, end_ms)
    local len = omni_seg_len_ms(seg)
    if start_ms > len then start_ms = len end
    if end_ms > len then end_ms = len end
    local a, b = omni_ms_frames(start_ms), omni_ms_frames(end_ms)
    local n = omni_seg_frames(seg)
    if b > n then b = n end
    if b <= a then return 0 end
    local p = omni_seg_prefix(seg)
    return math.floor(math.sqrt((p[b] - p[a]) / (b - a)))
end

local function omni_slice(seg, start_ms, end_ms)
    local len = omni_seg_len_ms(seg)
    if start_ms > len then start_ms = len end
    if end_ms > len then end_ms = len end
    local a, b = omni_ms_frames(start_ms), omni_ms_frames(end_ms)
    if b > omni_seg_frames(seg) then b = omni_seg_frames(seg) end
    if b < a then b = a end
    return omni_segment(seg.samples, seg.first + a, seg.first + b - 1)
end

-- `detect_silence(seg, min_silence_len, -50, seek_step)` -> `detect_nonsilent`, verbatim.
local function omni_nonsilent(seg, min_len, seek)
    local seg_len = omni_seg_len_ms(seg)
    local silent = {}
    if seg_len >= min_len then
        local starts = {}
        local last_start = seg_len - min_len
        for i = 0, last_start, seek do
            if omni_slice_rms(seg, i, i + min_len) <= OMNI_RMS_SILENT then starts[#starts + 1] = i end
        end
        if last_start % seek ~= 0 and omni_slice_rms(seg, last_start, last_start + min_len) <= OMNI_RMS_SILENT then
            starts[#starts + 1] = last_start
        end
        if #starts > 0 then
            local prev = starts[1]
            local range_start = prev
            for k = 2, #starts do
                local s = starts[k]
                if s ~= prev + seek and s > prev + min_len then
                    silent[#silent + 1] = {range_start, prev + min_len}
                    range_start = s
                end
                prev = s
            end
            silent[#silent + 1] = {range_start, prev + min_len}
        end
    end
    if #silent == 0 then return {{0, seg_len}} end
    if silent[1][1] == 0 and silent[1][2] == seg_len then return {} end
    local out, prev_end = {}, 0
    local end_i = 0
    for _, r in ipairs(silent) do
        out[#out + 1] = {prev_end, r[1]}
        prev_end, end_i = r[2], r[2]
    end
    if end_i ~= seg_len then out[#out + 1] = {prev_end, seg_len} end
    if out[1][1] == 0 and out[1][2] == 0 then table.remove(out, 1) end
    return out
end

-- `detect_leading_silence(seg, -50, 10)`: 10 ms steps while the slice's dBFS is below -50. dBFS is
-- -inf at rms 0, else 20 log10(rms / 32768).
local function omni_leading_silence(seg)
    local len = omni_seg_len_ms(seg)
    local trim = 0
    while trim < len do
        local rms = omni_slice_rms(seg, trim, trim + 10)
        if rms ~= 0 and 20 * math.log10(rms / 32768) >= -50 then break end
        trim = trim + 10
    end
    return math.min(trim, len)
end

local function omni_reverse(seg)
    local out = {}
    for i = seg.last, seg.first, -1 do out[#out + 1] = seg.samples[i] end
    return omni_segment(out, 1, #out)
end

-- `remove_silence(audio, sr, mid_sil, lead_sil, trail_sil)` on float samples -> float samples.
local function omni_remove_silence(wave, mid, lead, trail)
    local ints = {}
    for i = 1, #wave do
        local v = wave[i] * 32768.0
        if v < -32768 then v = -32768 elseif v > 32767 then v = 32767 end
        ints[i] = v >= 0 and math.floor(v) or math.ceil(v)
    end
    local seg = omni_segment(ints, 1, #ints)
    if mid > 0 then
        -- `split_on_silence(keep_silence=mid)`: each non-silent range widened by `mid` both ways,
        -- overlapping neighbours meeting halfway, then the pieces concatenated.
        local ranges = {}
        for _, r in ipairs(omni_nonsilent(seg, mid, 10)) do ranges[#ranges + 1] = {r[1] - mid, r[2] + mid} end
        for k = 1, #ranges - 1 do
            if ranges[k + 1][1] < ranges[k][2] then
                local m = math.floor((ranges[k][2] + ranges[k + 1][1]) / 2)
                ranges[k][2], ranges[k + 1][1] = m, m
            end
        end
        local joined = {}
        local len = omni_seg_len_ms(seg)
        for _, r in ipairs(ranges) do
            local piece = omni_slice(seg, math.max(r[1], 0), math.min(r[2], len))
            for i = piece.first, piece.last do joined[#joined + 1] = seg.samples[i] end
        end
        seg = omni_segment(joined, 1, #joined)
    end
    -- `remove_silence_edges`: keep `lead` ms before the first sound and `trail` after the last. A
    -- stream's middle piece passes nil for the edge it does not own: that edge is mid-utterance.
    if lead ~= nil then
        local start = math.max(0, omni_leading_silence(seg) - lead)
        seg = omni_slice(seg, start, omni_seg_len_ms(seg))
    end
    if trail ~= nil then
        seg = omni_reverse(seg)
        local start = math.max(0, omni_leading_silence(seg) - trail)
        seg = omni_slice(seg, start, omni_seg_len_ms(seg))
        seg = omni_reverse(seg)
    end
    local out = {}
    for i = seg.first, seg.last do out[#out + 1] = seg.samples[i] / 32768.0 end
    return out
end

-- numpy's `linspace(a, b, k, dtype=float32)`: computed in double (`a + i * step`, the last point set to
-- `b`), then cast.
local function omni_linspace_f32(a, b, k)
    local out = {}
    if k == 1 then out[1] = to_f32(a) return out end
    local step = (b - a) / (k - 1)
    for i = 0, k - 1 do out[i + 1] = to_f32(a + i * step) end
    out[k] = to_f32(b)
    return out
end

-- `fade_and_pad_audio`: a linear fade over `fade` seconds at each end, then `pad` seconds of zeros.
local function omni_fade_and_pad(wave, pad, fade, fade_in, fade_out)
    local n = #wave
    if n == 0 then return wave end
    local k = math.min(math.floor(fade * SAMPLE_RATE), math.floor(n / 2))
    if k > 0 then
        if fade_in then
            local w = omni_linspace_f32(0, 1, k)
            for i = 1, k do wave[i] = to_f32(wave[i] * w[i]) end
        end
        if fade_out then
            local w = omni_linspace_f32(1, 0, k)
            for i = 1, k do wave[n - k + i] = to_f32(wave[n - k + i] * w[i]) end
        end
    end
    local p = math.floor(pad * SAMPLE_RATE)
    local out = {}
    if fade_in then for _ = 1, p do out[#out + 1] = 0.0 end end
    for i = 1, n do out[#out + 1] = wave[i] end
    if fade_out then for _ = 1, p do out[#out + 1] = 0.0 end end
    return out
end

-- `cross_fade_chunks`' two halves, applied to one chunk: a 100 ms fade-out and 100 ms of silence after
-- every chunk but the last, a 100 ms fade-in before every chunk but the first. The reference fades the
-- merged audio, but its fades never reach across a boundary, so cutting it per chunk is exact.
local function omni_cross_fade_edges(wave, first, last)
    local fade_n = math.floor(math.floor(0.3 * SAMPLE_RATE) / 3)
    if not first then
        local k = math.min(fade_n, #wave)
        local w = omni_linspace_f32(0, 1, k)
        for i = 1, k do wave[i] = to_f32(wave[i] * w[i]) end
    end
    if not last then
        local n = #wave
        local k = math.min(fade_n, n)
        local w = omni_linspace_f32(1, 0, k)
        for i = 1, k do wave[n - k + i] = to_f32(wave[n - k + i] * w[i]) end
        for _ = 1, fade_n do wave[#wave + 1] = 0.0 end
    end
    return wave
end

-- `_post_process_audio`: silences trimmed (when asked), then the level -- the reference clip's when it
-- was quiet, else peak 0.5 when there was no reference -- then the edge fades and padding.
local function omni_post_process(wave, ref_rms, postprocess, first, last)
    if postprocess ~= 0 then
        wave = omni_remove_silence(wave, 500, first and 100 or nil, last and 100 or nil)
    end
    if ref_rms ~= nil and ref_rms >= 0 and ref_rms < 0.1 then
        local r = to_f32(ref_rms)
        for i = 1, #wave do wave[i] = to_f32(to_f32(wave[i] * r) / to_f32(0.1)) end
    elseif ref_rms == nil or ref_rms < 0 then
        local peak = 0.0
        for i = 1, #wave do local a = math.abs(wave[i]) if a > peak then peak = a end end
        if peak > 1e-6 then
            for i = 1, #wave do wave[i] = to_f32(wave[i] / peak) * 0.5 end
        end
    end
    return omni_fade_and_pad(wave, PAD_DURATION, FADE_DURATION, first, last)
end
