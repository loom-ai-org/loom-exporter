-- Voxtral-4B-TTS's helpers. Top level, so every fragment below can call them.

-- `[lo, hi)` of a flat table, as a new table: a frame's codes out of a frame-major list.
local function voxtral_slice(list, lo, hi)
    local out = {}
    for i = lo + 1, hi do out[#out + 1] = list[i] end
    return out
end

-- The codec over `n_frames` frame-major 37-code frames, in chunks of `chunk` frames that each re-read
-- `context` frames before them and drop those frames' samples. The decoder is causal with a finite
-- receptive field, so with enough context this is the one-call decode (voxtral_tts_export's
-- CODEC_CONTEXT); a chunk bounds the attention's `[8T, 8T]` bias, which one call over a minute of audio
-- would make gigabytes.
local function voxtral_decode(codes, n_frames, width, chunk, context, samples_per_frame)
    local wave = {}
    local start = 0
    while start < n_frames do
        local stop = math.min(start + chunk, n_frames)
        local from = math.max(0, start - context)
        local t = stop - from
        local out = loom.run_subgraph('codec', {n_codes = t, n_past = 0},
            {codes = voxtral_slice(codes, from * width, stop * width), pos1 = loom.range(0, t),
             pos2 = loom.range(0, 2 * t), pos4 = loom.range(0, 4 * t), pos8 = loom.range(0, 8 * t)})
        for i = (start - from) * samples_per_frame + 1, #out do wave[#wave + 1] = out[i] end
        start = stop
    end
    return wave
end
