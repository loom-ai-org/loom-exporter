-- Lua orchestration for the MIL-traced F5-TTS export (loom_exporter/f5_tts_export.py).
--
-- F5-TTS has no duration model and no phonemiser. It IN-FILLS one mel spectrogram: the reference
-- clip's mel occupies the first frames, the rest is noise, the text is the reference transcript
-- followed by the text to speak, and the flow-matching decoder integrates the whole thing at once.
-- Everything this file does before the sampler exists to build that single conditioning frame grid.
--
-- inputs:
--   waveform      the reference clip, f32 samples at SAMPLE_RATE, mono. REQUIRED.
--   text_ids      the reference transcript's character ids followed by the ids to speak. REQUIRED.
--   n_ref_text    how many leading entries of `text_ids` are the reference transcript. REQUIRED --
--                 the duration estimate is a ratio of transcript lengths and there is nothing in the
--                 ids that marks the join.
--   n_steps       ODE steps (default DEFAULT_STEPS). cfg_scale, sway_coef, speed: the three knobs
--                 `infer_process` exposes, at the reference's own defaults.
--   duration      total frames, overriding the estimate. The reference's `fix_duration`, in frames.
--   seed          seeds loom's shared RNG, which is where the sampler's initial noise comes from.
--   noise         the sampler's initial state outright, n_frames * N_MEL floats, overriding the draw.
--                 Not a convenience: torch's RNG and this engine's are different algorithms, so the
--                 same seed is a DIFFERENT draw, and flow matching from a different draw is a
--                 different valid sample -- which makes a tensor comparison against the reference
--                 impossible without it (loom.cpp's gate measured max |d| 1.25 on audio that was
--                 perfectly intelligible). Absent, the engine draws as before.
--
-- Returns: the GENERATED waveform alone (flat f32 array at SAMPLE_RATE) -- the reference's frames are
-- sliced off the mel before the vocoder ever sees them, so nothing pays to re-synthesise the prompt.
--
-- The RMS normalisation at both ends is `infer_batch_process`'s, not preprocessing a caller is
-- expected to do: the model was trained on clips at ~0.1 RMS, and a quiet reference conditions it on
-- the wrong loudness. Scaling the output back by the same factor is what makes it a normalisation
-- rather than a gain change.

-- A scalar driver input, however the HOST spelled it. `loom_cli` passes a one-element `--input` as a
-- bare number and a longer one as a table; a Python caller passes whatever it built. Indexing a
-- number is a Lua error and reading a table as a number is silently `nil`, so neither spelling can be
-- assumed -- and every knob below is one a caller may or may not name.
local function opt_scalar(v, default)
    if v == nil then return default end
    if type(v) == "table" then return v[1] end
    return v
end

-- `t + coef*(cos(pi/2 * t) - 1 + t)` over a linspace: F5-TTS's "sway sampling", which spends more
-- steps near t=0 where the field changes fastest. N+1 points, so N steps. `coef = 0` is the plain
-- linspace, which is what `sway_coef = 0` gets you.
local function sway_times(n_steps, coef)
    local times = {}
    for step = 0, n_steps do
        local t = step / n_steps
        times[step + 1] = t + coef * (math.cos(math.pi / 2 * t) - 1 + t)
    end
    return times
end

-- The character ids as the graph wants them: `+1` (the reference's own filler-token offset, so id 0
-- means "no character here"), CURTAILED to the frame count if the text is longer than the audio will
-- be, and zero-padded if it is shorter. `keep` is the complement of the reference's `text_mask` --
-- 1.0 where a real character sits -- and it is handed in rather than recomputed in the graph because
-- the unconditional branch replaces the ids and must still mask with THIS mask.
local function f5_text_arrays(text_ids, n_frames)
    local ids, uncond_ids, keep = {}, {}, {}
    local n_text = math.min(#text_ids, n_frames)
    for i = 1, n_frames do
        if i <= n_text then
            ids[i] = text_ids[i] + 1
            keep[i] = 1.0
        else
            ids[i] = 0
            keep[i] = 0.0
        end
        uncond_ids[i] = 0
    end
    return ids, uncond_ids, keep
end

-- The conditioning mel: the reference's frames, then zeros out to `n_frames`. This is the reference's
-- `where(cond_mask, F.pad(cond), 0)` with both halves written out, and the zero tail is what the
-- model in-fills. `mel` arrives frame-major (`n_mel` contiguous floats per frame), which is the
-- layout the estimator's own `cond` input declares.
local function f5_step_cond(mel, cond_len, n_frames, n_mel)
    local cond, zero = {}, {}
    local kept = cond_len * n_mel
    for i = 1, n_frames * n_mel do
        cond[i] = (i <= kept) and mel[i] or 0.0
        zero[i] = 0.0
    end
    return cond, zero
end
