-- ===== OmniVoice: a reference clip -> its codes (`create_voice_clone_prompt`, ported). =====
--
-- The clip arrives at 24 kHz (the host resamples, as for every cloning model here). Its level is
-- measured before anything touches it -- the generated audio is scaled back to it at the end when it was
-- quiet -- then a quiet clip is raised to rms 0.1, its silences trimmed (pydub again: 200 ms kept inside,
-- 100 ms before, 200 ms after), and its tail cut to a whole number of codec frames.
omni_encode_reference = function(waveform)
    local n = #waveform
    if n == 0 then error('omnivoice: the reference clip is empty') end
    local sq = 0.0
    for i = 1, n do sq = sq + to_f32(waveform[i] * waveform[i]) end
    -- `float(np.sqrt(np.mean(wav ** 2)))` on float32: the mean and the root are both float32 values.
    local rms = to_f32(math.sqrt(to_f32(sq / n)))
    local wave = {}
    if rms > 0 and rms < 0.1 then
        for i = 1, n do wave[i] = to_f32(to_f32(waveform[i] * 0.1) / rms) end
    else
        for i = 1, n do wave[i] = waveform[i] end
    end
    if loom.state('preprocess_prompt') ~= 0 then wave = omni_remove_silence(wave, 200, 100, 200) end
    local frames = math.floor(#wave / HOP)
    if frames < 2 then error('omnivoice: the reference clip is shorter than two codec frames after trimming') end
    local clip = {}
    for i = 1, frames * HOP do clip[i] = wave[i] end
    local rows = {}
    for j = 0, frames - 1 do rows[j + 1] = j * SEMANTIC_DOWNSAMPLE end
    local codes = loom.run_subgraph('ref_encode', {n_codes = frames, n_past = 0},
                                    {waveform = clip, semantic_rows = rows})
    return codes, frames, rms
end
