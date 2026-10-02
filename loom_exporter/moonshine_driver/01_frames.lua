    -- The processor zero-pads a clip to whole frames and MASKS the partial one, which zeroes its
    -- embedding. An all-zero frame embeds to exactly zero, so the same encoder input is the real
    -- samples of every whole frame followed by zeros to the next frame boundary. In place: at most
    -- `FRAME - 1` samples are rewritten and at most that many appended.
    local _wav = inputs.waveform
    local _n_real = inputs.audio_samples or #_wav
    local _n_frames = math.ceil(#_wav / FRAME)
    for i = math.floor(_n_real / FRAME) * FRAME + 1, _n_frames * FRAME do _wav[i] = 0.0 end

    -- The encoder rows this will produce -- two stride-2 causal convolutions over the frames -- checked
    -- against the decoder's position table BEFORE the encoder runs: a row past the table would be a
    -- gather out of bounds, not an error.
    local _rows = math.ceil(math.ceil(_n_frames / 2) / 2)
    if _rows > MAX_ENC_FRAMES then
        error(string.format("moonshine: %.1f s of audio is %d encoder frames; the model's position table "
            .. "has %d (%.1f s). Split the audio (a VAD finds the pauses) and transcribe each part.",
            _n_real / SAMPLE_RATE, _rows, MAX_ENC_FRAMES, MAX_ENC_FRAMES * 4 * FRAME / SAMPLE_RATE))
    end

    -- The card's budget, less the start token, inside the KV cache.
    local _max_new = math.min(math.floor(_n_real * TOKENS_PER_SECOND / SAMPLE_RATE), MAX_POSITIONS) - 1
    if inputs.max_new_tokens == nil then inputs.max_new_tokens = _max_new end
    if inputs.max_new_tokens <= 0 then return {} end
