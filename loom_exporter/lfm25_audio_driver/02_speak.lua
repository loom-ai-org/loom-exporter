    -- ===== LFM2.5-Audio, text to speech: `generate_sequential` with a "Perform TTS." voice prompt. =====
    --
    -- Inputs: `tokens` (the text's ids, no BOS), and optionally `voice_prompt` (a voice's system-prompt
    -- ids; a voice file carries one), `temperature` / `top_k` (the audio codes' sampler, the README's
    -- 0.8 / 64 by default; 0 is greedy), `seed`, `max_new_tokens`, `return_codes`. Returns the 24 kHz
    -- waveform (or, with `return_codes`, the frames' codes, frame-major).
    --
    -- One pass is the live stream (00_stream.lua) opened, fed to its end and finished inside this call,
    -- against the fresh state the engine gives every one-pass call -- so the one-pass gate is the live
    -- path's gate too. The whole turn in one feed and one detokenizer call, unless the caller asks for
    -- frames (`piece_frames`), which is the live cut run in one call.
    if inputs.piece_frames == nil then inputs.piece_frames = 0 end
    stream_open(inputs)
    wave = {}
    while loom.state('done') == 0 do
        local _piece = stream_feed({})
        local _base = #wave
        for _i = 1, #_piece do wave[_base + _i] = _piece[_i] end
    end
    stream_finish({})
