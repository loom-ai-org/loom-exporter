    -- ===== Voxtral-4B-TTS: text -> frames of 37 codes at 12.5 Hz -> 24 kHz. =====
    --
    -- vllm-omni's path for a preset voice: `encode_speech_request`'s prompt, the voice's rows written over
    -- its [AUDIO] slots, then one frame per LM step until the semantic code is [END_AUDIO]. Inputs:
    -- `tokens` (`Tekkenizer.encode(text, bos=False, eos=False)`), and optionally `voice`, `noise` (36 unit
    -- normals per frame), `teacher_codes`, `return_codes`, `cfg`, `max_frames`, `codec_chunk`, `seed`.
    --
    -- One pass is the live stream (00_stream.lua) opened, fed to its end and finished inside this call,
    -- against the fresh state the engine gives every one-pass call -- so the one-pass gate is the live
    -- path's gate too. The whole turn in one feed and the codec over every frame, unless the caller asks
    -- for frames (`piece_frames`), which is the live cut run in one call.
    if inputs.piece_frames == nil then inputs.piece_frames = 0 end
    stream_open(inputs)
    wave = {}
    while loom.state('done') == 0 do
        local _piece = stream_feed({})
        local _base = #wave
        for _i = 1, #_piece do wave[_base + _i] = _piece[_i] end
    end
    stream_finish({})
