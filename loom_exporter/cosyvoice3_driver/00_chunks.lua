    -- ===== Chunks: one generation per piece of the reference's text path. =====
    --
    -- `loom::CosyVoice3Vocab::encode` opens every chunk with CHUNK_HEADER (`<|endoftext|>`): the
    -- reference's `text_normalize` splits a paragraph into ~80-token pieces and synthesises each one
    -- SEPARATELY -- its own LM decode, flow and vocoder, from the same voice -- then joins the audio. So
    -- `infer` runs once per chunk and the waveforms are concatenated, the engine's stream continuing
    -- from one chunk to the next as the reference's RNG does. Ids with no header (a host that tokenized
    -- elsewhere) are one chunk, as they always were.
    if tokens[1] == CHUNK_HEADER then
        local _chunks = cosyvoice3_split_chunks(tokens, CHUNK_HEADER)
        if #_chunks > 1 then
            -- Each of these pins ONE generation (its tokens, its draws, its noise, or what it returns).
            for _, _key in ipairs({'speech_tokens', 'draws', 'noise', 'nsf_noise', 'return_tokens', 'return_mel'}) do
                if inputs[_key] ~= nil then
                    error('cosyvoice3: inputs.' .. _key .. ' pins one generation, and this text is ' ..
                          #_chunks .. ' chunks -- pass one sentence, or leave it out')
                end
            end
        end
        if inputs.seed then loom.seed_rng(inputs.seed) end
        local _wave = {}
        for _c = 1, #_chunks do
            local _chunk_inputs = {}
            for _k, _v in pairs(inputs) do _chunk_inputs[_k] = _v end
            _chunk_inputs.tokens = _chunks[_c]
            _chunk_inputs.seed = nil
            local _part = infer(_chunk_inputs)
            if #_chunks == 1 then return _part end
            for _i = 1, #_part do _wave[#_wave + 1] = _part[_i] end
        end
        return _wave
    end
