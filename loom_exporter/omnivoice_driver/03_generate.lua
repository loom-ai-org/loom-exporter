    -- ===== One pass: the stream's chunks, joined and post-processed ONCE, as the reference does. =====
    --
    -- The chunks are the stream's (`omni_chunk`), so the codes are the live path's; what differs is
    -- only where the post-processing runs -- over the joined audio here (`_decode_and_post_process`),
    -- per piece in a stream (ADR-131). With one chunk the two are the same audio.
    stream_open(inputs)
    local n = #loom.state('lens')
    wave = {}
    -- For a gate: the planned chunks' prompt prefixes and frame counts, to hold against the reference's
    -- own `input_ids` before any step runs -- `[n, lens..., offsets..., prefix ids...]`.
    if inputs.return_prompt ~= nil and inputs.return_prompt ~= 0 then
        wave[1] = n
        omni_append(wave, loom.state('lens'))
        omni_append(wave, loom.state('prefix_offsets'))
        omni_append(wave, loom.state('prefix'))
        return wave
    end
    for i = 0, n - 1 do
        local codes, t = omni_chunk(i)
        if inputs.return_codes ~= nil and inputs.return_codes ~= 0 then
            omni_append(wave, codes)
        else
            local piece = omni_decode(codes, t)
            if n > 1 then piece = omni_cross_fade_edges(piece, i == 0, i == n - 1) end
            omni_append(wave, piece)
        end
    end
    if inputs.return_codes == nil or inputs.return_codes == 0 then
        local ref_rms = loom.state('ref_rms')
        if loom.state('n_ref') == 0 then ref_rms = nil end
        wave = omni_post_process(wave, ref_rms, loom.state('postprocess'), true, true)
    end
    stream_finish({})
