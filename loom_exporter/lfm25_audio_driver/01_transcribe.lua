    -- ===== LFM2.5-Audio, speech to text: the prompt in three cached segments, then greedy. =====
    --
    -- Inputs: `waveform` (16 kHz) and `length` (`{n}`). Returns the transcript's ids, `<|im_end|>` not
    -- included. Optional: `max_new_tokens` (512, the README's).
    --
    -- The prompt is `ChatState`'s: the head ("<|startoftext|>", the system turn "Perform ASR.", the user
    -- turn's opening), the audio rows, then the tail (the user turn's end, the assistant turn's
    -- opening). Three cached calls compute what one concatenated call would -- attention is causal and
    -- the conv blocks carry their state -- which is family 3's segmented prefill.
    local _real = (inputs.length and inputs.length[1]) or #inputs.waveform
    loom.run_subgraph_and_retain('encoder', {n_samples = #inputs.waveform, n_past = 0},
        {waveform = inputs.waveform, length = {_real}})
    local _rows = loom.output_shape('encoder', 1)[2]

    local _n_past = 0
    local _last_n = 0                -- rows the decoder's last call produced; the head reads the last
    local function _feed_ids(ids)
        loom.run_subgraph_and_retain('embed', {n_tokens = #ids, n_past = 0}, {tokens = ids})
        loom.run_subgraph_and_retain('decoder', {n_tokens = #ids, n_past = _n_past},
            {inputs_embeds = {from = 'embed'}, position_ids = loom.range(_n_past, #ids),
             attention_mask = loom.causal_mask(#ids, _n_past)})
        _n_past = _n_past + #ids
        _last_n = #ids
    end
    _feed_ids(PROMPT_HEAD)
    loom.run_subgraph_and_retain('decoder', {n_tokens = _rows, n_past = _n_past},
        {inputs_embeds = {from = 'encoder'}, position_ids = loom.range(_n_past, _rows),
         attention_mask = loom.causal_mask(_rows, _n_past)})
    _n_past = _n_past + _rows
    _feed_ids(PROMPT_TAIL)

    ids = {}
    local _max_new = inputs.max_new_tokens or MAX_NEW_TOKENS
    for _step = 1, _max_new do
        if _n_past >= MAX_SEQ_LEN then error('lfm2.5-audio: the transcript outgrew the KV cache') end
        -- Only the last row's logits: the head is its own phase so the prompt's rows never pay for it.
        loom.run_subgraph_and_retain('lm_head', {n_tokens = 1, n_past = 0},
            {hidden = {from = 'decoder', row = _last_n - 1, rows = 1}})
        local _id = loom.argmax_row('lm_head', 0)
        if _id == END_OF_TURN then break end
        ids[#ids + 1] = _id
        _feed_ids({_id})
    end
