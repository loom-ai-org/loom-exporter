    -- ===== T3: text -> S3 speech tokens, under classifier-free guidance. =====
    --
    -- `inputs.tokens` is the text's BPE ids with no framing. The start/stop ids are added HERE because
    -- `ChatterboxTTS.generate` adds them after tokenizing, so a host that tokenizes the way the
    -- reference does hands over exactly what `text_to_tokens` returns.
    if inputs.seed then loom.seed_rng(inputs.seed) end
    local _text = {START_TEXT}
    for _i = 1, #inputs.tokens do _text[#_text + 1] = inputs.tokens[_i] end
    _text[#_text + 1] = STOP_TEXT

    local _spk = chatterbox_voice(inputs.speaker_emb, 'voice.speaker_emb')
    local _cond_tokens = chatterbox_voice(inputs.cond_prompt_tokens, 'voice.cond_prompt_tokens')
    local _exaggeration = inputs.exaggeration or DEFAULT_EXAGGERATION
    -- The checkpoint's own centring, `cond + w * (cond - uncond)`, is `uncond + (w + 1) * (cond -
    -- uncond)` -- the general form `loom.sample_row` implements. Dia's conversion, for Dia's reason.
    local _cfg_weight = inputs.cfg_weight or DEFAULT_CFG_WEIGHT
    local _use_cfg = _cfg_weight > 0.0
    local _temperature = inputs.temperature or DEFAULT_TEMPERATURE
    local _min_p = inputs.min_p or DEFAULT_MIN_P
    local _top_p = inputs.top_p or DEFAULT_TOP_P
    local _penalty = inputs.repetition_penalty or DEFAULT_REPETITION_PENALTY

    -- One prefill per stream. The unconditional one is the same graph with `text_keep = 0`: the
    -- reference zeroes the text's TOKEN embeddings and keeps their positions, which is what the flag
    -- multiplies. Each stream's prefill embedding is consumed immediately, so one module serves both.
    local _n_text = #_text
    local _prefill = N_COND_ROWS + _n_text + N_BOS_ROWS
    local function _prefill_embed(_keep)
        loom.run_subgraph_and_retain('t3_prefill_embed', {n_tokens = _n_text, n_past = 0},
            {speaker_emb = _spk, prompt_tokens = _cond_tokens, emotion_adv = {_exaggeration},
             text_ids = _text, text_keep = {_keep}})
        return {inputs_embeds = {from = 't3_prefill_embed'}, position_ids = loom.range(0, _prefill),
                attention_mask = loom.causal_mask(_prefill, 0)}
    end
    loom.run_subgraph_and_retain('t3_lm', {n_tokens = _prefill, n_past = 0}, _prefill_embed(1.0))
    if _use_cfg then
        loom.run_subgraph_and_retain('t3_lm_uncond', {n_tokens = _prefill, n_past = 0},
                                      _prefill_embed(0.0))
    end

    -- The penalised history starts at BOS: `T3.inference` seeds `generated_ids` with it, so the start
    -- token is penalised from the first draw on.
    local _history = {START_SPEECH}
    local speech_tokens = {}
    local _n_past = _prefill
    local _budget = math.min(inputs.max_new_tokens or DEFAULT_MAX_NEW_TOKENS,
                             T3_MAX_POSITIONS - _prefill)
    for _step = 1, _budget do
        local _opts = {temperature = _temperature, min_p = _min_p, top_p = _top_p,
                       repetition_penalty = _penalty, penalized = _history}
        if _use_cfg then
            _opts.guidance = {module = 't3_lm_uncond', scale = _cfg_weight + 1.0}
        end
        local _token = loom.sample_row('t3_lm', -1, _opts)
        _history[#_history + 1] = _token
        if _token == STOP_SPEECH then break end
        -- Control ids are FED BACK like any other draw -- the reference only drops them on the way to
        -- S3Gen -- so the filter is on what is kept, not on what is embedded.
        if _token < SPEECH_VOCAB then speech_tokens[#speech_tokens + 1] = _token end
        -- Speech position `_step`: the prefill's BOS rows are position 0 and draw k sits at k.
        loom.run_subgraph_and_retain('t3_step_embed', {n_tokens = 1, n_past = 0},
            {token = {_token}, position = {_step}})
        local _in = {inputs_embeds = {from = 't3_step_embed'}, position_ids = {_n_past},
                     attention_mask = loom.causal_mask(1, _n_past)}
        loom.run_subgraph_and_retain('t3_lm', {n_tokens = 1, n_past = _n_past}, _in)
        if _use_cfg then
            loom.run_subgraph_and_retain('t3_lm_uncond', {n_tokens = 1, n_past = _n_past}, _in)
        end
        _n_past = _n_past + 1
    end
    if #speech_tokens == 0 then
        error('chatterbox: T3 produced no speech tokens (the first draw was the stop token)')
    end
