    -- --- DurationEncoder: 3x (BiLSTM + AdaLayerNorm), each re-concatenating the style vector.
    --     NOTHING between `albert_bert_encoder` and `duration_proj` becomes a Lua table now: the
    --     concatenation the real DurationEncoder does four times is a traced graph
    --     (`duration_style_concat`), each BiLSTM leaves `[h_fwd | h_bwd]` in its own store, and every
    --     edge here is a name. It used to rebuild a T x 640 table per stage -- the AdaLayerNorm's whole
    --     output pulled back so that STYLE_DIM more numbers could be appended to each of its rows
    --     (ADR-031's last open edge). ---
    local d_channels = D_MODEL + STYLE_DIM
    loom.run_subgraph_and_retain("duration_style_concat", {n_tokens = T_text, n_past = 0},
                                  {x = {from = 'albert_bert_encoder'}, style = s_predictor})
    -- The module holding DurationEncoder's real "d" at each point, not the values: one name travels
    -- through the loop and out of this block.
    local d = "duration_style_concat"
    for i = 0, 2 do
        local lstm = run_bi_lstm("duration_lstm_" .. i, {from = d}, T_text, d_channels, HIDDEN_PER_DIR)
        loom.run_subgraph_and_retain("duration_adaln_" .. i, {n_tokens = T_text, n_past = 0},
                                      {x = {from = lstm}, style = s_predictor})
        -- Re-running the same graph over its own last consumer's output: the BiLSTM above has already
        -- read what this overwrites, which is the whole reason one phase can serve all four stages.
        loom.run_subgraph_and_retain("duration_style_concat", {n_tokens = T_text, n_past = 0},
                                      {x = {from = "duration_adaln_" .. i}, style = s_predictor})
        d = "duration_style_concat"
    end

    -- --- predictor.lstm (top BiLSTM) -> duration_proj -> predict_durations ---
    -- ONE duration_proj call over the whole sequence, not one per token: `top_lstm`'s output is a
    -- retained tensor, so a per-row call would build and compute a graph per token to apply one Linear.
    -- Its result IS host-side -- the durations size every stage after this -- so this is the one call
    -- in the chain that still marshals.
    local top = run_bi_lstm("top_lstm", {from = d}, T_text, d_channels, HIDDEN_PER_DIR)
    local duration_logits = loom.run_subgraph("duration_proj", {n_tokens = T_text, n_past = 0},
                                               {x = {from = top}})
    local pred_dur = predict_durations(duration_logits, T_text, inputs.speed)
