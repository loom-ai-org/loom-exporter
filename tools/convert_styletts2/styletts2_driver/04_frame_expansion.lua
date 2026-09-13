
    -- --- frame expansion: "en" (640ch, from d) and "asr" (512ch, from a SEPARATE plain TextEncoder) ---
    -- The counts are host-side and the sequences are not: `loom.expand_by_duration_and_retain` repeats
    -- each row in place, in the store the producer already wrote, so `en` and `asr` -- the two biggest
    -- tensors in this driver, T_frames rows each -- never become Lua tables. `en` stays one row per
    -- frame for the BiLSTM that reads it; `asr` is written in Layout A, which is what the vocoder's own
    -- `asr` input declares.
    local T_frames = array_sum(pred_dur)
    local en = d
    loom.expand_by_duration_and_retain(en, pred_dur)

    loom.run_subgraph_and_retain("text_encoder_cnn", {n_tokens = T_text, n_past = 0},
                                  {tokens = inputs.input_ids})
    -- The CNN's own output width, read without its data: it is a checkpoint property, and the driver
    -- would otherwise have to restate a number the graph already knows.
    local te_channels = loom.output_shape("text_encoder_cnn", 1)[1]
    local asr = run_bi_lstm("text_encoder_lstm", {from = "text_encoder_cnn"}, T_text, te_channels,
                             HIDDEN_PER_DIR)
    loom.expand_by_duration_and_retain(asr, pred_dur, "layout_a")
