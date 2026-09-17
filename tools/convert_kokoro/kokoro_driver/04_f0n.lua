
    -- --- F0Ntrain: shared BiLSTM -> F0/N AdainResBlk1d stacks -> projections ---
    -- Layout A out of the BiLSTM, because that is what an AdainResBlk1d declares: nothing converts
    -- between the two conventions any more, the producer is told which one its consumer wants.
    local shared_out = run_bi_lstm("f0n_shared_lstm", {from = en}, T_frames, D_MODEL + STYLE_DIM,
                                    HIDDEN_PER_DIR, "layout_a")
    local f0_feat = run_resblk_stack("f0n_f0", shared_out, s_predictor)
    local n_feat = run_resblk_stack("f0n_n", shared_out, s_predictor)

    -- `f0_feat`/`n_feat` are module NAMES, and so are these: from the shared BiLSTM to the vocoder,
    -- this branch is entirely references.
    local F0_curve = run_proj1x1("f0n_f0_proj", f0_feat)
    local N_curve = run_proj1x1("f0n_n_proj", n_feat)

