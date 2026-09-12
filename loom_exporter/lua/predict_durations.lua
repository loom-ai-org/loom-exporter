-- Per-token integer durations from `duration_proj`'s logit rows: `max(round(sum(sigmoid(row)) / speed), 1)`.
--
-- `logits_flat` is the whole `(n, max_dur)` block one `duration_proj` call returns, row-major -- it was
-- a table of per-token tables while the driver called that topology once per timestep, which it stopped
-- doing when the BiLSTM above it stopped marshalling (ADR-031's follow-up). The width is derived rather
-- than passed: there is exactly one row per token, so a caller that got `n` right cannot get it wrong.
local function predict_durations(logits_flat, n, speed)
    local max_dur = #logits_flat / n
    local pred_dur = {}
    for t = 0, n - 1 do
        local sum = 0.0
        for k = 1, max_dur do sum = sum + sigmoid(logits_flat[t * max_dur + k]) end
        local rounded = round_half_to_even(sum / speed)
        pred_dur[t + 1] = math.max(rounded, 1)
    end
    return pred_dur
end
