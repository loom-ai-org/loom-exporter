-- Continuous integrate-and-fire, as the LINEAR RESAMPLING MATRIX it is.
--
-- Returns `{weights = <flat row-major (n_tokens x n_frames) array>, n_tokens = L}`. One table rather
-- than several return values because a synthesized driver binds one name per statement, and fields
-- need no new driver-IR node.
--
-- **Why a matrix and not a list of indices.** A CIF predictor emits a token each time a running sum of
-- `alphas` crosses an integer, and FunASR forms each token by DIFFERENCING two cumulative sums of the
-- weighted encoder frames. Written that way the graph needs a cumulative sum along its slow axis
-- (which `ggml_cumsum` does not do -- it only sums over `ne[0]`) and a row gather from the permuted
-- result (which `ggml_get_rows` cannot take, for the contiguity reason family 5's first leaf hit in
-- `ggml_repeat`). But every token is just a WEIGHTED SUM OF FRAMES whose weights depend on `alphas`
-- alone -- so the host, which already has `alphas`, can hand the graph the weights and let one matmul
-- do the rest. The graph then contains no cumulative sum, no gather, and no threshold.
--
-- It is also more accurate than the reference, which is not why it was chosen but is worth stating:
-- differencing two cumulative sums is catastrophic cancellation by construction, while this sums the
-- ~4 terms per row that actually contribute. Measured against an f64 evaluation of the same algebra,
-- FunASR's own f32 result is 7.18e-07 away and this is 4.43e-08 -- sixteen times closer.
--
-- **The boundary arithmetic is float32 and its ORDER is part of the contract.** FunASR accumulates at
-- float64, CASTS TO FLOAT32, then floors; and it computes the remainder as
-- `(indicator + prefix_sum) - floor(prefix_sum)` in float32, left to right. Both matter, and neither
-- is a rounding detail:
--   * accumulating in double and never rounding is MORE precise and puts tokens on different frames --
--     the crossings sit within ~2e-06 of an integer;
--   * adding the indicator to the FULL running total first is what loses precision. At a prefix sum
--     near 32 the float32 spacing is 3.8e-06, so `1 + 31.999998` lands on a representable midpoint,
--     rounds to 33, and the remainder collapses from ~1 to 0. Computing `1 + (ps - floor(ps))` -- the
--     same value in exact arithmetic -- gives ~1 and a visibly wrong acoustic embedding.
-- Getting either wrong does not raise. It moves the embeddings by ~2.6e-01 and leaves a transcript
-- that is still almost right.
local function cif_fire(alphas, threshold, n_frames)
    local idx, remain = {}, {}
    local ps, prev_floor = 0.0, 0.0
    for i = 1, #alphas do
        ps = ps + alphas[i]
        local ps32 = to_f32(ps)
        local cur_floor = math.floor(ps32)
        if cur_floor > prev_floor then
            local fired = to_f32(to_f32(threshold + ps32) - cur_floor)
            idx[#idx + 1] = i - 1
            remain[#remain + 1] = fired - math.floor(fired)
        end
        prev_floor = cur_floor
    end

    -- Row i spans the frames after the previous firing up to and including this one, weighted by
    -- `alphas`, plus the part of the previous firing frame this token inherits and minus the part of
    -- its own firing frame it leaves to the next. `alphas` is one longer than `n_frames` -- the tail
    -- frame FunASR appends -- and that last column is dropped rather than represented: the encoder
    -- state it would weight is the zero row appended beside it, so it contributes nothing.
    local n = #idx
    local weights = {}
    for k = 1, n * n_frames do weights[k] = 0.0 end
    local prev_f = -1
    for i = 1, n do
        local f = idx[i]
        local row = (i - 1) * n_frames
        for t = prev_f + 1, math.min(f, n_frames - 1) do
            weights[row + t + 1] = weights[row + t + 1] + alphas[t + 1]
        end
        if i > 1 and prev_f < n_frames then
            weights[row + prev_f + 1] = weights[row + prev_f + 1] + remain[i - 1]
        end
        if f < n_frames then
            weights[row + f + 1] = weights[row + f + 1] - remain[i]
        end
        prev_f = f
    end
    return {weights = weights, n_tokens = n}
end
