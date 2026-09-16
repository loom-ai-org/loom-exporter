-- Rounds a Lua number (an IEEE double) to what it would be as an IEEE **float32**, round-to-nearest,
-- ties-to-even -- using only double arithmetic, because that is all a driver has.
--
-- **This is not a precision nicety, it is how a reference is reproduced.** Family 5's CIF predictor
-- decides where a token fires by comparing `floor(prefix_sum)` between adjacent frames, and FunASR
-- computes that prefix sum at float64 and then CASTS IT TO FLOAT32 before the floor. A driver that
-- accumulates in double and never rounds is MORE accurate and gives different answers: the boundaries
-- are knife-edge (a measured one sat 1.9e-06 below an integer), so a fraction of an ulp moves a token
-- from one frame to the next. Being more precise than the reference is a way of being wrong about it.
--
-- `frexp` splits x into a significand in [0.5, 1) and an exponent; scaling by 2^24 puts float32's
-- 24-bit significand on the integers, where `round_half_to_even` is exactly the tie rule IEEE uses --
-- and ties are reachable here rather than theoretical, which is the whole reason that helper is called
-- instead of `math.floor(x + 0.5)`.
local function to_f32(x)
    if x == 0 or x ~= x or x == math.huge or x == -math.huge then return x end
    local m, e = math.frexp(x)
    return math.ldexp(round_half_to_even(m * 16777216.0) / 16777216.0, e)
end
