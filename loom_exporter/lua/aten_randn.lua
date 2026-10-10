-- `torch.randn(n, generator=torch.Generator().manual_seed(seed))` on the CPU, float32: ATen's MT19937
-- seeded from the seed's low 32 bits, a 24-bit uniform per draw, and `normal_fill_16`'s Box-Muller in
-- blocks of 16 (the first 8 uniforms give the radius, the next 8 the angle). A size that is not a
-- multiple of 16 draws a FRESH block for its last 16 values, as ATen does. Sizes under 16 take ATen's
-- scalar path, which this does not reproduce, and are refused.
--
-- **Why a model needs ATen's stream rather than any Gaussian.** A noise-fed decoder renders a given seed
-- the same way in the reference only if the draw is the same draw (sanoTTS's nano voices default to a
-- fixed seed, and their Python and C runtimes both reproduce ATen's for exactly this reason). The
-- integers are exact; log/cos/sin are float32 libm calls in ATen, so a minority of values differ by a
-- few ulp -- upstream's own C runtime documents the same limit.
--
-- 32-bit arithmetic in doubles: LuaJIT's `bit` ops are signed 32-bit, and the seeding multiply needs
-- the low 32 bits of a 62-bit product, which a double cannot hold -- so it is done in 16-bit halves.
local function aten_randn(seed, n)
    if n < 16 then error("aten_randn: ATen's vectorised path starts at 16 values, got " .. n) end
    local bit = require("bit")
    local TWO32 = 4294967296.0
    local function u32(x) return x % TWO32 end
    local function mul32(a, b)
        local ah, al = math.floor(a / 65536), a % 65536
        return u32(((ah * b) % 65536) * 65536 + al * b)
    end
    local N, M = 624, 397
    local state = {}
    state[0] = u32(seed)
    for i = 1, N - 1 do
        local prev = state[i - 1]
        state[i] = u32(mul32(1812433253, bit.bxor(prev, math.floor(prev / 1073741824)) % TWO32) + i)
    end
    local left, nxt = 1, 0
    local function next_u32()
        left = left - 1
        if left <= 0 then
            for i = 0, N - 1 do
                local u, v = state[i], state[(i + 1) % N]
                local mixed = (u - u % 2147483648) + v % 2147483648
                local twist = math.floor(mixed / 2) % TWO32
                if v % 2 == 1 then twist = bit.bxor(twist, 0x9908B0DF) % TWO32 end
                state[i] = bit.bxor(state[(i + M) % N], twist) % TWO32
            end
            left, nxt = N, 0
        end
        local y = state[nxt]
        nxt = nxt + 1
        y = bit.bxor(y, math.floor(y / 2048)) % TWO32
        y = bit.bxor(y, bit.band(bit.lshift(y, 7), 0x9D2C5680)) % TWO32
        y = bit.bxor(y, bit.band(bit.lshift(y, 15), 0xEFC60000)) % TWO32
        y = bit.bxor(y, math.floor(y / 262144)) % TWO32
        return y
    end
    local function uniform() return (next_u32() % 16777216) / 16777216.0 end
    local TWO_PI = to_f32(2.0 * math.pi)
    local function fill16(block, base)
        for j = 0, 7 do
            local u1 = 1.0 - block[base + j]
            local u2 = block[base + j + 8]
            local radius = to_f32(math.sqrt(to_f32(-2.0 * to_f32(math.log(u1)))))
            local theta = to_f32(TWO_PI * u2)
            block[base + j] = to_f32(radius * to_f32(math.cos(theta)))
            block[base + j + 8] = to_f32(radius * to_f32(math.sin(theta)))
        end
    end
    local out = {}
    for i = 1, n do out[i] = uniform() end
    local whole = math.floor(n / 16) * 16
    for i = 1, whole, 16 do fill16(out, i) end
    if n % 16 ~= 0 then
        local tail = {}
        for i = 1, 16 do tail[i] = uniform() end
        fill16(tail, 1)
        for i = 1, 16 do out[n - 16 + i] = tail[i] end
    end
    return out
end
