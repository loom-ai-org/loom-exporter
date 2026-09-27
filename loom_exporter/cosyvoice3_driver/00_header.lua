-- CosyVoice3's helpers. Top level, so every fragment below can call them.

-- A voice tensor: the caller's when it passes one, else the default voice the export computed and
-- shipped as a driver weight. Read through `lm`, which is only a name here -- driver weights live in
-- the one namespace every module reads.
local function cosyvoice3_voice(override, name)
    if override ~= nil then return override end
    return loom.get_weight('lm', name)
end

-- `1 - cos(pi/2 * t)` over `n_steps + 1` points of a linspace: `CausalConditionalCFM.forward`'s cosine
-- schedule, which is why the sampler takes the CALLER's times.
local function cosyvoice3_cosine_times(n_steps)
    local times = {}
    for step = 0, n_steps do
        times[step + 1] = 1 - math.cos(step / n_steps * 0.5 * math.pi)
    end
    return times
end

local function cosyvoice3_zeros(n)
    local z = {}
    for i = 1, n do z[i] = 0.0 end
    return z
end

-- `CosyVoice3Model.silent_tokens`: the FSQ codes for silence and breath, as a set.
local COSYVOICE3_SILENT = {}
for _, id in ipairs({1, 2, 28, 29, 55, 248, 494, 2241, 2242, 2322, 2323}) do COSYVOICE3_SILENT[id] = true end

-- `loom::CosyVoice3Vocab::encode`'s chunks: the ids after each `header`, one array per chunk. An empty
-- chunk is not something the vocabulary emits, so it is refused rather than run as an empty text.
local function cosyvoice3_split_chunks(ids, header)
    local chunks = {}
    for i = 1, #ids do
        if ids[i] == header then
            chunks[#chunks + 1] = {}
        else
            local chunk = chunks[#chunks]
            chunk[#chunk + 1] = ids[i]
        end
    end
    for c = 1, #chunks do
        if #chunks[c] == 0 then error('cosyvoice3: chunk ' .. c .. ' of the ids is empty') end
    end
    return chunks
end
