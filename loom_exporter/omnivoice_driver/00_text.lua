-- ===== OmniVoice: the reference's text handling, ported (omnivoice/models/omnivoice.py, utils/). =====
--
-- Everything here works on CODEPOINTS, because the reference does: `len(text)` (the chunker's and the
-- duration estimate's unit), `list(text)` (the chunker's walk) and its regexes all count characters, not
-- bytes. LuaJIT has no utf8 library, so decoding and encoding are the two functions at the top.

local function omni_codepoints(s)
    local out, i, n = {}, 1, #s
    while i <= n do
        local c = s:byte(i)
        local cp, len
        if c < 0x80 then cp, len = c, 1
        elseif c < 0xE0 then cp, len = c % 0x20, 2
        elseif c < 0xF0 then cp, len = c % 0x10, 3
        else cp, len = c % 0x08, 4 end
        for k = 1, len - 1 do cp = cp * 64 + ((s:byte(i + k) or 0x80) % 0x40) end
        out[#out + 1] = cp
        i = i + len
    end
    return out
end

local function omni_utf8(cps, first, last)
    local parts = {}
    for i = first or 1, last or #cps do
        local cp = cps[i]
        if cp < 0x80 then parts[#parts + 1] = string.char(cp)
        elseif cp < 0x800 then
            parts[#parts + 1] = string.char(0xC0 + math.floor(cp / 64), 0x80 + cp % 64)
        elseif cp < 0x10000 then
            parts[#parts + 1] = string.char(0xE0 + math.floor(cp / 4096), 0x80 + math.floor(cp / 64) % 64,
                                            0x80 + cp % 64)
        else
            parts[#parts + 1] = string.char(0xF0 + math.floor(cp / 262144), 0x80 + math.floor(cp / 4096) % 64,
                                            0x80 + math.floor(cp / 64) % 64, 0x80 + cp % 64)
        end
    end
    return table.concat(parts)
end

-- A list of UTF-8 byte strings shipped as `(offsets, bytes)` number arrays -- ExportConstants binds
-- numbers only -- back into Lua strings.
local function omni_strings(offsets, bytes)
    local out = {}
    for i = 1, #offsets - 1 do
        local chars = {}
        for b = offsets[i] + 1, offsets[i + 1] do chars[#chars + 1] = string.char(bytes[b]) end
        out[i] = table.concat(chars)
    end
    return out
end

-- Python's `str.isspace()` / regex `\s` for str: ASCII whitespace plus the Unicode separators.
local function omni_is_space(cp)
    return (cp >= 9 and cp <= 13) or cp == 32 or (cp >= 0x1C and cp <= 0x1F) or cp == 0x85 or cp == 0xA0
        or cp == 0x1680 or (cp >= 0x2000 and cp <= 0x200A) or cp == 0x2028 or cp == 0x2029 or cp == 0x202F
        or cp == 0x205F or cp == 0x3000
end

local function omni_is_cjk(cp) return cp >= 0x4E00 and cp <= 0x9FFF end

local function omni_strip(cps)
    local a, b = 1, #cps
    while a <= b and omni_is_space(cps[a]) do a = a + 1 end
    while b >= a and omni_is_space(cps[b]) do b = b - 1 end
    local out = {}
    for i = a, b do out[#out + 1] = cps[i] end
    return out
end

-- `_combine_text`: the reference transcript and the text joined by a space, then four rewrites in the
-- reference's order -- newlines dropped, full-width parentheses to ASCII, runs of spaces/tabs to one
-- space, and whitespace beside a CJK ideograph removed.
local function omni_combine_text(text_cps, ref_cps)
    local cps = {}
    local text = omni_strip(text_cps)
    if ref_cps ~= nil and #ref_cps > 0 then
        for _, c in ipairs(omni_strip(ref_cps)) do cps[#cps + 1] = c end
        cps[#cps + 1] = 32
    end
    for _, c in ipairs(text) do cps[#cps + 1] = c end
    local a = {}
    for _, c in ipairs(cps) do
        if c ~= 10 and c ~= 13 then
            if c == 0xFF08 then c = 40 elseif c == 0xFF09 then c = 41 end
            a[#a + 1] = c
        end
    end
    local b = {}
    for i, c in ipairs(a) do
        if c == 32 or c == 9 then
            if not (#b > 0 and b[#b] == 32 and (a[i - 1] == 32 or a[i - 1] == 9)) then b[#b + 1] = 32 end
        else
            b[#b + 1] = c
        end
    end
    -- `(?<=cjk)\s+|\s+(?=cjk)`: a whitespace RUN goes when the character before it or after it is CJK.
    local out, i = {}, 1
    while i <= #b do
        if omni_is_space(b[i]) then
            local j = i
            while j + 1 <= #b and omni_is_space(b[j + 1]) do j = j + 1 end
            local before, after = b[i - 1], b[j + 1]
            if not ((before ~= nil and omni_is_cjk(before)) or (after ~= nil and omni_is_cjk(after))) then
                for k = i, j do out[#out + 1] = b[k] end
            end
            i = j + 1
        else
            out[#out + 1] = b[i]
            i = i + 1
        end
    end
    return out
end

-- `RuleDurationEstimator._get_char_weight`, from the export's run table: `WEIGHT_STARTS[i]` is the
-- first codepoint of a run whose weight is `WEIGHT_VALUES[i]` -- unicodedata's categories and the
-- script ranges folded together at export time, since a driver has no unicodedata.
local function omni_char_weight(cp)
    local lo, hi = 1, #WEIGHT_STARTS
    while lo < hi do
        local mid = math.floor((lo + hi + 1) / 2)
        if WEIGHT_STARTS[mid] <= cp then lo = mid else hi = mid - 1 end
    end
    return WEIGHT_VALUES[lo]
end

local function omni_weight(cps)
    local total = 0.0
    for _, c in ipairs(cps) do total = total + omni_char_weight(c) end
    return total
end

-- `estimate_duration` with its defaults (low_threshold 50, boost_strength 3), then
-- `_estimate_target_tokens`: the fallback reference, the speed, `max(1, int(est))`.
local function omni_target_tokens(text_cps, ref_cps, ref_frames, speed)
    if ref_frames == nil or ref_cps == nil or #ref_cps == 0 then
        ref_cps, ref_frames = FALLBACK_REF_TEXT, FALLBACK_REF_FRAMES
    end
    local est = 0.0
    local ref_weight = omni_weight(ref_cps)
    if ref_frames > 0 and ref_weight ~= 0 then
        est = omni_weight(text_cps) / (ref_weight / ref_frames)
        if est < 50 then est = 50 * (est / 50) ^ (1.0 / 3) end
    end
    if speed > 0 and speed ~= 1.0 then est = est / speed end
    local n = est >= 0 and math.floor(est) or math.ceil(est)
    if n < 1 then n = 1 end
    return n
end

local function omni_set(list)
    local s = {}
    for _, c in ipairs(list) do s[c] = true end
    return s
end

-- `chunk_text_punctuation(text, chunk_len, min_chunk_len=3)`: split after sentence punctuation (not
-- after an abbreviation's full stop), glue leading punctuation and closing marks onto the previous
-- sentence, merge sentences up to `chunk_len` characters, then fold undersized chunks into a neighbour.
local function omni_chunk_text(cps, chunk_len, min_chunk_len)
    local split, closing = omni_set(SPLIT_PUNCTUATION), omni_set(CLOSING_MARKS)
    local abbreviations = omni_set(omni_strings(ABBREVIATION_OFFSETS, ABBREVIATION_BYTES))
    local sentences, current = {}, {}
    for _, c in ipairs(cps) do
        if #current == 0 and #sentences ~= 0 and (split[c] or closing[c]) then
            local last = sentences[#sentences]
            last[#last + 1] = c
        else
            current[#current + 1] = c
            if split[c] then
                local abbreviation = false
                if c == 46 then
                    local stripped = omni_strip(current)
                    if #stripped > 0 then
                        local start = #stripped
                        while start > 1 and not omni_is_space(stripped[start - 1]) do start = start - 1 end
                        if abbreviations[omni_utf8(stripped, start, #stripped)] then abbreviation = true end
                    end
                end
                if not abbreviation then
                    sentences[#sentences + 1] = current
                    current = {}
                end
            end
        end
    end
    if #current ~= 0 then sentences[#sentences + 1] = current end

    local merged, chunk = {}, {}
    for _, sentence in ipairs(sentences) do
        if #chunk + #sentence <= chunk_len then
            for _, c in ipairs(sentence) do chunk[#chunk + 1] = c end
        else
            if #chunk > 0 then merged[#merged + 1] = chunk end
            chunk = {}
            for _, c in ipairs(sentence) do chunk[#chunk + 1] = c end
        end
    end
    if #chunk > 0 then merged[#merged + 1] = chunk end

    local final = {}
    local first_short = #merged > 0 and #merged[1] < min_chunk_len
    for i, c in ipairs(merged) do
        if i == 2 and first_short then
            for _, x in ipairs(c) do final[#final][#final[#final] + 1] = x end
        elseif #c >= min_chunk_len then
            final[#final + 1] = c
        elseif #final == 0 then
            final[#final + 1] = c
        else
            for _, x in ipairs(c) do final[#final][#final[#final] + 1] = x end
        end
    end
    local out = {}
    for _, c in ipairs(final) do
        local stripped = omni_strip(c)
        if #stripped > 0 then out[#out + 1] = stripped end
    end
    return out
end

-- `add_punctuation`: a reference transcript that does not end in punctuation gets a full stop (an
-- ideographic one when it contains CJK).
local function omni_add_punctuation(cps)
    local text = omni_strip(cps)
    if #text == 0 then return text end
    local tail = omni_set(END_PUNCTUATION)
    if not tail[text[#text]] then
        local cjk = false
        for _, c in ipairs(text) do if omni_is_cjk(c) then cjk = true end end
        text[#text + 1] = cjk and 0x3002 or 46
    end
    return text
end

local function omni_append(dst, src) for _, v in ipairs(src) do dst[#dst + 1] = v end end

-- `_tokenize_with_nonverbal_tags`: each `[laughter]`-style tag is tokenized on its own, so its ids do
-- not depend on the language around it. The wrapped text is `<|text_start|>...<|text_end|>`; HF splits
-- the special tokens out first, so a segment's ids are its plain text's.
local function omni_text_ids(cps)
    local tags = omni_set(omni_strings(TAG_OFFSETS, TAG_BYTES))
    local ids = {TEXT_START_ID}
    local seg_start, i = 1, 1
    while i <= #cps do
        local matched = nil
        if cps[i] == 91 then
            local j = i + 1
            while j <= #cps and cps[j] ~= 93 and j - i <= 40 do j = j + 1 end
            if j <= #cps and cps[j] == 93 and tags[omni_utf8(cps, i, j)] then matched = j end
        end
        if matched then
            if i > seg_start then omni_append(ids, loom.tokenize(omni_utf8(cps, seg_start, i - 1))) end
            omni_append(ids, loom.tokenize(omni_utf8(cps, i, matched)))
            seg_start = matched + 1
            i = matched + 1
        else
            i = i + 1
        end
    end
    if seg_start <= #cps then omni_append(ids, loom.tokenize(omni_utf8(cps, seg_start, #cps))) end
    ids[#ids + 1] = TEXT_END_ID
    return ids
end

-- `_resolve_instruct`: comma-separated voice-design items, validated against the model's list, then
-- unified to one language -- Chinese when the TEXT has CJK (or a dialect is named), English when an
-- accent is named -- and rejoined with that language's separator.
local function omni_resolve_instruct(instruct, use_zh)
    local en = omni_strings(INSTRUCT_EN_OFFSETS, INSTRUCT_EN_BYTES)
    local zh = omni_strings(INSTRUCT_ZH_OFFSETS, INSTRUCT_ZH_BYTES)
    local accents = omni_set(omni_strings(ACCENT_OFFSETS, ACCENT_BYTES))
    local dialects = omni_set(omni_strings(DIALECT_OFFSETS, DIALECT_BYTES))
    local en_to_zh, zh_to_en, category, valid = {}, {}, {}, {}
    for i = 1, #en do
        en_to_zh[en[i]], zh_to_en[zh[i]] = zh[i], en[i]
        category[en[i]], category[zh[i]] = INSTRUCT_CATEGORY[i], INSTRUCT_CATEGORY[i]
        valid[en[i]], valid[zh[i]] = true, true
    end
    for a in pairs(accents) do valid[a], category[a] = true, 100 end
    for d in pairs(dialects) do valid[d], category[d] = true, 101 end

    -- Split on ',' and '，' with the whitespace around them; lowercase (ASCII, as Python's lower() is a
    -- no-op on the CJK items).
    local cps = omni_codepoints(instruct)
    local items, current = {}, {}
    local function flush()
        local s = omni_strip(current)
        if #s > 0 then items[#items + 1] = string.lower(omni_utf8(s)) end
        current = {}
    end
    for _, c in ipairs(cps) do
        if c == 44 or c == 0xFF0C then flush() else current[#current + 1] = c end
    end
    flush()
    if #items == 0 then return nil end
    local unknown = {}
    for _, item in ipairs(items) do
        if not valid[item] then unknown[#unknown + 1] = "'" .. item .. "'" end
    end
    if #unknown > 0 then
        local names = {}
        for name in pairs(valid) do names[#names + 1] = name end
        table.sort(names)
        error('omnivoice: unsupported instruct item(s) ' .. table.concat(unknown, ', ') ..
              '; valid items: ' .. table.concat(names, ', '))
    end
    local has_dialect, has_accent = false, false
    for _, item in ipairs(items) do
        if dialects[item] then has_dialect = true end
        if accents[item] then has_accent = true end
    end
    if has_dialect and has_accent then
        error('omnivoice: a Chinese dialect and an English accent cannot be mixed in one instruct')
    end
    if has_dialect then use_zh = true elseif has_accent then use_zh = false end
    for i, item in ipairs(items) do
        if use_zh then items[i] = en_to_zh[item] or item else items[i] = zh_to_en[item] or item end
    end
    local seen = {}
    for _, item in ipairs(items) do
        local cat = category[item]
        if seen[cat] then error('omnivoice: conflicting instruct items in one category: ' .. seen[cat] ..
                                ' vs ' .. item) end
        seen[cat] = item
    end
    local any_zh = false
    for _, item in ipairs(items) do
        for _, c in ipairs(omni_codepoints(item)) do if omni_is_cjk(c) then any_zh = true end end
    end
    return table.concat(items, any_zh and '，' or ', ')
end
