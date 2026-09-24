-- Pocket-TTS's helpers. Top level, so every fragment below can call them.

-- `len(text.split())` of a prepared chunk, read off its ids: a word is a piece that opens with
-- SentencePiece's `▁`. `word_start` is the export's per-id flag for that (a driver weight, read through
-- `step_embed`, which is only a name here). A LONE `▁` piece is a word only when the next piece does
-- not open another one -- two spaces in a row reach the tokenizer as `▁`, `▁word`, one word.
local function pocket_count_words(tokens, starts)
    local n = 0
    for i = 1, #tokens do
        local flag = starts[tokens[i] + 1]
        if flag == 1 then
            n = n + 1
        elseif flag == 2 then
            local next_id = tokens[i + 1]
            if next_id == nil or starts[next_id + 1] == 0 then n = n + 1 end
        end
    end
    return n
end

-- The vocabulary's chunks, each with the tail `prepare_text_prompt` guessed for it. `loom::PocketTtsVocab`
-- opens every chunk with a header id -- `short_id` for a chunk of at most four words, `long_id` otherwise
-- -- because that guess is `len(text.split())` of TEXT, which the driver never sees (loom.cpp ADR-044).
-- A host that tokenizes elsewhere sends no header: one chunk, its words counted off the ids, which is
-- exact for ASCII spaces and misses words separated by a tab or an NBSP.
local function pocket_split_chunks(tokens, short_id, long_id, starts, max_words)
    local chunks, current = {}, nil
    for i = 1, #tokens do
        local id = tokens[i]
        if id == short_id or id == long_id then
            current = {ids = {}, short = (id == short_id)}
            chunks[#chunks + 1] = current
        else
            if current == nil then
                current = {ids = {}}
                chunks[#chunks + 1] = current
            end
            current.ids[#current.ids + 1] = id
        end
    end
    local kept = {}
    for _, chunk in ipairs(chunks) do
        if #chunk.ids > 0 then
            if chunk.short == nil then
                chunk.short = pocket_count_words(chunk.ids, starts) <= max_words
            end
            kept[#kept + 1] = chunk
        end
    end
    return kept
end
