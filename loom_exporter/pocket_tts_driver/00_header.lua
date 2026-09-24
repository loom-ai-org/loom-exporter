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

-- The vocabulary's chunks: `tokens` split on `separator`, the id `loom::PocketTtsVocab` puts between
-- the sentence chunks it cut (the reference generates each from a fresh copy of the voice). A host that
-- tokenizes elsewhere sends none, which is one chunk.
local function pocket_split_chunks(tokens, separator)
    local chunks, current = {}, {}
    for i = 1, #tokens do
        if tokens[i] == separator then
            if #current > 0 then chunks[#chunks + 1] = current end
            current = {}
        else
            current[#current + 1] = tokens[i]
        end
    end
    if #current > 0 then chunks[#chunks + 1] = current end
    return chunks
end
