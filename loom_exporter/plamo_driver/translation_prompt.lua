-- ===== PLaMo 3 translate: the PFML3 prompt, as the checkpoint's chat template renders it. =====
--
-- `translation_prompt(inputs)` -> the prompt's ids, for a host to generate from (loom-py
-- `text2text.translate`, loom.cpp ADR-132). It runs no graph. The template is
--
--   <|plamo:bos|><|plamo:tag|>dataset<|plamo:msg|>translation
--   [<|plamo:tag|>context<|plamo:msg|>CONTEXT]
--   ( <|plamo:tag|>input[<|plamo:key|>lang<|plamo:val|>SRC]<|plamo:msg|>SOURCE
--     <|plamo:tag|>output<|plamo:key|>lang<|plamo:val|>TGT<|plamo:msg|>TRANSLATION )*   (history)
--   [<|plamo:tag|>glossary(<|plamo:index|>TERM<|plamo:item|>RENDERING)*]
--   <|plamo:tag|>input[<|plamo:key|>lang<|plamo:val|>SRC]<|plamo:msg|>TEXT
--   <|plamo:tag|>output<|plamo:key|>lang<|plamo:val|>TGT[<|plamo:key|>form<|plamo:val|>FORM]<|plamo:msg|>
--
-- and its ids are the ids of these pieces end to end: the tokenizer cuts at every `<|plamo:...|>` tag,
-- so no piece crosses one, and a piece's ids are its own. The host hands each caller string already
-- encoded (`text`, `target_lang`, `source_lang`, `context`, `form`; `glossary` and `history` as one id
-- array plus `*_offsets`, two strings per entry); the fixed words are encoded here.
--
-- The template's refusals are kept: an empty text, a language code that is not `xx`, `xxx` or with a
-- four-letter script (`zh-hans`, `ja-easy`; `ja(easy)` is its alias), a form other than polite/plain or
-- with a target other than ja, and a reserved `<|plamo:` inside any string.

local function plamo_piece(ids, offsets, k)
    local out = {}
    for i = offsets[k] + 1, offsets[k + 1] do out[#out + 1] = ids[i] end
    return out
end

local function plamo_text(ids, name)
    local s = loom.detokenize(ids)
    if string.find(s, '<|plamo:', 1, true) then
        error('plamo: ' .. name .. ' must not contain reserved PFML tokens (<|plamo:...)')
    end
    return s
end

local function plamo_lang(ids, name)
    local s = plamo_text(ids, name)
    if s == 'ja(easy)' then s = 'ja-easy' end
    local base, script = string.match(s, '^([a-z]+)%-([a-z]+)$')
    if base == nil then base, script = s, nil end
    if not string.match(base, '^[a-z]+$') or (#base ~= 2 and #base ~= 3) or (script ~= nil and #script ~= 4) then
        error('plamo: ' .. name .. " must be a lowercase language code, e.g. en, ja, zh-hans or ja-easy; got '" ..
              s .. "'")
    end
    return s
end

function translation_prompt(inputs)
    local out = {}
    local function add(ids) for _, id in ipairs(ids) do out[#out + 1] = id end end
    local function tag(name) out[#out + 1] = PLAMO_TAG_IDS[name] end
    local function word(s) add(loom.tokenize(s)) end

    if inputs.text == nil or #inputs.text == 0 then error('plamo: the text to translate is empty') end
    local text = plamo_text(inputs.text, 'text')
    if not string.match(text, '%S') then error('plamo: the text to translate is empty') end
    if inputs.target_lang == nil then error('plamo: target_lang is required, e.g. "ja"') end
    local target = plamo_lang(inputs.target_lang, 'target_lang')
    local source = inputs.source_lang and plamo_lang(inputs.source_lang, 'source_lang') or nil
    local form = nil
    if inputs.form ~= nil then
        form = plamo_text(inputs.form, 'form')
        if (form ~= 'polite' and form ~= 'plain') or target ~= 'ja' then
            error('plamo: form must be polite or plain and needs target_lang "ja"')
        end
    end

    tag('bos') tag('tag') word('dataset') tag('msg') word('translation')
    if inputs.context ~= nil and #inputs.context > 0 then
        plamo_text(inputs.context, 'context')
        tag('tag') word('context') tag('msg') add(inputs.context)
    end
    local function input_head()
        tag('tag') word('input')
        if source ~= nil then tag('key') word('lang') tag('val') word(source) end
        tag('msg')
    end
    local function output_head()
        tag('tag') word('output') tag('key') word('lang') tag('val') word(target)
    end
    if inputs.history ~= nil then
        local offsets = inputs.history_offsets
        if offsets == nil or (#offsets - 1) % 2 ~= 0 then
            error('plamo: history is (source, translation) pairs: history_offsets must mark an even number')
        end
        for k = 1, #offsets - 1, 2 do
            local src, tgt = plamo_piece(inputs.history, offsets, k), plamo_piece(inputs.history, offsets, k + 1)
            plamo_text(src, 'history') plamo_text(tgt, 'history')
            input_head() add(src)
            output_head() tag('msg') add(tgt)
        end
    end
    if inputs.glossary ~= nil then
        local offsets = inputs.glossary_offsets
        if offsets == nil or (#offsets - 1) % 2 ~= 0 then
            error('plamo: glossary is (source, target) pairs: glossary_offsets must mark an even number')
        end
        if #offsets > 1 then
            tag('tag') word('glossary')
            for k = 1, #offsets - 1, 2 do
                local src, tgt = plamo_piece(inputs.glossary, offsets, k), plamo_piece(inputs.glossary, offsets, k + 1)
                plamo_text(src, 'glossary source') plamo_text(tgt, 'glossary target')
                tag('index') add(src) tag('item') add(tgt)
            end
        end
    end
    input_head() add(inputs.text)
    output_head()
    if form ~= nil then tag('key') word('form') tag('val') word(form) end
    tag('msg')
    return out
end
