-- ===== EnCodec, live (loom.cpp ADR-104): windows round the convolutions, the LSTM's state carried. =====
--
-- `infer`'s chain -- `pre`, one `run_recurrent` per LSTM layer, `post` -- cut per push. The two graphs are
-- convolutions that are NOT causal (`use_causal_conv: false`), so each runs over a window with context
-- either side, as DAC's whole decoder does (00_window_stream.lua in codec_stream_driver): `pre`'s one
-- kernel-7 convolution needs `PRE_MARGIN` frames each side and `post`'s upsampling stack `POST_MARGIN`
-- (3 and 1 frames, both under 1e-15 from one call at f64). The LSTM between them is causal but cannot be
-- windowed -- 32 frames of left context still leave it O(1) from one call -- so its h and c are CARRIED:
-- `loom.run_recurrent`'s `carry` table continues each layer's sweep where the last push left it.
--
-- A frame goes through the LSTM once its `pre` row is final (`PRE_MARGIN` frames after it have arrived),
-- and its samples come back once `POST_MARGIN` LSTM rows follow it: the stream runs `PRE_MARGIN +
-- POST_MARGIN` frames (80 ms at 50 Hz) behind the codes. `stream_finish` takes the rest, the clip's real
-- end standing in for the context after it, as it does for one call.
--
-- State: the codes from `code_start` (the left context `pre` still needs, then the frames not yet through
-- the LSTM); each LSTM layer's carry as `h_l` and `c_l`; `post`'s inputs -- each LSTM row and its residual -- from
-- `post_start`; `lstm_done` frames through the LSTM and `emitted` frames handed back.

local function rows_of(flat, from, to)
    -- Rows [from, to) (0-based) of a time-major `[n, HIDDEN]` array.
    local out = {}
    for i = from * HIDDEN + 1, to * HIDDEN do out[#out + 1] = flat[i] end
    return out
end

local function append(dst, src)
    for i = 1, #src do dst[#dst + 1] = src[i] end
end

local function through_lstm(final_to)
    -- `pre` over the codes it has, then each LSTM layer over the rows whose `pre` output is final, frames
    -- [lstm_done, final_to): their rows and residuals join `post`'s inputs.
    local codes, code_start = loom.state('codes'), loom.state('code_start')
    local lstm_done = loom.state('lstm_done')
    if final_to <= lstm_done then return end
    local have = code_start + #codes / CODES_PER_FRAME
    local first = math.max(code_start, lstm_done - PRE_MARGIN)
    local window = {}
    for i = (first - code_start) * CODES_PER_FRAME + 1, #codes do window[#window + 1] = codes[i] end
    local pre = loom.run_subgraph('pre', {n_codes = have - first, n_past = 0}, {codes = window})
    local residual = rows_of(pre, lstm_done - first, final_to - first)
    local seq, n = residual, final_to - lstm_done
    for l = 1, N_LSTM_LAYERS do
        -- A state value is a flat array, so the carry's two halves are kept under names of their own and
        -- the table `run_recurrent` reads and writes is rebuilt per call; absent before the first push.
        local carry = {h = loom.state('h_' .. l), c = loom.state('c_' .. l)}
        seq = loom.run_recurrent('lstm_l' .. (l - 1) .. '_fwd', seq, n, HIDDEN, HIDDEN, false, carry)
        loom.set_state('h_' .. l, carry.h)
        loom.set_state('c_' .. l, carry.c)
    end
    local lstm_rows, res_rows = loom.state('lstm_rows'), loom.state('res_rows')
    append(lstm_rows, seq)
    append(res_rows, residual)
    loom.set_state('lstm_rows', lstm_rows)
    loom.set_state('res_rows', res_rows)
    loom.set_state('lstm_done', final_to)
    -- Keep only the left context the next `pre` window reaches back to.
    local keep_from = math.max(code_start, final_to - PRE_MARGIN)
    local kept = {}
    for i = (keep_from - code_start) * CODES_PER_FRAME + 1, #codes do kept[#kept + 1] = codes[i] end
    loom.set_state('codes', kept)
    loom.set_state('code_start', keep_from)
end

local function out_of_post(ready)
    -- `post` over the LSTM rows it has, with up to POST_MARGIN before frame `emitted`; the samples of
    -- frames [emitted, ready).
    local emitted, post_start = loom.state('emitted'), loom.state('post_start')
    if ready <= emitted then return {} end
    local lstm_rows, res_rows = loom.state('lstm_rows'), loom.state('res_rows')
    local done = post_start + #lstm_rows / HIDDEN
    local first = math.max(post_start, emitted - POST_MARGIN)
    local wave = loom.run_subgraph('post', {n_codes = done - first, n_past = 0},
                                   {lstm_out = rows_of(lstm_rows, first - post_start, done - post_start),
                                    residual = rows_of(res_rows, first - post_start, done - post_start)})
    local out = {}
    for i = (emitted - first) * SAMPLES_PER_FRAME + 1, (ready - first) * SAMPLES_PER_FRAME do out[#out + 1] = wave[i] end
    -- Drop the rows no later window reaches back to.
    local keep_from = math.max(post_start, ready - POST_MARGIN)
    loom.set_state('lstm_rows', rows_of(lstm_rows, keep_from - post_start, done - post_start))
    loom.set_state('res_rows', rows_of(res_rows, keep_from - post_start, done - post_start))
    loom.set_state('post_start', keep_from)
    loom.set_state('emitted', ready)
    return out
end

function stream_open(inputs)
    loom.set_state('codes', {})
    loom.set_state('code_start', 0)
    for l = 1, N_LSTM_LAYERS do
        loom.set_state('h_' .. l, nil)
        loom.set_state('c_' .. l, nil)
    end
    loom.set_state('lstm_rows', {})
    loom.set_state('res_rows', {})
    loom.set_state('post_start', 0)
    loom.set_state('lstm_done', 0)
    loom.set_state('emitted', 0)
    return {}
end

function stream_feed(inputs)
    local codes = inputs.codes or inputs.tokens or {}
    if #codes % CODES_PER_FRAME ~= 0 then
        error(string.format('encodec stream: %d ids is not a whole number of %d-id frames', #codes, CODES_PER_FRAME))
    end
    local pending = loom.state('codes')
    append(pending, codes)
    loom.set_state('codes', pending)
    local have = loom.state('code_start') + #pending / CODES_PER_FRAME
    through_lstm(have - PRE_MARGIN)
    return out_of_post(loom.state('lstm_done') - POST_MARGIN)
end

function stream_finish(inputs)
    local have = loom.state('code_start') + #loom.state('codes') / CODES_PER_FRAME
    through_lstm(have)
    return out_of_post(loom.state('lstm_done'))
end
