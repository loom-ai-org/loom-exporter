-- T5: a text encoder run once, then a KV-cached cross-attention decode loop.
--
-- Three traced topologies (t5_export.py): `encoder` turns the caller's source tokens into one hidden
-- state each; `cross_kv` projects those into every decoder layer's cross-attention K/V, once per call;
-- `decoder` is one cached step, called at n_tokens = 1 throughout (its prompt is the single
-- `decoder_start_token_id`, so there is no multi-token prefill to amortise).
--
-- inputs: tokens (source token ids from this model's own SentencePiece vocabulary -- for flan-t5 that
-- means the instruction is part of the text, e.g. "translate English to German: ..."), plus
-- max_new_tokens and eos_token. Returns the generated ids, not including the start token.
--
-- **What is T5's rather than generic is entirely below: the relative attention bias.** T5 has no
-- positional embedding at all. Every attention score is offset by a learned value chosen by the
-- BUCKETED distance between the query and the key, and the same tensor is what carries the causal
-- mask, because HF sums the two before any layer sees them. So the mask this driver hands the decoder
-- is bias-plus-mask, and building it is the reason these functions exist. The table it reads is the
-- checkpoint's own `relative_attention_bias.weight`, flattened `bucket * n_head + head` and bound as
-- an ExportConstants local -- 192 floats for flan-t5-small.

-- `_relative_position_bucket` (modeling_t5.py), in Lua. `rel` is key - query, i.e. NEGATIVE for a key
-- in the past. Half the buckets are exact small distances and half are log-spaced up to
-- `max_distance`; a bidirectional stack splits the range in two and encodes the sign in the upper
-- half, a causal one has only the past and uses the whole range for it.
local function t5_bucket(rel, bidirectional, num_buckets, max_distance)
  local bucket = 0
  local n = num_buckets
  if bidirectional then
    n = math.floor(num_buckets / 2)
    if rel > 0 then bucket = n end
    if rel < 0 then rel = -rel end
  else
    -- `-min(rel, 0)`: a causal stack never sees a positive relative position, and the masked cells
    -- this still evaluates (see below) fold onto bucket 0 rather than off the end of the table.
    if rel > 0 then rel = 0 end
    rel = -rel
  end
  local max_exact = math.floor(n / 2)
  if rel < max_exact then
    return bucket + rel
  end
  -- `.to(torch.long)` in the reference, which TRUNCATES toward zero -- the same as floor here, since
  -- rel >= max_exact makes the logarithm non-negative.
  local large = max_exact + math.floor(
    math.log(rel / max_exact) / math.log(max_distance / max_exact) * (n - max_exact))
  if large > n - 1 then large = n - 1 end
  return bucket + large
end

-- The `[1, n_head, n_tokens, n_kv]` additive tensor a T5 stack adds to its scores, flattened in the
-- engine's own ne-order: key fastest, then query, then head. That is the layout `loom.causal_mask`
-- already returns one head of, and it is what `ggml_soft_max_ext` reads -- its mask may carry a head
-- axis, which is what lets a per-head bias be a mask at all.
--
-- `n_past` is the cache extent, so a decode step at n_tokens = 1 still builds a full n_kv-wide row:
-- query i sits at absolute position n_past + i, exactly as `loom.causal_mask` places it.
--
-- Bucketed ONCE per (query, key) pair and then read n_head times, rather than re-bucketing per head:
-- the logarithm is the expensive part and it does not depend on the head.
function t5_position_bias(tab, n_head, num_buckets, max_distance, n_tokens, n_past, bidirectional)
  local bidir = bidirectional ~= 0
  local n_kv = n_tokens + n_past
  local rows = n_tokens * n_kv
  local base = {}
  local blocked = {}
  local w = 1
  for i = 0, n_tokens - 1 do
    local query = n_past + i
    for j = 0, n_kv - 1 do
      base[w] = t5_bucket(j - query, bidir, num_buckets, max_distance) * n_head
      -- A causal stack's mask, folded into the same array. A bidirectional one attends over
      -- everything it is given, so there is nothing to block.
      blocked[w] = (not bidir) and j > query
      w = w + 1
    end
  end
  local out = {}
  local o = 1
  for h = 1, n_head do
    for k = 1, rows do
      if blocked[k] then
        out[o] = -math.huge
      else
        out[o] = tab[base[k] + h]
      end
      o = o + 1
    end
  end
  return out
end
