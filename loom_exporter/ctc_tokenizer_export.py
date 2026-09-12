"""Writes a `Wav2Vec2CTCTokenizer` directory's `vocab.json` into a GGUF, `tokenizer.ggml.model`="ctc".

**A new tag, and the question of whether it needed to be one is the interesting part.** A CTC
tokenizer is a flat id -> piece table with no merges, no scores, no normalizer and no segmentation
algorithm: the ids come out of a CTC head's argmax, never out of an encode, and turning them back into
text is "concatenate the pieces, then write a space wherever the word-delimiter piece appears". That is
close enough to `loom::Vocab`'s SentencePiece decode -- concatenate, then unescape U+2581 -- that the
whole family could have been shipped with no engine change at all, by rewriting the delimiter piece to
"▁" here and tagging the file "t5".

It is not done that way because the rewrite would be a lie the artifact tells about itself. `vocab.json`
says piece 4 is `|`; a "t5" file saying it is `▁` would decode correctly and answer `id_to_piece(4)`
with a character the checkpoint has never heard of, and it would claim a segmentation algorithm
(Viterbi over unigram scores, against a normalizer that does not exist here) that nothing in the file
supports. The per-family-tag convention every other vocabulary in this schema follows -- "t5"/"llama",
"gpt2", "bert", "byt5", "phonemes", "supertonic" -- exists precisely so that a reader can tell which
scheme it is looking at, and the cost of honouring it here is one small decode-only C++ class.

Three KVs beyond the token list, and each is a number this family cannot derive:

* `tokenizer.ggml.padding_token_id` -- the CTC blank, which is `pad_token`'s id. NOT the last class
  (family 1's NeMo convention) and not always id 0's `<pad>` spelling; see `ctc_asr_export`.
* `tokenizer.ggml.word_delimiter_id` -- the piece that decodes as a space. `|` in the English
  checkpoints, a literal space in the multilingual one.
* `tokenizer.ggml.unknown_token_id` -- the usual role, the usual KV.

Requires: pip install gguf
"""
from gguf import GGUFWriter

from .ctc_asr_export import read_ctc_vocab

# gguf's own `TokenType` values. NORMAL for a real piece, CONTROL for the four roles a CTC tokenizer
# names -- the same two values every other writer in this package uses, and the array is written for
# the same reason: it is what tells a reader which ids are text and which are bookkeeping.
_TOKEN_TYPE_NORMAL = 1
_TOKEN_TYPE_CONTROL = 3


def write_ctc_vocab(writer: GGUFWriter, tokenizer_dir: str) -> None:
    vocab = read_ctc_vocab(tokenizer_dir)
    pieces = vocab["pieces"]

    # The control set is the three named roles plus `bos`/`eos` where the checkpoint declares them.
    # The word delimiter is deliberately NOT in it: it is a piece that decodes to real text (a space),
    # and marking it control would make it the one character a transcript is missing.
    control_ids = {vocab["blank_id"], vocab["unk_id"]}
    token_type = [_TOKEN_TYPE_CONTROL if i in control_ids else _TOKEN_TYPE_NORMAL
                  for i in range(len(pieces))]

    writer.add_tokenizer_model("ctc")
    writer.add_token_list(pieces)
    writer.add_token_types(token_type)
    writer.add_pad_token_id(vocab["blank_id"])
    writer.add_unk_token_id(vocab["unk_id"])
    # A project-specific key, like `tokenizer.ggml.byte_offset` before it: llama.cpp's schema has no
    # concept of a word delimiter because no vocabulary type it writes has one.
    writer.add_uint32("tokenizer.ggml.word_delimiter_id", vocab["word_delimiter_id"])
