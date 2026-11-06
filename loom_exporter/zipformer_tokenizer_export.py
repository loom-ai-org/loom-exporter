"""The vocabulary of a `pkufool/zipformer-*` checkpoint: `data/tokens.txt` (`piece id`) and, beside it, the
SentencePiece `.vocab` (`piece<TAB>score`). Neither is a protobuf or a `tokenizer.json`, and both state the
same pieces in the same order, so this reads them into the layout the SentencePiece writer takes.

The transcript is all that is ever decoded, so the scores only have to survive.
"""
from pathlib import Path

from gguf import GGUFWriter

from .spm_tokenizer_export import HfIdLayout, write_sentencepiece_vocab

_CONTROL = ("<blk>", "<sos>", "<eos>", "<pad>")


def read_zipformer_layout(model_dir) -> HfIdLayout:
    from .zipformer_export import _read_tokens

    data = Path(model_dir) / "data"
    pieces = _read_tokens(Path(model_dir))
    vocabs = sorted(data.glob("*.vocab"))
    if len(vocabs) != 1:
        raise ValueError(f"{data} should hold one .vocab beside tokens.txt, found {[v.name for v in vocabs]}")
    rows = [line.rsplit("\t", 1) for line in vocabs[0].read_text(encoding="utf-8").splitlines()]
    if [r[0] for r in rows] != pieces:
        raise ValueError(f"{vocabs[0].name} and tokens.txt state different pieces")
    return HfIdLayout(
        pieces=pieces, scores=[float(r[1]) for r in rows],
        unk_id=pieces.index("<unk>"), bos_id=pieces.index("<sos>"), eos_id=pieces.index("<eos>"),
        pad_id=pieces.index("<pad>"), special_ids=frozenset(pieces.index(p) for p in _CONTROL),
        model_type="bpe",
    )


def write_zipformer_vocab(writer: GGUFWriter, model_dir) -> None:
    write_sentencepiece_vocab(writer, None, hf_ids=read_zipformer_layout(model_dir))
