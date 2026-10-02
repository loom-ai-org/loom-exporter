"""LFM2.5-Audio's speech-to-text door (P5): family 3's shape over NeMo's FastConformer and an LFM2 hybrid.

What is tested here is what the engine reads as DATA or what the trace re-spells:

* the recognizer, and that no other family claims the directory;
* the prompt scaffolding, tokenized piece by piece as `ChatState.add_text` does;
* the decoder phase asks for BOTH fusions -- an LM with conv blocks and no conv state decodes wrongly
  after its first step, silently;
* the encoder's NeMo modules against the checkpoint, when it is present.

The numbers -- the encoder rows, every step's logits teacher-forced, the transcript -- are the engine
gate's and the export's docstring's.
"""
import json
from pathlib import Path

import pytest

from loom_exporter.lfm25_audio_export import (
    ASR_SYSTEM_PROMPT, PROMPT_HEAD, PROMPT_TAIL, Lfm25AudioAsrExportConfig, _is_lfm25_audio,
)
from loom_exporter.registry import default_registry

MODEL_DIR = Path("/home/flavio/Dev/models/lfm2.5-audio-1.5b")
needs_checkpoint = pytest.mark.skipif(not MODEL_DIR.is_dir(), reason="no LFM2.5-Audio checkpoint")


def _checkpoint(tmp_path: Path, arch="Lfm2AudioForConditionalGeneration") -> Path:
    d = tmp_path / "lfm"
    d.mkdir(parents=True)
    (d / "config.json").write_text(json.dumps({"architectures": [arch]}))
    (d / "model.safetensors").write_bytes(b"")
    return d


def test_an_lfm2_audio_checkpoint_is_claimed(tmp_path):
    assert _is_lfm25_audio(_checkpoint(tmp_path))
    assert default_registry().detect(_checkpoint(tmp_path / "again")).name == "lfm2.5-audio"


def test_another_architecture_is_not(tmp_path):
    assert not _is_lfm25_audio(_checkpoint(tmp_path, arch="Lfm2ForCausalLM"))


def test_the_prompt_is_the_readmes_asr_turns():
    assert ASR_SYSTEM_PROMPT == "Perform ASR."
    assert "".join(PROMPT_HEAD) == ("<|startoftext|><|im_start|>system\nPerform ASR.<|im_end|>\n"
                                    "<|im_start|>user\n")
    assert "".join(PROMPT_TAIL) == "<|im_end|>\n<|im_start|>assistant\n"


@needs_checkpoint
def test_the_scaffolding_is_tokenized_piece_by_piece():
    """`ChatState.add_text` encodes each piece on its own, so the ids are the concatenation of the
    pieces' ids -- which is not, in general, the ids of the concatenated text."""
    from transformers import AutoTokenizer

    from loom_exporter.lfm25_audio_export import prompt_ids

    tok = AutoTokenizer.from_pretrained(str(MODEL_DIR))
    head, tail, eot = prompt_ids(str(MODEL_DIR))
    assert head == [i for p in PROMPT_HEAD for i in tok.encode(p, add_special_tokens=False)]
    assert tail == [i for p in PROMPT_TAIL for i in tok.encode(p, add_special_tokens=False)]
    assert eot == tok.convert_tokens_to_ids("<|im_end|>")


@needs_checkpoint
def test_the_decoder_fuses_attention_and_convolution(tmp_path):
    phases = {p.name: p for p in Lfm25AudioAsrExportConfig(output_path=str(tmp_path / "x.gguf"),
                                                           model_dir=str(MODEL_DIR)).phases()}
    assert set(phases) == {"encoder", "embed", "decoder", "lm_head"}
    assert phases["decoder"].fuse_attention and phases["decoder"].fuse_conv
    assert not phases["encoder"].fuse_conv, "the conformer's depthwise convs must not acquire state"
