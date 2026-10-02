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


def test_an_lfm2_audio_checkpoint_is_claimed_once_per_door(tmp_path):
    """Two doors over one directory: each task names its own, and an untasked export is ambiguous
    by design -- which door a caller wants is not in the checkpoint."""
    d = _checkpoint(tmp_path)
    assert _is_lfm25_audio(d)
    registry = default_registry()
    assert registry.detect(d, task="automatic-speech-recognition").name == "lfm2.5-audio"
    assert registry.detect(d, task="text-to-speech").name == "lfm2.5-audio-tts"
    with pytest.raises(ValueError, match="more than one recognizer"):
        registry.detect(d)


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


# -- the speaking door ------------------------------------------------------------------------------

def test_the_detokenizer_window_is_its_own_mask():
    """liquid-audio: `d_idx = idx - idx[:, None]`, allowed iff `d_idx <= 0 and d_idx > -window`."""
    import torch

    from loom_exporter.lfm25_audio_export import detokenizer_window_mask

    n, window = 40, 30
    mask = detokenizer_window_mask(torch.arange(n, dtype=torch.int32).view(1, -1), window)[0, 0]
    idx = torch.arange(n)
    d = idx - idx[:, None]
    assert torch.equal(mask == 0, (d <= 0) & (d > -window))


def test_each_depth_row_embeds_the_previous_code_in_the_previous_table():
    """Row j of the depthformer adds `depth_embeddings[j - 1](code_{j-1})`; row 0 adds nothing."""
    import torch

    from loom_exporter.lfm25_audio_export import AUDIO_VOCAB, N_CODEBOOKS, DepthPhase

    dim = 64
    w = {"depth_linear.weight": torch.randn(dim * N_CODEBOOKS, 32), "depth_linear.bias": torch.zeros(dim * N_CODEBOOKS)}
    for i in range(N_CODEBOOKS):
        w[f"depth_embeddings.{i}.embedding.weight"] = torch.randn(AUDIO_VOCAB, dim)
        w[f"depth_embeddings.{i}.embedding_norm.weight"] = torch.ones(dim)
    depth = DepthPhase(w, layers=0, dim=dim, heads=4, kv_heads=2)
    assert depth.offsets.tolist() == [0] + [j * AUDIO_VOCAB for j in range(N_CODEBOOKS - 1)]
    assert depth.row_mask.view(-1).tolist() == [0.0] + [1.0] * (N_CODEBOOKS - 1)


def test_a_voice_file_is_its_prompt_under_the_driver_input_name(tmp_path):
    pytest.importorskip("gguf")
    from gguf import GGUFReader

    from loom_exporter.lfm25_audio_voices import write_voice

    write_voice([1, 2, 3], tmp_path / "v.gguf", name="v", compat="abc", origin="test")
    r = GGUFReader(str(tmp_path / "v.gguf"))
    assert [t.name for t in r.tensors] == ["voice_prompt"]
    assert r.tensors[0].data.tolist() == [1.0, 2.0, 3.0]
    field = r.fields["loom.voice.architecture"]
    assert bytes(field.parts[field.data[0]]).decode() == "lfm2.5-audio-tts"


@needs_checkpoint
def test_the_voices_are_the_readmes_four_and_the_default_is_built_in():
    from loom_exporter.lfm25_audio_export import DEFAULT_VOICE, TTS_VOICES, tts_prompt_ids

    ids = tts_prompt_ids(str(MODEL_DIR))
    assert set(ids["voices"]) == set(TTS_VOICES) and DEFAULT_VOICE in TTS_VOICES
    # `<|startoftext|>` opens the prompt; it is also the id the driver drops from the head of the text.
    assert ids["pre"][0] == 1
