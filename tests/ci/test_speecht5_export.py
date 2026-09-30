"""Family 9b (P5): SpeechT5 -- text in, mel frames out of an autoregressive loop, HiFi-GAN audio.

Everything here runs against a real, randomly initialised `SpeechT5ForTextToSpeech` and
`SpeechT5HifiGan` small enough to trace in a unit test. What is under test:

1. **The wrappers are the reference.** The encoder's relative position bias (Shaw-style, it depends
   on the QUERY), the decoder step with its always-on prenet dropout pinned, the postnet's batch-norm
   affine and the split speaker projection -- each compared with the HF modules at float64, where a
   re-spelling that is not algebraically identical cannot hide in rounding.
2. **The relative index's reshape keeps both of its axes.** `rel_index.shape[0]` is read by the shape
   walk as a batch size (a torch axis 0 that derives to the root axis becomes 1), and the gathered
   `[n, n, head_dim]` rows came out `[1, 1, head_dim]`: an export that wrote, loaded and then failed
   at the first call.
3. **The traced lengths do not reach the graph**, and cross-attention stays unfused.
"""
import io
import json
import zipfile
from pathlib import Path

import numpy as np
import pytest

from loom_exporter.registry import default_registry
from loom_exporter.speecht5_export import DEFAULT_VOICE, SpeechT5ExportConfig, _build_speecht5, _is_speecht5_tts
from loom_exporter.speecht5_voices import SPEAKERS, compat, convert, read_xvector, utterance

torch = pytest.importorskip("torch")

SPEAKER_DIM = 6


def _hf_dir(tmp_path: Path, name: str, config: dict) -> Path:
    d = tmp_path / name
    d.mkdir()
    (d / "config.json").write_text(json.dumps(config))
    return d


# -- detection -------------------------------------------------------------------------------------

def test_a_tts_checkpoint_is_claimed(tmp_path):
    assert _is_speecht5_tts(_hf_dir(tmp_path, "tts", {
        "model_type": "speecht5", "architectures": ["SpeechT5ForTextToSpeech"]}))


@pytest.mark.parametrize("head", ["SpeechT5ForSpeechToText", "SpeechT5ForSpeechToSpeech"])
def test_the_other_speecht5_heads_are_not_claimed(tmp_path, head):
    """ASR and voice conversion share the `model_type` and none of this export's shape."""
    assert not _is_speecht5_tts(_hf_dir(tmp_path, "other", {"model_type": "speecht5", "architectures": [head]}))


def test_the_registry_routes_a_tts_checkpoint_here(tmp_path):
    recognizer = default_registry().detect(_hf_dir(tmp_path, "tts", {
        "model_type": "speecht5", "architectures": ["SpeechT5ForTextToSpeech"]}))
    assert recognizer.name == "speecht5" and recognizer.task == "text-to-speech"


def test_a_missing_vocoder_is_named_not_guessed(tmp_path):
    """The mel spectrogram is only half the model; the export folds the separate HiFi-GAN in."""
    d = _hf_dir(tmp_path, "tts", {"model_type": "speecht5", "architectures": ["SpeechT5ForTextToSpeech"]})
    with pytest.raises(FileNotFoundError, match="speecht5_hifigan"):
        _build_speecht5(d, "/tmp/x.gguf").load_models()


def test_a_missing_voice_set_says_where_to_get_it(tmp_path):
    with pytest.raises(FileNotFoundError, match="cmu-arctic-xvectors"):
        read_xvector(tmp_path, DEFAULT_VOICE)


# -- a tiny checkpoint -----------------------------------------------------------------------------

def _tiny_models():
    from transformers import SpeechT5Config, SpeechT5ForTextToSpeech, SpeechT5HifiGan, SpeechT5HifiGanConfig

    torch.manual_seed(0)
    # `encoder_max_relative_position` 4 against traced and tested lengths of 9-13, so the clipping of
    # `i - j` to [-4, 3] is exercised; 81 ids, SpeechT5's own vocabulary size.
    config = SpeechT5Config(
        vocab_size=81, hidden_size=16, encoder_layers=1, decoder_layers=2, encoder_attention_heads=2,
        decoder_attention_heads=2, encoder_ffn_dim=32, decoder_ffn_dim=32, num_mel_bins=8,
        speech_decoder_prenet_units=8, speech_decoder_postnet_units=8, speech_decoder_postnet_layers=3,
        speaker_embedding_dim=SPEAKER_DIM, max_text_positions=32, max_speech_positions=64,
        encoder_max_relative_position=4, reduction_factor=2)
    model = SpeechT5ForTextToSpeech(config).eval()
    # Non-trivial batch-norm statistics: at init they are 0 and 1, and an affine that dropped either
    # term would still match.
    for layer in model.speech_decoder_postnet.layers:
        layer.batch_norm.running_mean.normal_()
        layer.batch_norm.running_var.uniform_(0.5, 2.0)
    vocoder = SpeechT5HifiGan(SpeechT5HifiGanConfig(
        model_in_dim=8, upsample_initial_channel=8, upsample_rates=[2, 2], upsample_kernel_sizes=[4, 4],
        resblock_kernel_sizes=[3], resblock_dilation_sizes=[[1, 3]])).eval()
    vocoder.mean.normal_()
    vocoder.scale.uniform_(0.5, 2.0)
    return model, vocoder


def _char_proto() -> bytes:
    """A CHAR `ModelProto` in SpeechT5's layout: `<s> <pad> </s> <unk>`, then one piece per character."""
    from sentencepiece import sentencepiece_model_pb2 as spm_pb2

    m = spm_pb2.ModelProto()
    m.trainer_spec.model_type = m.trainer_spec.CHAR
    m.normalizer_spec.add_dummy_prefix = True
    m.normalizer_spec.remove_extra_whitespaces = True
    m.normalizer_spec.precompiled_charsmap = b"\x00\x00\x00\x00charsmap"
    for piece, kind in (("<s>", 3), ("<pad>", 3), ("</s>", 3), ("<unk>", 2)):
        entry = m.pieces.add()
        entry.piece, entry.score, entry.type = piece, 0.0, kind
    for i, ch in enumerate("▁abcdefghijklmnopqrstuvwxyz,.!?'"):
        entry = m.pieces.add()
        entry.piece, entry.score, entry.type = ch, -float(i), 1
    return m.SerializeToString()


@pytest.fixture(scope="module")
def checkpoint(tmp_path_factory):
    pytest.importorskip("sentencepiece")
    out = tmp_path_factory.mktemp("tiny-speecht5")
    model, vocoder = _tiny_models()
    model.save_pretrained(out)
    vocoder.save_pretrained(out / "hifigan")
    (out / "spm_char.model").write_bytes(_char_proto())
    (out / "special_tokens_map.json").write_text(json.dumps({
        "bos_token": "<s>", "eos_token": "</s>", "unk_token": "<unk>", "pad_token": "<pad>"}))
    buf = io.BytesIO()
    np.save(buf, np.linspace(-1, 1, SPEAKER_DIM, dtype=np.float32))
    (out / "xvectors").mkdir()
    with zipfile.ZipFile(out / "xvectors" / "spkrec-xvect.zip", "w") as z:
        for speaker in SPEAKERS:
            z.writestr(f"spkrec-xvect/{utterance(speaker)}.npy", buf.getvalue())
    return out


# -- the wrappers against the reference ------------------------------------------------------------

def _pinned_generation(model, ids, speaker, masks):
    """`generate_speech` with the prenet's dropout masks pinned for the last row, as
    `loom.cpp/scripts/speecht5_reference.py` pins them; returns the per-step `feat_out`/`prob_out`."""
    from transformers.models.speecht5 import modeling_speecht5 as m

    calls = {"n": 0}

    def pinned(self, x, p):
        step, layer = divmod(calls["n"], 2)
        calls["n"] += 1
        mask = torch.zeros_like(x[0])
        mask[-1] = masks[step, layer]
        return torch.where(mask.unsqueeze(0) == 1, x, 0) * 1 / (1 - p)

    spectra, logits = [], []
    postnet = model.speech_decoder_postnet
    hooks = [postnet.feat_out.register_forward_hook(lambda mod, a, out: spectra.append(out)),
             postnet.prob_out.register_forward_hook(lambda mod, a, out: logits.append(out))]
    original = m.SpeechT5SpeechDecoderPrenet._consistent_dropout
    m.SpeechT5SpeechDecoderPrenet._consistent_dropout = pinned
    try:
        with torch.no_grad():
            mel = model.generate_speech(ids, speaker, threshold=2.0, maxlenratio=1.0)  # never stops early
    finally:
        m.SpeechT5SpeechDecoderPrenet._consistent_dropout = original
        for h in hooks:
            h.remove()
    n = len(spectra)
    return mel, torch.cat(spectra).view(n, 2, -1), torch.cat(logits).view(n, 2)


def test_the_wrappers_reproduce_the_reference_at_f64():
    from loom_exporter.speecht5_export import CrossKvPhase, DecoderPhase, EncoderPhase, PostnetPhase, VocoderPhase

    model, vocoder = _tiny_models()
    model, vocoder = model.double(), vocoder.double()
    ids = torch.tensor([[4, 7, 12, 5, 9, 4, 20, 11, 6, 30, 8, 2]])
    n = ids.shape[1]
    speaker = torch.randn(1, SPEAKER_DIM, dtype=torch.float64)
    masks = (torch.rand(n, 2, 8) < 0.5).double()
    mel, spectra, logits = _pinned_generation(model, ids, speaker, masks)
    steps = spectra.shape[0]
    assert steps == n // 2                                    # maxlenratio 1: n / reduction_factor

    with torch.no_grad():
        rel = (torch.arange(n)[:, None] - torch.arange(n)[None]).clamp(-4, 3) + 4
        enc = EncoderPhase(model)(ids, torch.arange(n)[None], rel)
        want_enc = model.speecht5.encoder(input_values=ids, attention_mask=torch.ones_like(ids)).last_hidden_state
        torch.testing.assert_close(enc, want_enc, rtol=0, atol=1e-12)

        cross = CrossKvPhase(model)(enc)
        frames = torch.cat([torch.zeros(1, 8, dtype=torch.float64), spectra[:-1, 1]])[None]
        causal = torch.triu(torch.full((steps, steps), float("-inf"), dtype=torch.float64), 1)[None, None]
        spec, logit = DecoderPhase(model)(frames, torch.arange(steps)[None], masks[None, :steps, 0],
                                          masks[None, :steps, 1], speaker, causal, *cross)
        torch.testing.assert_close(spec[0].view(steps, 2, 8), spectra, rtol=0, atol=1e-12)
        torch.testing.assert_close(logit[0], logits, rtol=0, atol=1e-12)

        torch.testing.assert_close(PostnetPhase(model)(spectra.reshape(1, -1, 8))[0], mel, rtol=0, atol=1e-12)
        torch.testing.assert_close(VocoderPhase(vocoder)(mel[None])[0], vocoder(mel), rtol=0, atol=0)


# -- the export ------------------------------------------------------------------------------------

def _export(checkpoint: Path, out: Path) -> dict:
    from gguf import GGUFReader

    config = SpeechT5ExportConfig(model_dir=str(checkpoint), output_path=str(out))
    config.task = "text-to-speech"
    config.export()
    reader = GGUFReader(str(out))
    topo = {name: json.loads(reader.fields[f"model.graph_topology.{name}"].contents())
            for name in ("encoder", "cross_kv", "decoder", "postnet", "vocoder")}
    return {"topo": topo, "fields": reader.fields, "tensors": {t.name for t in reader.tensors}}


@pytest.fixture(scope="module")
def exported(checkpoint, tmp_path_factory):
    pytest.importorskip("coremltools")
    return _export(checkpoint, tmp_path_factory.mktemp("out") / "tiny.gguf")


def test_the_relative_index_reshape_keeps_both_axes(exported):
    """See the module docstring's point 2: `[1, 1, head_dim]` here is the regression."""
    (reshape,) = [n for n in exported["topo"]["encoder"]["nodes"]
                  if n["op"] == "RESHAPE" and n["outputs"] == ["pos"]]
    assert reshape["attrs"]["shape"] == ["8", "n_tokens", "n_tokens"], reshape["attrs"]["shape"]
    inputs = {i["name"]: i["shape"] for i in exported["topo"]["encoder"]["inputs"]}
    assert inputs["rel_index"] == ["n_tokens", "n_tokens"]


def test_only_the_self_attention_is_fused_and_cached(exported):
    attention = [n for n in exported["topo"]["decoder"]["nodes"] if n["op"] == "ATTENTION"]
    assert len(attention) == 2 and all(n.get("attrs", {}).get("kv_cache", True) for n in attention)
    assert not any(n["op"] == "ATTENTION" for n in exported["topo"]["encoder"]["nodes"])


def test_the_cross_kv_inputs_ride_the_source_axis(exported):
    shapes = {i["name"]: i["shape"] for i in exported["topo"]["decoder"]["inputs"]}
    for name in ("xk_0", "xv_0", "xk_1", "xv_1"):
        assert shapes[name] == ["16", "n_enc_frames", "1"], (name, shapes[name])
    assert shapes["attention_mask"] == ["n_kv", "n_tokens"]
    assert shapes["speaker"] == [str(SPEAKER_DIM), "1"]


def test_the_vocoder_upsamples_by_its_hop(exported):
    """The tiny vocoder's two x2 upsamplers: the waveform is `4 * n_enc_frames`, and every
    intermediate VIEW carries the running product rather than a fresh symbol."""
    views = [n["attrs"]["shape"][0] for n in exported["topo"]["vocoder"]["nodes"] if n["op"] == "VIEW"]
    assert views == ["2*n_enc_frames", "4*n_enc_frames"], views


def test_the_tokenizer_and_the_voice_travel_with_the_model(exported):
    fields = exported["fields"]
    assert fields["tokenizer.ggml.model"].contents() == "t5"      # the CHAR model, written as Unigram
    assert fields["loom.output.kind"].contents() == "audio"
    assert fields["loom.sample_rate"].contents() == 16000
    assert fields["loom.tts.voices"].contents() == [DEFAULT_VOICE] == ["slt"]
    assert fields["loom.voice.compat"].contents() == compat(SPEAKER_DIM)
    # The reference's number speller rides with the vocabulary, because the vocabulary has no digits.
    assert fields["tokenizer.ggml.numbers.scheme"].contents() == "english_number_normalizer"
    assert fields["tokenizer.ggml.numbers.symbol_chain"].contents()[:2] == ["-", "$"]


def test_the_traced_lengths_do_not_reach_the_graph(checkpoint, tmp_path, monkeypatch):
    pytest.importorskip("coremltools")
    import loom_exporter.speecht5_export as mod

    first = _export(checkpoint, tmp_path / "a.gguf")["topo"]
    for name, value in (("TRACE_TOKENS", 9), ("TRACE_STEPS", 5), ("TRACE_SRC", 7), ("TRACE_FRAMES", 11)):
        monkeypatch.setattr(mod, name, value)
    second = _export(checkpoint, tmp_path / "b.gguf")["topo"]
    for name in first:
        assert json.dumps(first[name], sort_keys=True) == json.dumps(second[name], sort_keys=True), name


# -- voice files (loom.cpp ADR-045, ADR-058) --------------------------------------------------------

def test_the_seven_speakers_become_voice_files_the_model_accepts(checkpoint, tmp_path):
    """Each file's one tensor is the driver input it becomes, and its stamp is the one the model
    declares -- `loom::load_voice` compares the two strings and nothing else."""
    from gguf import GGUFReader

    written = convert(checkpoint, tmp_path, dim=SPEAKER_DIM)
    assert sorted(written) == sorted(SPEAKERS)
    for speaker in SPEAKERS:
        reader = GGUFReader(str(tmp_path / f"{speaker}.gguf"))
        assert reader.fields["loom.voice.architecture"].contents() == "speecht5"
        assert reader.fields["loom.voice.compat"].contents() == compat(SPEAKER_DIM)
        assert reader.fields["loom.voice.name"].contents() == speaker
        assert "commercial or otherwise" in reader.fields["loom.voice.license"].contents()
        (tensor,) = reader.tensors
        assert tensor.name == "speaker" and tensor.n_elements == SPEAKER_DIM


def test_the_stamp_names_the_embedding_space_not_the_weights():
    """An x-vector is the extractor's output, so it fits every SpeechT5 trained on that extractor --
    and a different width is a different space."""
    assert compat(512) == "xvector:speechbrain/spkrec-xvect-voxceleb:512"
    assert compat(512) != compat(192)


def test_your_own_xvector_needs_a_name_and_the_recordings_licence(checkpoint, tmp_path):
    own = tmp_path / "me.npy"
    np.save(own, np.ones((1, SPEAKER_DIM), dtype=np.float32))   # a batched extractor's [1, dim]
    with pytest.raises(ValueError, match="--license"):
        convert(checkpoint, tmp_path, source=own, name="me", dim=SPEAKER_DIM)
    assert list(convert(checkpoint, tmp_path, source=own, name="me", license="CC0-1.0",
                        dim=SPEAKER_DIM)) == ["me"]


def test_an_unknown_speaker_or_a_wrong_width_is_refused(checkpoint, tmp_path):
    with pytest.raises(KeyError, match="bdl"):
        convert(checkpoint, tmp_path, only=["xyz"], dim=SPEAKER_DIM)
    with pytest.raises(ValueError, match="speaker_embedding_dim"):
        convert(checkpoint, tmp_path, only=["slt"], dim=SPEAKER_DIM + 1)
