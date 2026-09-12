"""Family 11's third leaf: EnCodec, the codec that is not one graph.

**What this file asserts is that the lengths survive**, and that is the same assertion
`test_audio_codec_export.py` exists for, reached by a different road. DAC's first export returned one
frame's audio forever because the shape walk gave up on a rank-reducing slice; EnCodec's first export
returned a two-hundredth of its audio because MIL retires the symbolic algebra through a shape-derived
slice and the walk then substitutes the root axis for the fresh opaque dim. Neither raised anything.

So the checks below are on the crop expressions -- which must carry the CUMULATIVE upsampling, not the
root axis alone -- and on the two facts the tracing patches rest on, re-derived from the real modules
rather than trusted from a comment.
"""
import json
from pathlib import Path

import pytest

from loom_exporter.registry import default_registry
from loom_exporter.tasks import task_spec


def _hf_dir(tmp_path: Path, name: str, config: dict) -> Path:
    d = tmp_path / name
    d.mkdir()
    (d / "config.json").write_text(json.dumps(config))
    return d


def test_the_registry_routes_encodec_to_its_own_export_shape(tmp_path):
    """Same task, same contract, a different config class -- which is what `tasks.py`'s own
    `audio-codec` entry argues for declaring the root base class."""
    from loom_exporter.encodec_export import EnCodecExportConfig

    recognizer = default_registry().detect(_hf_dir(tmp_path, "enc", {"model_type": "encodec"}))
    assert recognizer.name == "encodec" and recognizer.task == "audio-codec"
    config = recognizer.build_config(tmp_path / "enc", "/tmp/x.gguf")
    assert isinstance(config, EnCodecExportConfig)
    assert task_spec("audio-codec").base_config_class().__name__ == "LoomExportConfig"


def test_hparams_are_empty_without_a_checkpoint(tmp_path):
    from loom_exporter.encodec_export import EnCodecExportConfig

    assert EnCodecExportConfig(output_path="/tmp/x.gguf", model_dir=str(tmp_path)).hparams() == {}


# -- the two facts the tracing patches rest on -----------------------------------------------------

def _real_model():
    pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    checkpoint = Path("/home/flavio/Dev/models/encodec-32khz")
    if not checkpoint.is_dir():
        pytest.skip(f"{checkpoint} is not present")
    import torch

    return transformers.EncodecModel.from_pretrained(str(checkpoint), dtype=torch.float32).eval()


@pytest.mark.gate
def test_every_decode_path_convolution_is_stride_one_and_pads_by_zero():
    """Blocker 1, PROVED rather than argued.

    The family's note said EnCodec's length-derived padding "works out to exactly 0 for a stride-1
    convolution" and had to be proved per stage before it could be patched to a constant. This is that
    proof, re-derived from the real modules every run: every `EncodecConv1d` on the decode path has
    stride 1, and its own `_get_extra_padding_for_conv1d` returns 0 at every length -- including the
    short ones, where the reflect-pad branch this export does not take would otherwise matter.
    """
    import torch
    from transformers.models.encodec.modeling_encodec import EncodecConv1d

    model = _real_model()
    convs = [m for m in model.decoder.modules() if isinstance(m, EncodecConv1d)]
    assert len(convs) == 10, f"the decode path has {len(convs)} EncodecConv1d, not 10"
    assert {int(c.stride) for c in convs} == {1}
    for length in (1, 2, 3, 4, 5, 7, 8, 13, 16, 37, 64, 100, 301, 1024, 4096):
        for conv in convs:
            extra = conv._get_extra_padding_for_conv1d(torch.zeros(1, conv.conv.in_channels, length))
            assert int(extra) == 0, f"extra padding {int(extra)} at length {length}"


@pytest.mark.gate
def test_the_static_unpad_keeps_the_same_elements():
    """Patch 3 changes `x[..., left : len - right]` into `x[..., left : -right]`. Same elements, and
    this is what says so -- on the real modules, at several lengths, rather than by reading the two
    expressions and agreeing they look equal."""
    import torch
    from transformers.models.encodec.modeling_encodec import EncodecConvTranspose1d

    from loom_exporter.encodec_export import patch_encodec_padding

    model = _real_model()
    transposes = [m for m in model.decoder.modules() if isinstance(m, EncodecConvTranspose1d)]
    assert transposes, "no transposed convolutions on the decode path"
    before = []
    for module in transposes:
        for length in (4, 16, 37):
            x = torch.randn(1, module.conv.in_channels, length)
            with torch.no_grad():
                before.append(module(x).clone())
    patch_encodec_padding()
    i = 0
    for module in transposes:
        for length in (4, 16, 37):
            torch.manual_seed(length)
            x = torch.randn(1, module.conv.in_channels, length)
            with torch.no_grad():
                after = module(x)
            assert after.shape == before[i].shape, f"unpad changed the shape at length {length}"
            i += 1


# -- the real export -------------------------------------------------------------------------------

@pytest.mark.gate
def test_every_crop_carries_the_cumulative_upsampling(tmp_path):
    """THE CHECK THIS FILE EXISTS FOR.

    Each upsampling stage multiplies the length by its own stride, so the crops must read
    `8*n_codes`, `40*n_codes`, `160*n_codes`, `640*n_codes` -- the running product. A crop that reads
    `5*n_codes` where it should read `40*n_codes` is what a walk that lost the symbol produces: the
    export succeeds, the GGUF loads, the driver returns floats, and the waveform is a fraction of its
    length. `640` is the model's own `hop_length`, which is the arithmetic stated from the other end.
    """
    pytest.importorskip("coremltools")
    from gguf import GGUFReader

    from loom_exporter.encodec_export import EnCodecExportConfig

    model = _real_model()
    out = tmp_path / "encodec.gguf"
    config = EnCodecExportConfig(output_path=str(out),
                                 model_dir="/home/flavio/Dev/models/encodec-32khz", n_frames=8)
    config.task = "audio-codec"
    config.export()
    reader = GGUFReader(str(out))
    post = json.loads(reader.fields["model.graph_topology.post"].contents())
    crops = [n["attrs"]["shape"] for n in post["nodes"] if n["op"] == "VIEW"]
    assert [c[0] for c in crops] == ["8*n_codes", "40*n_codes", "160*n_codes", "640*n_codes"], crops
    assert int(model.config.hop_length) == 640

    driver = reader.fields["model.driver_script"].contents()
    # One call per layer, chained, and the residual is `pre`'s own output rather than a second copy.
    assert "loom.run_recurrent('lstm_l0_fwd', seq," in driver
    assert "loom.run_recurrent('lstm_l1_fwd', lstm_0," in driver
    assert "{lstm_out = lstm_1, residual = seq}" in driver
