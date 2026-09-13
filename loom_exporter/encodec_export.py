"""EnCodec 32 kHz -- family 11's second published leaf, and the first codec that is not one graph.

DAC and SNAC are `Flattened` exports: codes in, waveform out, one traced topology
(`audio_codec_export.py`). EnCodec decodes through a **2-layer LSTM over the time axis**, and ggml has
no LSTM op -- a topology is a pure dataflow graph, so a recurrence is not expressible as one at all.
That makes this a three-phase export with the recurrence between the graphs rather than inside one,
and it is the only thing about this model that is genuinely different. **The timestep loop is in C++,
not in the driver**: `loom.run_recurrent` takes a whole sequence and walks it, carrying `h`/`c` in
`std::vector<float>` and reusing the cell's built graph across steps, so the Lua side makes ONE call
per layer. (Parakeet's transducer is the other shape and loops in Lua by necessity -- how many symbols
it emits per frame depends on what it just emitted, so there is no fixed-length sweep to hand over.)

    codes -> [pre] -> sequence -> [lstm_l0] -> [lstm_l1] -> [post] -> waveform
                          \\------------ residual ------------/

Both blockers this family recorded against EnCodec are now closed, and neither was where it looked:

* **The dynamic-padding wall was one line, and it is now PROVED rather than argued.** `EncodecConv1d`
  pads by a length-derived amount (`_get_extra_padding_for_conv1d`), which coremltools refuses once
  the length is symbolic -- the same wall Supertonic hit. The family's note said the extra padding
  "works out to exactly 0 for a stride-1 convolution" and had to be proved per stage. It is: every one
  of the **10** `EncodecConv1d` on the decode path has stride 1, and the expression evaluates to 0 at
  every length from 1 to 4096 (`tests/ci/test_encodec_export.py` re-derives it rather than trusting
  this comment). So the patch is a constant 0, and the padding becomes static.
* **The LSTM needed no new machinery at all.** `RecurrentPhase` has traced a stacked `nn.LSTM` into
  per-timestep cell topologies since Parakeet's prediction network, and `loom.run_recurrent` runs one
  in C++. What was missing was the driver-side call -- `RecurrentCall`, added with this model, which
  is one Lua call per layer rather than `run_bi_lstm`'s per-timestep Lua loop.

What WAS missing, and is the one engine change: **ELU**. EnCodec's SEANet decoder activates with it
where every vocoder in families 7-9 uses LeakyReLU, and `ggml_elu` already existed unexposed. One
registration, one topology rule (with an `alpha != 1` guard, because ggml's is fixed at 1).

**The contract is unchanged**: `audio_codes -> audio`, codes frame-major, `codec.n_codebooks` wide,
and a driver that takes one array. A caller cannot tell from the outside that this one is three
topologies and a loop, which is the point.

**Time-major across the phase boundary.** `loom.run_recurrent` indexes its sequence as
`seq[t * input_dim + k]`, so `pre` emits `[1, n_codes, 1024]` -- the transpose of the model's own
convolutional layout -- and `post` takes it back the same way. Doing it in the wrappers keeps the
driver free of layout arithmetic, and the residual add lives in `post` because both of its operands
are already there.
"""
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
import coremltools as ct
from torch import nn

from .decomposition import Decomposition, MultiPhase
from .export_config import LoomExportConfig
from .multi_phase_export import BaseMultiPhaseModelExportConfig, ExportPhase, RecurrentPhase
from .spec_protocol import Unchecked


def patch_encodec_padding() -> None:
    """Make every length computation on EnCodec's decode path STATIC. Three rewrites, one cause.

    All three are the same shape of problem -- an index derived from `hidden_states.shape[-1]` -- and
    it is worth naming what each one costs, because two of them are silent:

    **1. `_get_extra_padding_for_conv1d` -> 0.** The loud one, and the blocker this family recorded:
    coremltools' torch frontend refuses a length-derived pad once the length is symbolic (`Dynamic
    padding for n-dimensional tensors is not supported`). The real expression is
    `ceil((L - k + (k - stride)) / stride) * stride + k - (k - stride) - L`, which for `stride == 1`
    is identically 0 for every L -- and every convolution on the DECODE path is stride 1
    (`tests/ci/test_encodec_export.py` re-derives both facts from the real modules).

    **2. `_pad1d`'s trailing re-slice.** With (1) at zero, `padded[..., :padded.shape[-1] - 0]` is the
    whole tensor -- but it still traces as a `slice_by_index` whose end is read off the shape.

    **3. `EncodecConvTranspose1d`'s unpad, rewritten to NEGATIVE static indices.**
    `hidden_states[..., padding_left : hidden_states.shape[-1] - padding_right]` becomes
    `[..., padding_left : -padding_right]`: the same elements, a compile-time-constant slice.

    **Why (2) and (3) matter is the interesting half.** They do not fail -- they lose the SYMBOL. MIL
    cannot express "this length minus that constant" through a shape-derived slice, so it retires the
    algebra and invents a fresh opaque dim: `8*is0 + 8` goes in and `is118` comes out. Every later
    length is then unrelated to the root axis, and the exporter's rule for an unknown symbol is to
    substitute the root -- so a crop that should read `8*n_codes + 2` reads `n_codes + 2`, the graph
    builds, the decode runs, and the waveform comes back a fraction of its proper length. That is
    family 11's signature failure (the first DAC export returned one frame's audio forever) reached by
    a different road, which is why `tests/ci/test_encodec_export.py` asserts on the emitted crop
    expressions rather than on the call.

    Class-level, like every other tracing patch here, and applied at `load_model` rather than at
    import: `transformers` is imported lazily by the families that need it.
    """
    from transformers.models.encodec.modeling_encodec import EncodecConv1d, EncodecConvTranspose1d

    EncodecConv1d._get_extra_padding_for_conv1d = lambda self, hidden_states: 0

    def _pad1d(hidden_states, paddings, mode="zero", value=0.0):
        # The module's own body minus the `length <= max_pad` branch and the trailing re-slice that
        # branch exists to undo. Both are reachable only when the input is SHORTER than the pad, which
        # on the decode path means fewer than `min_frames` frames.
        return nn.functional.pad(hidden_states, paddings, mode, value)

    EncodecConv1d._pad1d = staticmethod(_pad1d)

    def _conv_transpose_forward(self, hidden_states):
        kernel_size, stride = self.conv.kernel_size[0], self.conv.stride[0]
        padding_total = kernel_size - stride
        hidden_states = self.conv(hidden_states)
        if self.norm_type == "time_group_norm":
            hidden_states = self.norm(hidden_states)
        if self.causal:
            raise NotImplementedError(
                "EncodecConvTranspose1d with use_causal_conv=True trims by `ceil(padding_total * "
                "trim_right_ratio)`, which this patch does not reproduce. No checkpoint this family "
                "targets sets it."
            )
        padding_right = padding_total // 2
        padding_left = padding_total - padding_right
        if padding_right:
            return hidden_states[..., padding_left:-padding_right]
        return hidden_states[..., padding_left:]

    EncodecConvTranspose1d.forward = _conv_transpose_forward


class _PreWrapper(nn.Module):
    """`codes [1, n_codes, n_q]` -> the LSTM's input sequence, flat and TIME-MAJOR.

    Two layout changes and one model call. The transpose to `[n_q, batch, frames]` is what
    `EncodecResidualVectorQuantizer.decode` wants (`_decode_frame` does the same one); the transpose
    on the way out is what `loom.run_recurrent` indexes.
    """

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, codes):
        embeddings = self.model.quantizer.decode(codes.transpose(1, 2).transpose(0, 1))
        hidden = self.model.decoder.layers[0](embeddings)        # [1, hidden, n_codes]
        # `[1, n_codes, hidden]` and NOT flattened: the engine reads this as one ROW per timestep out of
        # a retained `[hidden, n_codes]` tensor, so it has to keep its second axis. Flattening it made
        # the driver marshal the whole sequence to hand it back unchanged.
        return hidden.transpose(1, 2)


class _PostWrapper(nn.Module):
    """`(lstm_out, residual)` -> the waveform.

    **The residual add is here rather than in the driver**, because both operands are already inputs
    to this phase and adding two `n_codes * 1024` arrays in Lua would marshal both of them across the
    boundary to do arithmetic the graph does for free. It is `EncodecLSTM.forward`'s own
    `self.lstm(x)[0] + x`, with the `permute`s absorbed into the layout this phase already takes.
    """

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, lstm_out, residual):
        hidden = (lstm_out + residual).transpose(1, 2)           # [1, hidden, n_codes]
        for layer in self.model.decoder.layers[2:]:
            hidden = layer(hidden)
        return hidden.reshape(1, -1)


@dataclass(kw_only=True)
class EnCodecExportConfig(BaseMultiPhaseModelExportConfig):
    """EnCodec's decode half -> one Loom GGUF carrying three topologies and a driver."""

    architecture: Optional[str] = "encodec"
    model_dir: str
    decomposition: Decomposition = field(default_factory=MultiPhase)
    driver_script_path: Path = Path(__file__).resolve().parent / "encodec_driver"
    # Frames the trace runs at, and the RangeDim ceiling. 4096 frames is 82 s at this checkpoint's
    # 50 Hz; the LSTM loop is per-frame host-side work, so a longer clip is a slower call rather than
    # a different graph.
    n_frames: int = 16
    max_frames: int = 4096
    # Read off the checkpoint by `load_model`, never declared.
    _n_codebooks: Optional[int] = None
    _codebook_size: Optional[int] = None
    _sample_rate: Optional[int] = None
    _hop_length: Optional[int] = None
    _hidden: Optional[int] = None
    _n_lstm_layers: Optional[int] = None
    _model: Optional[object] = None

    __unchecked__ = {
        "architecture": Unchecked("the checkpoint's own `model_type`, and the name the engine reads "
                                  "back. Stated rather than read only because this config is built "
                                  "by a recognizer that already matched on that exact string"),
        "model_dir": Unchecked("path to the HF directory. The recognizer's detect() already read its "
                               "config.json, and the loader raises on anything it cannot load"),
        "n_frames": Unchecked("the concrete length torch.jit.trace runs at; the dynamic range is "
                              "declared separately, so this constrains nothing"),
        "max_frames": Unchecked("the ct.RangeDim upper bound -- a declaration about the export. A "
                                "convolutional decoder is length-agnostic and the LSTM is a loop, so "
                                "there is no ceiling in the checkpoint to check it against"),
        "_n_codebooks": Unchecked("READ off the checkpoint's own config during load_model -- the "
                                  "width of the matrix a caller passes"),
        "_codebook_size": Unchecked("same: the checkpoint's own config"),
        "_sample_rate": Unchecked("same"),
        "_hop_length": Unchecked("same. The frame rate the contract declares is their quotient"),
        "_hidden": Unchecked("the LSTM's width, read off the real `nn.LSTM.input_size` rather than "
                             "recomputed from `num_filters * 2 ** len(upsampling_ratios)` -- the "
                             "module is the authority and the formula is a restatement of it"),
        "_n_lstm_layers": Unchecked("read off the real module's `num_layers`, for the same reason: "
                                    "the driver calls one `RecurrentCall` per layer and the module "
                                    "states how many there are"),
        "_model": Unchecked("the loaded model, cached so phases() can build wrappers around it. A "
                            "field only because this is a dataclass"),
    }

    def load_model(self):
        from transformers import EncodecModel

        print(f"Loading encodec codec from {self.model_dir}...")
        patch_encodec_padding()
        model = EncodecModel.from_pretrained(self.model_dir, dtype=torch.float32).eval()
        config = model.config
        if config.normalize:
            raise NotImplementedError(
                "this checkpoint declares normalize=True, so its decode needs the per-clip scale its "
                "encoder produced. That is a second input and a different contract; this family "
                "exports the un-normalised path only."
            )
        if config.chunk_length_s is not None:
            raise NotImplementedError(
                f"this checkpoint declares chunk_length_s={config.chunk_length_s}, so its decode is "
                f"chunked with overlap-add across frames. The 32 kHz checkpoint this targets declares "
                f"none, and a chunked one is a different driver, not a longer call."
            )
        lstm = model.decoder.layers[1].lstm
        self._n_codebooks = int(config.num_quantizers)
        self._codebook_size = int(config.codebook_size)
        self._sample_rate = int(config.sampling_rate)
        self._hop_length = int(config.hop_length)
        self._hidden = int(lstm.input_size)
        self._n_lstm_layers = int(lstm.num_layers)
        self._model = model
        print(f"  {self._n_codebooks} codebooks at {self._sample_rate / self._hop_length:g} Hz, "
              f"{self._n_lstm_layers}-layer LSTM of width {self._hidden}")
        return model

    def export_architecture(self) -> str:
        return self.architecture

    def phases(self) -> List[ExportPhase]:
        model = self._model if self._model is not None else self.load_model()
        frames = ct.RangeDim(1, self.max_frames)
        codes = torch.zeros((1, self.n_frames, self._n_codebooks), dtype=torch.long)
        sequence = torch.zeros((1, self.n_frames, self._hidden), dtype=torch.float32)
        # ONE RangeDim instance across `post`'s two inputs, deliberately: they are the same sequence
        # and its residual and can never differ in length, which is exactly the case coremltools'
        # shared-symbol behaviour is right for (`_validate_input_axes` names the alternative).
        post_frames = ct.RangeDim(1, self.max_frames)
        return [
            ExportPhase(
                name="pre", wrapper=_PreWrapper(model), dummy_inputs=(codes,), root_axis="n_codes",
                mil_inputs=[ct.TensorType(name="codes",
                                          shape=(1, frames, self._n_codebooks), dtype=np.int32)],
            ),
            RecurrentPhase(name="lstm", module=model.decoder.layers[1].lstm, number_layers=True),
            ExportPhase(
                name="post", wrapper=_PostWrapper(model), dummy_inputs=(sequence, sequence),
                root_axis="n_codes",
                mil_inputs=[
                    ct.TensorType(name="lstm_out", shape=(1, post_frames, self._hidden),
                                  dtype=np.float32),
                    ct.TensorType(name="residual", shape=(1, post_frames, self._hidden),
                                  dtype=np.float32),
                ],
            ),
        ]

    def driver_components(self) -> List:
        """Pre, one `run_recurrent` per LSTM layer, post. No hand-written Lua at all.

        The chain is the whole driver: each layer's sequence is the previous one's output, and `pre`'s
        own output is still live at the end as the residual -- which is why it is bound to a local
        rather than retained. `n_codes` is `#codes / n_codebooks`, the same divisor the flattened
        leaves' synthesized driver computes.
        """
        from .driver_components import (
            CALLER, DriverInputs, DriverReturn, RecurrentCall, SubgraphCallComponent,
        )
        from .driver_ir import BinOp, Len, Lit, OutputRef, Var

        # `or` defaults, for the ONE caller that has no checkpoint: `component_registry.usage()`
        # builds every registered config without one to attribute components to models, and this
        # method is what it reads. The catalogue's question is WHICH components this family uses, so
        # a shape-only build naming each of them once answers it; a real export has both numbers.
        n_codebooks = self._n_codebooks or 1
        n_lstm_layers = self._n_lstm_layers or 1
        n_codes = BinOp("floordiv", Len(Var("codes")), Lit(n_codebooks))
        components = [
            # The one binding, for the `inputs.tokens` alias every other family's caller may use --
            # and so the three call sites below read one local rather than re-deriving `#inputs.codes`
            # each time.
            DriverInputs(bindings=(("codes", CALLER),), n_tokens=n_codes),
            SubgraphCallComponent(
                topology="pre", outputs=(), retain=True, length=n_codes,
                inputs={"codes": Var("codes")},
                note=("The RVQ sum and the first convolution, emitted time-major for the loop below. "
                      "RETAINED: its sequence is read by the first LSTM layer and again at the end as "
                      "the residual, and neither reader is the host -- so it never becomes a Lua "
                      "table. `output_store.h`'s rule: marshal only what is genuinely host-side."),
            ),
        ]
        previous = "pre"
        for layer in range(n_lstm_layers):
            topology = f"lstm_l{layer}_fwd"
            components.append(RecurrentCall(
                topology=topology, out_var=f"lstm_{layer}_gen", sequence=OutputRef(previous),
                seq_len=n_codes, input_dim=self._hidden or 1, hidden_dim=self._hidden or 1,
                retain=True,
                note=("The recurrence: ONE call per layer, with the timestep loop, the h/c carry and "
                      "the sequence itself all on the C++ side. It is outside the GRAPH because a "
                      "topology is pure -- no node in one can carry state across timesteps -- but "
                      "outside the graph is not the same as across the boundary, and nothing here "
                      "crosses: each layer reads the previous one's retained rows."
                      if layer == 0 else None),
            ))
            previous = topology
        components.append(SubgraphCallComponent(
            topology="post", outputs=("wav",), length=n_codes,
            inputs={"lstm_out": OutputRef(previous), "residual": OutputRef("pre")},
            note=("The residual add EncodecLSTM.forward performs, then the four upsampling stages. "
                  "Its two inputs are the only two retained tensors still live, and the waveform it "
                  "returns is the one value this driver marshals."),
        ))
        components.append(DriverReturn(values=("wav",)))
        return components

    def hparams(self) -> dict:
        """The same four keys DAC and SNAC declare, computed the same way -- see
        `audio_codec_export.AudioCodecExportConfig.hparams`.

        EnCodec's codebooks all run at one rate, so `codes_per_frame` is `n_codebooks` and
        `frame_rate` is `sample_rate / hop_length` with no coarse-stride divisor. Stated here rather
        than inherited because this config is a different export SHAPE, not a different codec: the
        contract it writes is deliberately identical.
        """
        if self._n_codebooks is None:
            return {}
        return {
            "codec.n_codebooks": self._n_codebooks,
            "codec.codebook_size": self._codebook_size,
            "codec.frame_rate": float(self._sample_rate) / float(self._hop_length),
            "sample_rate": self._sample_rate,
        }

    def backend_kwargs(self) -> dict:
        return dict(hparams=self.hparams())


def _build_encodec(path: Path, output_path: str) -> LoomExportConfig:
    return EnCodecExportConfig(output_path=output_path, model_dir=str(path))
