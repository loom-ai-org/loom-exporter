"""NVIDIA Nemotron-Labs-Unified-ASR (`EncDecRNNTBPEModelWithPrompt`): a unified offline/streaming
FastConformer RNN-T with language-ID prompt conditioning, restored from its `.nemo` -- the fourth
transducer leaf, Parakeet's loader with Nemotron 3.5's prompt.

**Needs NeMo 3.0** (`~/.venvs/nemo3`): the prompt model class and the unified encoder's options
(`att_chunk_context_size`, `chunked_limited_with_rc` attention, `dcc` convolutions) are not in the 2.6
the `piper` venv pins for every other `.nemo` export. On Python 3.11 that venv must be >= 3.11.4, whose
`tarfile.extract` takes the `filter=` NeMo 3.0 passes it.

**`infer` is the OFFLINE mode**, which is what loom's one-pass `speech2text` door is: the encoder at its
restored `att_context_size = [-1, -1, -1]` (full-context attention, the non-causal convolutions it was
trained with for offline decoding), then Parakeet's greedy transducer loop.

**The live door is the streaming mode** (loom.cpp ADR-110): NeMo's BUFFERED streaming, which reruns the
whole model on up to `left + chunk + right` seconds per step (5.6 + 0.56 + 0.56 by default) under the
chunked attention mask of those sizes. `encoder_step` is the encoder phase with that mask, its three
sizes inputs; `nemotron_asr_driver/02_unified_stream.lua` is the script's buffer loop. Its ids are
NeMo's streaming ids, which are not the one pass's.

**Everything after the encoder is the transducer template unchanged**: NeMo's `decoder.prediction`
(embedding + 2-layer LSTM) and `joint`, read by `ASRParakeetExportConfig.transducer_parts`.

**The encoder phase is the leaf's own, because of the prompt.** NeMo's `forward` builds a `[B, T, 128]`
one-hot of the prompt id with `scatter_`, concatenates it to the encoder output and applies
`prompt_kernel` (Linear(1152 -> 2048), ReLU, Linear(2048 -> 1024)). The first linear over `[h; onehot(p)]`
is `W_h h + W_p[:, p] + b`, so the scatter and the concat become an embedding lookup of `W_p`'s column --
the same fold as `nemotron_asr_export`, and the same `[1]` int input the driver fills from
`inputs.language` (the contract's language table maps names to prompt ids), else `auto`.

**Its reference is `forward(..., prompt_indices=...)` plus the model's own greedy decode, not
`transcribe()`.** NeMo 3.0's `transcribe(target_lang=...)` drops the language on the way to its
dataset, and that dataset's default prompt mode is the TRAINING one (`unified`: the auto prompt or the
real language, chosen at random per clip). See `fixture_gen/reference_forward_nemotron_unified.py`.
"""
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn

from .multi_phase_export import ExportPhase
from .parakeet_export import ASRParakeetExportConfig
from .spec_protocol import Unchecked

# The restore class this leaf is for, as the archive's `model_config.yaml` names it.
TARGET = "EncDecRNNTBPEModelWithPrompt"

# A stream's default left context, chunk and right context, in seconds: the model card's (1.12 s of
# latency; 5.6 s of left context is what it was trained with).
STREAM_SECONDS = (5.6, 0.56, 0.56)


class _UnifiedEncoderWrapper(nn.Module):
    """`(waveform [1, n], length [1], prompt [1]) -> [1, T, d_model]`: NeMo's preprocessor and encoder,
    then the language prompt folded into an embedding lookup (see the module docstring)."""

    def __init__(self, model):
        super().__init__()
        self.preprocessor = model.preprocessor
        self.encoder = model.encoder
        kernel = model.prompt_kernel
        if not (isinstance(kernel, nn.Sequential) and len(kernel) == 3 and isinstance(kernel[0], nn.Linear)
                and isinstance(kernel[2], nn.Linear)):
            raise ValueError(f"prompt_kernel is expected to be Linear, activation, Linear; it is {kernel}")
        hidden = int(model.cfg.encoder.d_model)
        n_prompts = int(model.num_prompts)
        weight = kernel[0].weight.detach()                                  # [inter, hidden + n_prompts]
        if weight.shape[1] != hidden + n_prompts:
            raise ValueError(f"prompt_kernel reads {weight.shape[1]} features, not d_model {hidden} + "
                             f"{n_prompts} prompts")
        self.prompt_hidden = nn.Linear(hidden, weight.shape[0])
        self.prompt_hidden.weight = nn.Parameter(weight[:, :hidden].clone())
        self.prompt_hidden.bias = nn.Parameter(kernel[0].bias.detach().clone())
        # Row p of this table is W_p[:, p]: the one-hot's contribution, looked up rather than multiplied.
        self.prompt_table = nn.Embedding.from_pretrained(weight[:, hidden:].t().contiguous(), freeze=True)
        self.prompt_act = kernel[1]
        self.prompt_out = kernel[2]

    def forward(self, waveform, length, prompt):
        features, feature_length = self.preprocessor(input_signal=waveform, length=length)
        # The preprocessor's centred STFT emits `n // hop + 1` frames, the last past `feature_length` and
        # zeroed by its own mask. Left in, it is one frame too many for the subsampling whenever the
        # valid count is a multiple of 8: the encoder then emits a frame past its own `encoded_len`,
        # NeMo's greedy decode stops before it and the driver -- which decodes every frame it is given
        # -- did not, writing a final "." NeMo never wrote (6 of 73 LibriSpeech clips). Dropped, the
        # encoder emits exactly `encoded_len` frames and every valid one is unchanged, because the
        # convolutions' zero padding reads the same zero the mask wrote. A STATIC negative index, so
        # the frame count stays an expression of `n_samples` (a shape-derived one would not).
        features = features[:, :, :-1]
        encoded, _ = self.encoder(audio_signal=features, length=feature_length)
        hidden = encoded.transpose(1, 2)                                    # [1, T, d_model]
        fused = self.prompt_hidden(hidden) + self.prompt_table(prompt).unsqueeze(1)
        return self.prompt_out(self.prompt_act(fused))


def chunked_masks(att_left, att_chunk, att_right, padding_length, max_audio_length):
    """NeMo's `ConformerEncoder._create_masks` for `att_context_style == "chunked_limited_with_rc"`,
    with the context sizes as TENSORS (frames, `[1]` floats) rather than the Python ints of
    `set_default_att_context_size`, so one graph serves every latency the stream is opened with.

    Frame `i` is in chunk `floor(i / chunk)` and sees frames `[chunk * c - left, chunk * c + chunk - 1 +
    right]`. NeMo also clamps that window to `[0, T - 1]`, which a comparison against `j` in `[0, T)`
    makes redundant. The floor is exact: `i / chunk` is correctly rounded and `i`, `chunk` are small
    integers, so a non-multiple is at least `1 / chunk` from the next integer. Returned as NeMo returns
    them: `(pad_mask, att_mask)`, both True where masked."""
    # Float by construction, through an input: a bare float `arange` reaches MIL as an int32 range and a
    # cast, and the cast did not survive to the outer products below -- the engine got an int32 matmul,
    # which ggml's CPU kernel refuses (`GGML_ASSERT(!widen || ...)`). Adding the zero `att_left * 0` keeps
    # both vectors f32 tensors of the graph.
    zero = att_left * 0.0
    frame = torch.arange(max_audio_length, dtype=torch.float32) + zero
    ones = frame * 0.0 + 1.0
    first = torch.floor(frame / att_chunk) * att_chunk
    # Every comparison between two `[T, T]` outer products against a ones vector, never a `[1, T]` against
    # a `[T, 1]`: ggml broadcasts only its second operand, so a two-sided broadcast aborts the engine's
    # comparison ops (it did, in `op_greater_equal`). `chunk_key_mask`'s form, for the same reason.
    keys = torch.matmul(ones.view(-1, 1), frame.view(1, -1))                                  # [i, j] = j
    start = torch.matmul((first - att_left).view(-1, 1), ones.view(1, -1))
    end = torch.matmul((first + att_chunk - 1.0 + att_right).view(-1, 1), ones.view(1, -1))
    visible = (keys >= start) & (keys <= end)
    valid = frame.unsqueeze(0) < padding_length.unsqueeze(-1).to(torch.float32)              # [B, T]
    # NeMo's `pad.unsqueeze(1).repeat([1, T, 1])` AND its transpose is the outer product of the valid
    # vector with itself (batch 1, which the step is).
    both = valid.to(torch.float32).view(-1)
    attend = (torch.matmul(both.view(-1, 1), both.view(1, -1)) > 0.5) & visible
    attend = attend.unsqueeze(0)
    return ~valid, ~attend


class UnifiedEncoderStepPhase(nn.Module):
    """`(waveform, length, prompt, att_left, att_chunk, att_right) -> [1, T, d_model]`: the encoder
    phase's computation over a stream buffer under the streaming mode's chunked attention mask
    (loom.cpp ADR-110). NeMo's buffered streaming runs the WHOLE model per step -- preprocessor
    (its per-feature normalisation over the buffer), encoder, prompt -- on `left + chunk + right`
    of audio; the mask is the one difference from the offline encoder. The convolutions stay
    full-width: the checkpoint declares `conv_context_style: dcc`, but NeMo 3.0 (and main) store that
    and never read it.

    The offline wrapper's modules under the offline wrapper's attribute names, so every weight is the
    `encoder` phase's own (aliased, not a second copy). The mask goes in by shadowing the encoder's
    `_create_masks` for the duration of this forward -- the trace runs NeMo's own encoder, layers and
    attention, and the shared `encoder` phase is untouched once it returns."""

    def __init__(self, wrapper: "_UnifiedEncoderWrapper"):
        super().__init__()
        for name in ("preprocessor", "encoder", "prompt_hidden", "prompt_table", "prompt_act", "prompt_out"):
            setattr(self, name, getattr(wrapper, name))

    def forward(self, waveform, length, prompt, att_left, att_chunk, att_right):
        encoder = self.encoder

        def masks(att_context_size, padding_length, max_audio_length, offset, device):
            if offset is not None:
                raise ValueError("the streaming step has no cache, so no offset")
            return chunked_masks(att_left, att_chunk, att_right, padding_length, max_audio_length)

        encoder._create_masks = masks
        try:
            return _UnifiedEncoderWrapper.forward(self, waveform, length, prompt)
        finally:
            del encoder._create_masks


@dataclass
class ASRNemotronUnifiedExportConfig(ASRParakeetExportConfig):
    """Nemotron-Labs-Unified-ASR, a `.nemo` restored through NeMo's own loader."""

    architecture: str = "nemotron-unified-asr"
    output_path: str = "nemotron-unified-asr.gguf"
    # Filled during `phases()`, READ off the restored checkpoint.
    default_prompt: Optional[int] = None
    prompt_dictionary: Optional[dict] = None

    __unchecked__ = {
        **ASRParakeetExportConfig.__unchecked__,
        "default_prompt": Unchecked("READ off `model_defaults.prompt_dictionary['auto']`, the "
                                    "language-agnostic prompt, during phases(); checked to be a row of "
                                    "the prompt table"),
        "prompt_dictionary": Unchecked("READ off `model_defaults.prompt_dictionary`, the checkpoint's own "
                                       "language -> prompt-id table; every id is checked to be a row of "
                                       "the table"),
    }

    def encoder_phase(self, model):
        import coremltools as ct

        from .nemo_asr_export import MAX_SECONDS, MIN_SECONDS, TRACE_SECONDS

        sample_rate = self.validate_encoder(model)
        table = dict(model.cfg.model_defaults.prompt_dictionary)
        n_prompts = int(model.num_prompts)
        bad = {k: v for k, v in table.items() if not 0 <= int(v) < n_prompts}
        if bad or "auto" not in table:
            raise ValueError(f"prompt table: ids outside its {n_prompts} rows {bad}, or no 'auto' entry")
        self.prompt_dictionary = {str(k): int(v) for k, v in table.items()}
        self.default_prompt = self.prompt_dictionary["auto"]
        if not model.concat:
            raise ValueError("this checkpoint does not concatenate its prompt (`concat` is False); the "
                             "leaf's encoder would apply a projection the reference skips")

        wrapper = _UnifiedEncoderWrapper(model).eval()
        _check_valid_frames(model, wrapper, sample_rate, self.default_prompt)
        n_samples = int(TRACE_SECONDS * sample_rate)
        dummy = (torch.randn(1, n_samples, dtype=torch.float32),
                 torch.tensor([n_samples], dtype=torch.int64),
                 torch.tensor([self.default_prompt], dtype=torch.int64))
        seq_dim = ct.RangeDim(int(MIN_SECONDS * sample_rate), int(MAX_SECONDS * sample_rate))
        mil_inputs = [ct.TensorType(name="waveform", shape=(1, seq_dim), dtype=np.float32),
                      ct.TensorType(name="length", shape=(1,), dtype=np.int32),
                      ct.TensorType(name="prompt", shape=(1,), dtype=np.int32)]
        phase = ExportPhase(name="encoder", wrapper=wrapper, dummy_inputs=dummy,
                            mil_inputs=mil_inputs, root_axis=self.root_axis)
        self._encoder_wrapper = wrapper
        self.stream_facts = _stream_facts(model, sample_rate)
        return phase, int(model.cfg.encoder.d_model)

    def phases(self) -> List[ExportPhase]:
        """The transducer template's four, then the live door's `encoder_step` (loom.cpp ADR-110): the
        encoder over a stream buffer under the streaming mode's chunked attention mask, its three
        context sizes inputs. `encoder` is unchanged."""
        import coremltools as ct

        from .nemo_asr_export import MAX_SECONDS, MIN_SECONDS

        phases = super().phases()
        rate = self.stream_facts["SAMPLE_RATE"]
        frame = self.stream_facts["FRAME_SAMPLES"]
        # The model card's default streaming buffer: 70 + 7 + 7 encoder frames (5.6 + 0.56 + 0.56 s).
        n = 84 * frame
        step = UnifiedEncoderStepPhase(self._encoder_wrapper).eval()
        phases.append(ExportPhase(
            name="encoder_step", wrapper=step,
            dummy_inputs=(torch.randn(1, n) * 0.1, torch.tensor([n], dtype=torch.int64),
                          torch.tensor([self.default_prompt], dtype=torch.int64),
                          torch.tensor([70.0]), torch.tensor([7.0]), torch.tensor([7.0])),
            mil_inputs=[
                ct.TensorType(name="waveform", shape=(1, ct.RangeDim(int(MIN_SECONDS * rate),
                                                                     int(MAX_SECONDS * rate))),
                              dtype=np.float32),
                ct.TensorType(name="length", shape=(1,), dtype=np.int32),
                ct.TensorType(name="prompt", shape=(1,), dtype=np.int32),
                ct.TensorType(name="att_left", shape=(1,), dtype=np.float32),
                ct.TensorType(name="att_chunk", shape=(1,), dtype=np.float32),
                ct.TensorType(name="att_right", shape=(1,), dtype=np.float32),
            ],
            root_axis=self.root_axis,
        ))
        return phases

    def driver_components(self) -> List:
        """The template's one-pass decode, and beside it the live door: the shared live decode
        (`nemotron_asr_driver/00_stream_decode.lua`) and NeMo's buffered loop
        (`02_unified_stream.lua`), top level so their constants are too."""
        from .driver_components import ExportConstants, LuaFragment
        from .nemo_asr_export import MAX_SECONDS
        from .nemotron_asr_export import stream_decode_fragment

        # The checkpoint's numbers after `phases()`; this release's before it (a catalogue build, which
        # never traces).
        facts = dict(getattr(self, "stream_facts", None) or {
            "SAMPLE_RATE": 16000, "FRAME_SAMPLES": 1280, "FEATURES_PER_SEC": 100.0, "SUBSAMPLING": 8})
        constants = {"S_" + k: v for k, v in facts.items()}
        constants.update({
            "S_MAX_BUFFER_SAMPLES": int(MAX_SECONDS * facts["SAMPLE_RATE"]),
            "S_DEFAULT_LEFT_SECONDS": STREAM_SECONDS[0], "S_DEFAULT_CHUNK_SECONDS": STREAM_SECONDS[1],
            "S_DEFAULT_RIGHT_SECONDS": STREAM_SECONDS[2],
            "S_BLANK_ID": self.blank_id if self.blank_id is not None else 4095,
            "S_PRED_HIDDEN": self.pred_hidden or 640, "S_N_PRED_LAYERS": self.num_pred_layers or 2,
            "S_MAX_SYMBOLS_PER_STEP": self.max_symbols_per_step,
            "S_DEFAULT_PROMPT": self.default_prompt if self.default_prompt is not None else 127,
        })
        if self.durations:
            raise ValueError("the unified stream decodes plain RNN-T; this checkpoint has durations")
        here = Path(__file__).resolve().parent / "nemotron_asr_driver"
        return [
            ExportConstants(values=constants, top_level=True),
            stream_decode_fragment(here, self.num_pred_layers),
            LuaFragment(here / "02_unified_stream.lua", top_level=True,
                        defines=("stream_open", "stream_feed", "stream_finish")),
        ] + super().driver_components()

    def encoder_inputs(self) -> dict:
        from .driver_ir import FieldAccess, Var

        return {"waveform": Var("_waveform"), "length": FieldAccess("inputs", "length"),
                "prompt": Var("_prompt")}

    def encoder_prelude(self) -> List:
        from .driver_components import ExportConstants, LuaFragment

        # Nemotron 3.5's fragment, verbatim: the same `inputs.language or DEFAULT_PROMPT`.
        return [
            ExportConstants(values={"DEFAULT_PROMPT": self.default_prompt}),
            LuaFragment(Path(__file__).resolve().parent / "nemotron_asr_driver" / "00_prompt.lua",
                        reads=("DEFAULT_PROMPT",), defines=("_prompt",)),
        ]

    def contract(self) -> dict:
        """The checkpoint's own language table, names -> prompt ids, `auto` included."""
        contract = super().contract()
        if self.prompt_dictionary:
            names = sorted(self.prompt_dictionary)
            contract["asr.language_names"] = names
            contract["asr.language_ids"] = [self.prompt_dictionary[n] for n in names]
        return contract


def _stream_facts(model, sample_rate: int) -> dict:
    """The numbers the buffered loop's arithmetic is in, as NeMo's streaming script derives them: the
    hop in samples (which it rounds to a multiple of the subsampling, a no-op here, refused otherwise),
    an encoder frame in samples, and the features per second its seconds-to-frames rounding uses."""
    stride = float(model.cfg.preprocessor.window_stride)
    factor = int(model.encoder.subsampling_factor)
    hop = int(sample_rate * stride)
    if hop % factor:
        raise ValueError(f"the hop ({hop} samples) is not a multiple of the subsampling ({factor}); NeMo's "
                         "streaming script would round it and the driver does not")
    return {"SAMPLE_RATE": sample_rate, "FRAME_SAMPLES": hop * factor, "FEATURES_PER_SEC": 1.0 / stride,
            "SUBSAMPLING": factor}


def _check_valid_frames(model, wrapper, sample_rate: int, prompt: int) -> None:
    """The wrapper's frame count against NeMo's own `encoded_len`, and its frames against NeMo's, at
    lengths whose valid mel count is and is not a multiple of the subsampling -- the case the dropped
    frame exists for (see `_UnifiedEncoderWrapper.forward`)."""
    hop = int(round(float(model.cfg.preprocessor.window_stride) * sample_rate))
    generator = torch.Generator().manual_seed(0)
    for mel_frames in (96, 97, 100):
        n = mel_frames * hop + hop // 2
        x = torch.randn(1, n, generator=generator) * 0.1
        length = torch.tensor([n])
        p = torch.tensor([prompt])
        with torch.inference_mode():
            ref, ref_len = model.forward(input_signal=x, input_signal_length=length, prompt_indices=p)
            got = wrapper(x, length, p)
        valid = int(ref_len[0])
        if got.shape[1] != valid:
            raise ValueError(f"{mel_frames} mel frames: the encoder phase emits {got.shape[1]} frames, "
                             f"NeMo's encoded_len is {valid}")
        diff = float((got[0] - ref[0, :, :valid].transpose(0, 1)).abs().max())
        if diff > 1e-4:
            raise ValueError(f"{mel_frames} mel frames: the encoder phase is {diff:.3e} from NeMo's")


def _is_nemotron_unified(path: Path) -> bool:
    from .nemo_asr_export import _is_nemo_archive, _read_nemo_model_config

    if not _is_nemo_archive(path):
        return False
    return str(_read_nemo_model_config(path).get("target", "")).endswith(TARGET)


def _build_nemotron_unified(path: Path, output_path: str) -> ASRNemotronUnifiedExportConfig:
    return ASRNemotronUnifiedExportConfig(checkpoint=str(path), output_path=output_path)


def register(registry) -> None:
    from .registry import ModelRecognizer, TaskRegistryEntry

    registry.register(TaskRegistryEntry(
        task="automatic-speech-recognition",
        config_class=ASRNemotronUnifiedExportConfig,
        recognizers=[ModelRecognizer(name="nemotron-unified-asr", detect=_is_nemotron_unified,
                                     build_config=_build_nemotron_unified)],
    ))
