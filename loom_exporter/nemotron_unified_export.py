"""NVIDIA Nemotron-Labs-Unified-ASR (`EncDecRNNTBPEModelWithPrompt`): a unified offline/streaming
FastConformer RNN-T with language-ID prompt conditioning, restored from its `.nemo` -- the fourth
transducer leaf, Parakeet's loader with Nemotron 3.5's prompt.

**Needs NeMo 3.0** (`~/.venvs/nemo3`): the prompt model class and the unified encoder's options
(`att_chunk_context_size`, `chunked_limited_with_rc` attention, `dcc` convolutions) are not in the 2.6
the `piper` venv pins for every other `.nemo` export. On Python 3.11 that venv must be >= 3.11.4, whose
`tarfile.extract` takes the `filter=` NeMo 3.0 passes it.

**What is exported is the OFFLINE mode**, which is what loom's one-pass `speech2text` door is: the
encoder at its restored `att_context_size = [-1, -1, -1]` (full-context attention, the non-causal
convolutions it was trained with for offline decoding), then Parakeet's greedy transducer loop. The
buffered streaming pipeline (5.6 s of left context recomputed per chunk, chunked masks) is not.

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
        return phase, int(model.cfg.encoder.d_model)

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
