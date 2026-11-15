"""A wav2vec 2.0-style CTC model (HuBERT, data2vec-audio, wav2vec 2.0) streamed the way the transformers ASR
pipeline chunks it (loom.cpp ADR-126): `pipeline("automatic-speech-recognition", chunk_length_s=...,
stride_length_s=...)`.

These checkpoints have no streaming mode. The pipeline's chunking is the reference transformers documents
for exactly them ("Making automatic speech recognition work on large files with Wav2Vec2"): chunks of the raw
waveform overlapping by a stride each side, each run through the model on its own, the strides' logit frames
dropped and one CTC merge over the rest. A chunk's ids depend on nothing after it, so a stream answers a
chunk once a sample past its end has arrived, and ends on the pipeline's ids.

The one-pass export is unchanged -- the same graph, weights and `infer` -- and the three stream functions sit
beside `infer` at top level (`ctc_chunk_stream_driver/00_stream.lua`). The graph already normalises the
waveform over whatever it is handed, which is the feature extractor's per-chunk normalisation.
"""
from pathlib import Path
from typing import List

DRIVER_DIR = Path(__file__).resolve().parent / "ctc_chunk_stream_driver"

# The live default: 4 s chunks, so a stride of 0.66 s each side by the pipeline's own rule (a sixth of the
# chunk, rounded to whole logit frames) and a chunk every 2.68 s. The pipeline itself has no default (0 turns
# chunking off).
DEFAULT_CHUNK_SECONDS = 4.0


def stream_components(*, rate: int, align: int, blank_id: int, min_samples: int, max_samples: int) -> List:
    """The constants and the fragment `CtcGreedyBuilder.stream` adds. `align` is the checkpoint's
    `inputs_to_logits_ratio` (samples per logit frame), which the pipeline rounds every size to."""
    from .driver_components import ExportConstants, LuaFragment

    return [
        ExportConstants(values={
            "CTC_RATE": int(rate), "CTC_ALIGN": int(align), "CTC_BLANK": int(blank_id),
            "CTC_MIN_SAMPLES": int(min_samples), "CTC_MAX_SAMPLES": int(max_samples),
            "CTC_DEFAULT_CHUNK_SECONDS": DEFAULT_CHUNK_SECONDS,
        }, top_level=True),
        LuaFragment(DRIVER_DIR / "00_stream.lua", top_level=True,
                    defines=("stream_open", "stream_feed", "stream_finish")),
    ]
