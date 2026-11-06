"""Ground truth for the exported Zipformer (`test_e2e_zipformer_mil_export`, loom.cpp).

From the reference's own `AsrModel` and `greedy_search`, not this repo's wrappers, for each clip:

    ref_waveform_{i}.npy    the mono f32 samples
    ref_encoder_{i}.npy     `forward_encoder` + `joiner.encoder_proj`, (T, joiner_dim)
    ref_ids_{i}.npy         the greedy token ids
    ref_text_{i}.txt        `token_ids_to_text`
    ref_names.txt           the clip names

Needs the `pkufool/zipformer` clone (LOOM_ZIPFORMER_REPO). The checkpoint's two sample clips cover English
and Chinese; `samples/jfk.wav` is one neither was trained around.

    python fixture_gen/reference_forward_zipformer.py <model_dir> <out_dir> [more.wav ...]
"""
import os
import sys
from pathlib import Path

# `SwooshL`/`SwooshR` go through `torch.compile` when run eagerly, which needs a working inductor toolchain
# and has nothing to do with the numbers: the traced branch is the same formula.
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")

import numpy as np
import soundfile as sf
import torch


def main() -> int:
    if len(sys.argv) < 3:
        print(__doc__)
        return 2
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from loom_exporter.zipformer_export import ZipformerExportConfig, _read_tokens, import_zipformer

    import_zipformer()
    from zipformer.decode.search import greedy_search
    from zipformer.utils.utils import token_ids_to_text
    import torchaudio.compliance.kaldi as kaldi

    model_dir, out_dir = Path(sys.argv[1]), Path(sys.argv[2])
    out_dir.mkdir(parents=True, exist_ok=True)
    clips = [model_dir / "data" / "en.wav", model_dir / "data" / "zh.wav",
             Path(__file__).resolve().parents[1] / "samples" / "jfk.wav"]
    clips += [Path(p) for p in sys.argv[3:]]

    model = ZipformerExportConfig(model_dir=str(model_dir)).load_model()
    pieces = dict(enumerate(_read_tokens(model_dir)))
    names = []
    for i, clip in enumerate(c for c in clips if c.is_file()):
        waveform, rate = sf.read(str(clip), dtype="float32")
        if rate != 16000 or waveform.ndim != 1:
            raise SystemExit(f"{clip}: expected mono 16 kHz, got {rate} Hz, shape {waveform.shape}")
        x = torch.from_numpy(waveform)[None]
        features = kaldi.fbank(x, num_mel_bins=80, sample_frequency=16000, dither=0, snip_edges=False,
                               high_freq=-400)
        with torch.no_grad():
            encoder_out, lengths = model.forward_encoder(features[None], torch.tensor([features.shape[0]]))
            projected = model.joiner.encoder_proj(encoder_out)
            ids = greedy_search(model, encoder_out, lengths)[0]
        text = token_ids_to_text(ids, pieces)
        np.save(out_dir / f"ref_waveform_{i}.npy", waveform)
        np.save(out_dir / f"ref_encoder_{i}.npy", projected[0].numpy().astype(np.float32))
        np.save(out_dir / f"ref_ids_{i}.npy", np.asarray(ids, dtype=np.float32))
        (out_dir / f"ref_text_{i}.txt").write_text(text, encoding="utf-8")
        names.append(clip.stem)
        print(f"{clip.name}: {waveform.shape[0] / 16000:.2f} s, {projected.shape[1]} frames, {len(ids)} tokens: {text}")
    (out_dir / "ref_names.txt").write_text("\n".join(names) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
