#!/usr/bin/env python3
"""Three extra reference fixtures for the MIL-traced SupertonicTTS export (supertonic_export.py).

The first two exist because the export's `dp`/`vfe` topologies were traced at exactly ten text
positions (see that script's own module docstring) -- the EXISTING
`reference_forward_supertonic_dp.py` (T=12) and `reference_forward_supertonic_vfe.py` (T=6) fixtures
don't match that shape, so this dumps the same real modules' own
`.forward()`/`.compute_velocity()` again at T=10 instead. `decoder` needs no new fixture: it never
touches the text axis at all.

The third (`real_*`, added by BACKLOG.md P4.6) is a different kind of check. Ten ids is the empty
string after the `<lang>` wrap, so the first two say nothing about text a person would actually
synthesize -- and now that the text axis is padded, the interesting case is precisely one where the
real/padding boundary sits in the MIDDLE of the axis rather than at position ten of ten. So this runs
the REAL `TextVectorizer` over a real sentence and dumps both encoders' output at that length,
UNPADDED. Unpadded is the point: `TextVectorizer.tokenize` pads only to the longest string in its
batch, so a batch of one -- which is what synthesis is -- is never padded, and this is therefore the
answer the reference implementation itself produces for that sentence. The engine-side comparison
feeds the same ids padded to the traced axis and has to reproduce it.

Usage: python3 reference_forward_supertonic_mil_extra.py <supertonic-tts repo root> <out_dir>
"""
import sys
from pathlib import Path

import numpy as np
import torch

# Long enough that the real/padding boundary lands mid-axis (161 ids against a 256-wide axis), which
# is what the ten-id fixtures cannot exercise, and fixed so the engine-side test can assert its own
# tokenizer produced the same ids before comparing anything numeric.
REAL_TEXT = ("Supertonic is a text to speech model, and this sentence is deliberately long enough "
             "to be a realistic test of what a single synthesis call has to carry.")


def main() -> None:
    if len(sys.argv) < 3:
        print(f"usage: {sys.argv[0]} <supertonic-tts root> <out_dir>", file=sys.stderr)
        sys.exit(1)
    repo_root, out_dir = Path(sys.argv[1]), Path(sys.argv[2])
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- dp, T=10 (same recipe as reference_forward_supertonic_dp.py, T=12 there) ---
    dp = torch.load(repo_root / "assets/pt/duration_predictor.pt", weights_only=False, map_location="cpu")
    dp_se = torch.load(repo_root / "assets/pt/dp-style-encoder.pt", weights_only=False, map_location="cpu")
    dp.eval()
    dp_se.eval()

    torch.manual_seed(0)
    T = 10
    txt_ids = torch.randint(1, 163, (1, T), dtype=torch.int64)
    txt_msk = torch.ones(1, 1, T)
    lat_crop = torch.randn(1, 144, 50)
    with torch.no_grad():
        stl_emb = dp_se(lat_crop)
        duration = dp(txt_ids, stl_emb, txt_msk)

    txt_ids.numpy().astype(np.int32).tofile(out_dir / "dp_mil_txt_ids.bin")
    stl_emb.numpy().astype(np.float32).tofile(out_dir / "dp_mil_stl_emb.bin")
    duration.numpy().astype(np.float32).tofile(out_dir / "dp_mil_expected_duration.bin")
    print(f"dp T={T}: duration={duration.item():.6f}")

    # --- vfe, T=10 (same recipe as reference_forward_supertonic_vfe.py, T=6 there), L=9 unchanged ---
    ve = torch.load(repo_root / "assets/pt/vector_estimator.pt", weights_only=False, map_location="cpu")
    ve.eval()

    torch.manual_seed(0)
    L, T2 = 9, 10
    z_t = torch.randn(1, 144, L)
    txt_emb = torch.randn(1, 256, T2)
    stl_emb2 = torch.randn(1, 50, 256)
    lat_msk = torch.ones(1, 1, L)
    txt_msk2 = torch.ones(1, 1, T2)
    t = torch.tensor([0.3])
    with torch.no_grad():
        v = ve.compute_velocity(z_t, txt_emb, stl_emb2, lat_msk, txt_msk2, t)

    z_t.numpy().astype(np.float32).tofile(out_dir / "vfe_mil_z_t.bin")
    txt_emb.numpy().astype(np.float32).tofile(out_dir / "vfe_mil_txt_emb.bin")
    stl_emb2.numpy().astype(np.float32).tofile(out_dir / "vfe_mil_stl_emb.bin")
    v.numpy().astype(np.float32).tofile(out_dir / "vfe_mil_expected_v.bin")
    print(f"vfe L={L}, T={T2}: v mean_abs={v.abs().mean().item():.4f}")

    # --- real text, both text encoders, UNPADDED at its own length (see module docstring) ---
    from supertonic_tts.models.modules.text_vectorizer import TextVectorizer

    te = torch.load(repo_root / "assets/pt/text_encoder.pt", weights_only=False, map_location="cpu")
    ttl_se = torch.load(repo_root / "assets/pt/ttl-style-encoder.pt", weights_only=False,
                        map_location="cpu")
    te.eval()
    ttl_se.eval()

    real_ids, real_msk = TextVectorizer().tokenize([REAL_TEXT])  # (1,T), (1,1,T) -- all ones, B=1
    T3 = real_ids.shape[1]
    torch.manual_seed(0)
    lat_crop3 = torch.randn(1, 144, 50)
    with torch.no_grad():
        dp_stl3 = dp_se(lat_crop3)
        ttl_stl3 = ttl_se(lat_crop3)
        real_duration = dp(real_ids, dp_stl3, real_msk)
        real_txt_emb = te(real_ids, ttl_stl3, real_msk)  # (1, 256, T3)

    real_ids.numpy().astype(np.int32).tofile(out_dir / "real_txt_ids.bin")
    dp_stl3.numpy().astype(np.float32).tofile(out_dir / "real_dp_stl_emb.bin")
    ttl_stl3.numpy().astype(np.float32).tofile(out_dir / "real_ttl_stl_emb.bin")
    real_duration.numpy().astype(np.float32).tofile(out_dir / "real_expected_duration.bin")
    real_txt_emb.numpy().astype(np.float32).tofile(out_dir / "real_expected_txt_emb.bin")
    print(f"real text T={T3}: duration={real_duration.item():.6f}, "
          f"mean_abs_txt_emb={real_txt_emb.abs().mean().item():.4f}")


if __name__ == "__main__":
    main()
