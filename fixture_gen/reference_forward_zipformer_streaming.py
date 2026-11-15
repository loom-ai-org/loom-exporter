"""Ground truth for the exported Zipformer streaming (`test_e2e_zipformer_streaming_mil_export`, loom.cpp).

From upstream's released jit file at the export's latency (`jit_model-chunk-32-left-128.pt`), run as its own
`zipformer inference --streaming 1` runs it: the clip's fbank, `DecodeStream`'s padding and chunking, the
scripted `StreamingEncoderWrapper` with its states, and `streaming_greedy_search`. Not this repo's wrappers.
For each clip:

    ref_waveform_{i}.npy    the mono f32 samples
    ref_encoder_{i}.npy     every chunk's `encoder` + `joiner.encoder_proj` rows, (n_chunks * 16, joiner_dim)
    ref_ids_{i}.npy         the greedy token ids
    ref_text_{i}.txt        `token_ids_to_text`
    ref_names.txt           the clip names

Needs the `pkufool/zipformer` clone (LOOM_ZIPFORMER_REPO). The checkpoint's two sample clips cover English
and Chinese; `samples/jfk.wav` is one neither was trained around.

    python fixture_gen/reference_forward_zipformer_streaming.py <model_dir> <out_dir> [more.wav ...]
"""
import math
import os
import sys
from pathlib import Path

# `SwooshL`/`SwooshR` go through `torch.compile` when run eagerly; the jit file does not, but the clone's
# modules are imported either way.
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")

import numpy as np
import soundfile as sf
import torch


def main() -> int:
    if len(sys.argv) < 3:
        print(__doc__)
        return 2
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from loom_exporter.zipformer_export import _read_tokens, import_zipformer
    from loom_exporter.zipformer_streaming_export import CHUNK_SIZE, LEFT_CONTEXT

    import_zipformer()
    from zipformer.decode.search import streaming_greedy_search
    from zipformer.decode.stream import DecodeStream
    from zipformer.utils.utils import AttributeDict, token_ids_to_text
    import torchaudio.compliance.kaldi as kaldi

    model_dir, out_dir = Path(sys.argv[1]), Path(sys.argv[2])
    out_dir.mkdir(parents=True, exist_ok=True)
    clips = [model_dir / "data" / "en.wav", model_dir / "data" / "zh.wav",
             Path(__file__).resolve().parents[1] / "samples" / "jfk.wav"]
    clips += [Path(p) for p in sys.argv[3:]]

    model = torch.jit.load(str(model_dir / f"jit_model-chunk-{CHUNK_SIZE}-left-{LEFT_CONTEXT}.pt")).eval()
    model.device = torch.device("cpu")
    params = AttributeDict({"blank_id": model.decoder.blank_id, "context_size": model.decoder.context_size,
                            "decoding_method": "greedy_search"})
    chunk_length = model.encoder.chunk_size * 2
    tail_length = chunk_length + model.encoder.pad_length
    pieces = dict(enumerate(_read_tokens(model_dir)))
    names = []
    for i, clip in enumerate(c for c in clips if c.is_file()):
        waveform, rate = sf.read(str(clip), dtype="float32")
        if rate != 16000 or waveform.ndim != 1:
            raise SystemExit(f"{clip}: expected mono 16 kHz, got {rate} Hz, shape {waveform.shape}")
        features = kaldi.fbank(torch.from_numpy(waveform)[None], num_mel_bins=80, sample_frequency=16000,
                               dither=0, snip_edges=False, high_freq=-400)
        # `_infer_jit_streaming` for one stream, keeping each chunk's encoder rows.
        stream = DecodeStream(params=params, utt_id=clip.stem, initial_states=model.encoder.get_init_states(),
                              pad_length=model.encoder.pad_length)
        stream.set_features(features)
        states, rows = stream.states, []
        with torch.no_grad():
            while not stream.done:
                feat, _ = stream.get_feature_frames(chunk_length)
                feat = torch.nn.functional.pad(feat, (0, 0, 0, tail_length - feat.shape[0]),
                                               value=math.log(1e-10))
                encoder_out, _, states = model.encoder(features=feat[None],
                                                       feature_lengths=torch.tensor([tail_length]),
                                                       states=states)
                encoder_out = model.joiner.encoder_proj(encoder_out)
                rows.append(encoder_out[0])
                streaming_greedy_search(model=model, encoder_out=encoder_out, streams=[stream])
        ids = stream.hyp[params.context_size:]
        projected = torch.cat(rows)
        text = token_ids_to_text(ids, pieces)
        np.save(out_dir / f"ref_waveform_{i}.npy", waveform)
        np.save(out_dir / f"ref_encoder_{i}.npy", projected.numpy().astype(np.float32))
        np.save(out_dir / f"ref_ids_{i}.npy", np.asarray(ids, dtype=np.float32))
        (out_dir / f"ref_text_{i}.txt").write_text(text, encoding="utf-8")
        names.append(clip.stem)
        print(f"{clip.name}: {waveform.shape[0] / 16000:.2f} s, {len(rows)} chunks, {len(ids)} tokens: {text}")
    (out_dir / "ref_names.txt").write_text("\n".join(names) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
