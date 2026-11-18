#!/usr/bin/env python3
"""misaki's pronunciation dictionaries as a `word<TAB>ipa` lexicon (`loom.phonemizers.set_lexicon`'s shape).

    python tools/misaki_lexicon.py us en_US-misaki.tsv
    python tools/misaki_lexicon.py gb en_GB-misaki.tsv

`<region>_gold.json` over `<region>_silver.json` (gold wins), each word's default reading (a dict's
`DEFAULT`), sorted by word. The US file this writes is byte-identical to the `lexicons/en_US-misaki.tsv`
the Vocetta2, sanoTTS nano and Vaniq-Edge cards ship (checked 2026-11-18, 183,561 words); the GB one
(197,116 words) is what Paradee's British voices fall back on without misaki installed. Apache-2.0, as
misaki's data.
"""
import json
import os
import sys

import misaki

DATA = os.path.join(os.path.dirname(misaki.__file__), "data")


def main(region: str, out: str) -> None:
    merged = {}
    for name in (f"{region}_silver.json", f"{region}_gold.json"):
        with open(os.path.join(DATA, name), encoding="utf-8") as f:
            for word, value in json.load(f).items():
                if isinstance(value, dict):
                    value = value.get("DEFAULT")
                if isinstance(value, str) and value:
                    merged[word] = value
    with open(out, "w", encoding="utf-8") as f:
        for word in sorted(merged):
            f.write(f"{word}\t{merged[word]}\n")
    print(f"{out}: {len(merged)} words")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
