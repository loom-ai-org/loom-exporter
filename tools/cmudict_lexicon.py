#!/usr/bin/env python3
"""CMUdict -> the `word<TAB>ipa` lexicon AuroraTTS-100K's card ships (`lexicons/en_US-cmudict.tsv`).

    python tools/cmudict_lexicon.py cmudict.dict en_US-cmudict.tsv

`cmudict.dict` is cmusphinx/cmudict's (BSD-2-Clause); the card's was commit 7479086. Each word keeps its
FIRST pronunciation (`word(2)` alternates are dropped), which is the one g2p_en -- the model's training
front end -- looks up.

The IPA is written for one reader, loom-py's `arpabet` fold (`loom.phonemizers`), and the fold returns
every entry's ARPAbet exactly (135,166 of 135,166 on that commit). Two choices make that so:
* the stress mark goes right before its vowel, and an unstressed vowel carries none (AH0 is `ə`, ER0 `ɚ`);
* a `.` marks a syllable break wherever two phones would otherwise spell a third -- two vowels in
  hiatus (`AO IH0` is not `OY`), a schwa before R (`AH0 R` is not `ER0`), and `T SH`/`D ZH` (not
  `CH`/`JH`). The fold reads `.` as a boundary and emits nothing for it.
"""
import sys

ARPA_TO_IPA = {"AA": "ɑ", "AE": "æ", "AH": "ʌ", "AO": "ɔ", "AW": "aʊ", "AY": "aɪ", "EH": "ɛ", "ER": "ɝ",
               "EY": "eɪ", "IH": "ɪ", "IY": "i", "OW": "oʊ", "OY": "ɔɪ", "UH": "ʊ", "UW": "u",
               "B": "b", "CH": "tʃ", "D": "d", "DH": "ð", "F": "f", "G": "ɡ", "HH": "h", "JH": "dʒ",
               "K": "k", "L": "l", "M": "m", "N": "n", "NG": "ŋ", "P": "p", "R": "ɹ", "S": "s", "SH": "ʃ",
               "T": "t", "TH": "θ", "V": "v", "W": "w", "Y": "j", "Z": "z", "ZH": "ʒ"}


def to_ipa(phones) -> str:
    out, prev = [], None
    for phone in phones:
        base, stress = (phone[:-1], phone[-1]) if phone[-1].isdigit() else (phone, None)
        if prev is not None and ((stress is not None and prev[-1].isdigit())
                                 or (prev == "AH0" and base == "R")
                                 or (prev in ("T", "D") and base in ("SH", "ZH"))):
            out.append(".")
        prev = phone
        symbol = "ə" if (base, stress) == ("AH", "0") else "ɚ" if (base, stress) == ("ER", "0") else ARPA_TO_IPA[base]
        out.append({"1": "ˈ", "2": "ˌ"}.get(stress, "") + symbol)
    return "".join(out)


def main(src: str, dst: str) -> None:
    words = {}
    with open(src, encoding="utf-8") as f:
        for line in f:
            line = line.split("#")[0].strip()
            if not line:
                continue
            word, *phones = line.split()
            if "(" not in word:
                words.setdefault(word.lower(), to_ipa(phones))
    with open(dst, "w", encoding="utf-8") as f:
        for word, ipa in words.items():
            f.write(f"{word}\t{ipa}\n")
    print(f"{len(words)} words -> {dst}")


if __name__ == "__main__":
    main(*sys.argv[1:3])
