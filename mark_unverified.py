#!/usr/bin/env python3
"""Mark the test 1 words whose correctness the recording does not confirm.

Test 1 assumes Common Voice speakers read the prompt as written, and the CER filter
(build_tier1.CER_LIMIT) lets a clip through with up to 10% of it misread. A misread word
is then a "correct" word that was not said correctly: a system that flags it is right,
yet counts a false alarm. Each manifest item gets `unverified`, the indices of the
target words that overlap a place where Whisper's transcript differs from the prompt.
report.py leaves those words out as negatives; a planted error stays a positive, since
the audio does not say the target there either way.

Japanese is compared on kana readings (pyopenjtalk), so 書く vs 描く is no difference
and 詩 vs 歌 (し vs うた) is one. Other languages compare letters, case and punctuation
ignored, so "2" vs "two" counts as a difference: dropping a word costs nothing but data.

Usage:
    python mark_unverified.py data/tier1/manifest_*.jsonl
"""

from __future__ import annotations

import json
import re
import sys

from rapidfuzz.distance import Levenshtein

_LETTER = re.compile(r"\w", re.UNICODE)
_KATA = re.compile(r"[ァ-ヿ]")


def chars(text: str, lang: str) -> tuple[str, list[range]]:
    """Comparable characters of `text`, and for each the source characters it stands for.
    Japanese: the katakana reading, each kana standing for its whole morpheme."""
    if lang != "ja":
        idx = [i for i, c in enumerate(text) if _LETTER.match(c) and c != "_"]
        return "".join(text[i].lower() for i in idx), [range(i, i + 1) for i in idx]
    import pyopenjtalk
    out, src, cur = [], [], 0
    for m in pyopenjtalk.run_frontend(text):
        j = text.find(m["string"], cur)
        if j < 0:
            continue
        cur = j + len(m["string"])
        kana = _KATA.findall(m["pron"].replace("’", ""))
        out += kana
        src += [range(j, cur)] * len(kana)
    return "".join(out), src


def deviated(prompt: str, heard: str, lang: str) -> set[int]:
    """Prompt character offsets the transcript does not match (whole morphemes in ja)."""
    a, src = chars(prompt, lang)
    b, _ = chars(heard, lang)
    out = set()
    for op in Levenshtein.opcodes(a, b):
        if op.tag == "equal":
            continue
        # an insertion has no source characters: blame the one before it
        span = range(op.src_start, op.src_end) if op.src_end > op.src_start else \
            [max(op.src_start - 1, 0)] if a else []
        for k in span:
            out.update(src[k])
    return out


def target_spans(item: dict) -> list[tuple[int, int]]:
    if "units" in item:
        return [(u["start"], u["end"]) for u in item["units"]]
    spans, cur = [], 0
    for w in item["target"].split():
        j = item["target"].index(w, cur)
        spans.append((j, j + len(w)))
        cur = j + len(w)
    return spans


def unverified(item: dict) -> list[int]:
    dev = deviated(item["prompt"], item["whisper"]["transcript"], item["lang"])
    # prompt offsets -> target offsets; the planted edit itself maps nowhere
    to_target = set()
    for op in Levenshtein.opcodes(item["prompt"], item["target"]):
        if op.tag == "equal":
            to_target.update(op.dest_start + k for k in range(op.src_end - op.src_start)
                             if op.src_start + k in dev)
    return [i for i, (s, e) in enumerate(target_spans(item))
            if any(s <= k < e for k in to_target)]


def main() -> None:
    for path in sys.argv[1:]:
        with open(path, encoding="utf-8") as f:
            items = [json.loads(line) for line in f if line.strip()]
        n = 0
        for it in items:
            it["unverified"] = unverified(it)
            n += len(it["unverified"])
        with open(path, "w", encoding="utf-8") as f:
            f.writelines(json.dumps(it, ensure_ascii=False) + "\n" for it in items)
        print(f"{path}: {n} words unverified in {sum(bool(i['unverified']) for i in items)}"
              f"/{len(items)} clips")


if __name__ == "__main__":
    main()
