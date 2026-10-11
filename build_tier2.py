#!/usr/bin/env python3
"""Build Tier 2: speechocean762 — real non-native English with human phoneme scores.

Tier 1 perturbs the target on clean native audio, which measures text-audio mismatch.
Tier 2 measures the actual product question: does the API agree with a human rater about
how well a learner pronounced a word they were genuinely trying to say? An ASR baseline
does well on the first and should collapse on the second, because real L2 errors
frequently leave word identity intact.

speechocean762 is OpenSLR SLR101, CC-BY-4.0, so it can be redistributed alongside the
benchmark. Word `accuracy` is a human 0-10 score; we keep it as the regression target
and also binarise it for detection F1.

Usage:
    python build_tier2.py --out data/tier2 --n 500
"""

from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
import tarfile
import urllib.request
from pathlib import Path

URL = "https://www.openslr.org/resources/101/speechocean762.tar.gz"
MISPRONOUNCED_AT = 6  # human word accuracy <= this counts as mispronounced (0-10 scale)


def fetch(cache: Path) -> Path:
    root = cache / "speechocean762"
    if root.exists():
        return root
    cache.mkdir(parents=True, exist_ok=True)
    tgz = cache / "speechocean762.tar.gz"
    if not tgz.exists():
        print(f"downloading {URL} (~520 MB)", file=sys.stderr)
        urllib.request.urlretrieve(URL, tgz)
    with tarfile.open(tgz) as t:
        t.extractall(cache, filter="data")
    return root


def read_scp(path: Path) -> dict[str, str]:
    out = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            uid, rest = line.split(maxsplit=1)
            out[uid] = rest
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("data/tier2"))
    ap.add_argument("--cache", type=Path, default=Path("data/_cache"))
    ap.add_argument("--split", default="test", choices=["test", "train"])
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    root = fetch(args.cache)
    scores = json.loads((root / "resource" / "scores.json").read_text(encoding="utf-8"))
    texts = read_scp(root / args.split / "text")
    wavs = read_scp(root / args.split / "wav.scp")

    uids = sorted(u for u in texts if u in scores and u in wavs)
    random.Random(args.seed).shuffle(uids)
    uids = uids[: args.n]

    (args.out / "audio").mkdir(parents=True, exist_ok=True)
    manifest = []
    for uid in uids:
        s = scores[uid]
        src = root / wavs[uid]
        dst = args.out / "audio" / f"{uid}.wav"
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(src),
                        "-ac", "1", "-ar", "16000", str(dst)], check=True)
        manifest.append({
            "id": uid, "lang": "en", "audio": f"audio/{uid}.wav",
            # Reference text is what the learner was asked to read, so scripted mode
            # applies exactly as in Tier 1 and the two tiers stay comparable.
            "target": texts[uid], "prompt": texts[uid], "label": None,
            "human": {
                "utterance": {k: s.get(k) for k in
                              ("accuracy", "completeness", "fluency", "prosodic", "total")},
                "words": [{"text": w["text"], "accuracy": w["accuracy"],
                           "stress": w.get("stress"), "total": w.get("total"),
                           "mispronounced": w["accuracy"] <= MISPRONOUNCED_AT,
                           "phones": w.get("phones"),
                           "phone_accuracy": w.get("phones-accuracy")}
                          for w in s["words"]],
            },
        })

    path = args.out / "manifest_speechocean762.jsonl"
    with open(path, "w", encoding="utf-8") as f:
        for m in manifest:
            f.write(json.dumps(m, ensure_ascii=False) + "\n")
    n_bad = sum(w["mispronounced"] for m in manifest for w in m["human"]["words"])
    n_all = sum(len(m["human"]["words"]) for m in manifest)
    print(f"wrote {len(manifest)} utterances, {n_bad}/{n_all} words mispronounced "
          f"(human accuracy <= {MISPRONOUNCED_AT})", file=sys.stderr)


if __name__ == "__main__":
    main()
