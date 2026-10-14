#!/usr/bin/env python3
"""Score the vendor responses.

Primary metric is word-level and threshold-free. Lingolix returns 0-1 and Azure 0-100;
Lingolix v3's HTTP response stops at the syllable while Azure reaches the phoneme. Word
score is the one granularity everyone provides, and ranking-based metrics need no
threshold, which removes the "the vendor picked a flattering cut-off" objection.

False-alarm rate is reported at a *matched operating point* (the threshold giving each
vendor the same recall) rather than at each vendor's default, because the defaults are
not documented comparably.

Usage:
    python report.py --manifest data/tier1/manifest_en.jsonl --responses data/responses
"""

from __future__ import annotations

import argparse
import json
import math
import random
import re
import sys
from collections import defaultdict
from pathlib import Path

from rapidfuzz.distance import Levenshtein

_PUNCT = re.compile(r"[^\w\s]", re.UNICODE)
BOOTSTRAP = 1000


def key(w: str) -> str:
    return _PUNCT.sub("", w).lower()


# ── Metrics (no sklearn; these are small and exact) ────────────────────────────

def roc_auc(labels: list[int], scores: list[float]) -> float:
    """Mann-Whitney U with ties averaged. `scores` is the error signal, higher = worse."""
    n1 = sum(labels)
    n0 = len(labels) - n1
    if not n1 or not n0:
        return math.nan
    order = sorted(range(len(scores)), key=lambda i: scores[i])
    ranks = [0.0] * len(scores)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and scores[order[j + 1]] == scores[order[i]]:
            j += 1
        avg = (i + j) / 2 + 1
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    s = sum(r for r, l in zip(ranks, labels) if l)
    return (s - n1 * (n1 + 1) / 2) / (n1 * n0)


def average_precision(labels: list[int], scores: list[float]) -> float:
    n1 = sum(labels)
    if not n1:
        return math.nan
    order = sorted(range(len(scores)), key=lambda i: -scores[i])
    tp = 0
    ap = 0.0
    for k, i in enumerate(order, 1):
        if labels[i]:
            tp += 1
            ap += tp / k
    return ap / n1


def fpr_at_recall(labels: list[int], scores: list[float], target: float) -> float:
    """False-alarm rate on correct words at the threshold hitting `target` recall."""
    pos = sorted((s for s, l in zip(scores, labels) if l), reverse=True)
    neg = [s for s, l in zip(scores, labels) if not l]
    if not pos or not neg:
        return math.nan
    thr = pos[min(int(math.ceil(target * len(pos))) - 1, len(pos) - 1)]
    return sum(1 for s in neg if s >= thr) / len(neg)


def bootstrap_ci(units: list[tuple[list[int], list[float]]], fn, seed=0) -> tuple[float, float]:
    """Resample whole utterances, not words: words within an utterance are correlated."""
    rng = random.Random(seed)
    vals = []
    for _ in range(BOOTSTRAP):
        pick = [units[rng.randrange(len(units))] for _ in range(len(units))]
        labels = [l for u in pick for l in u[0]]
        scores = [s for u in pick for s in u[1]]
        v = fn(labels, scores)
        if not math.isnan(v):
            vals.append(v)
    if not vals:
        return math.nan, math.nan
    vals.sort()
    return vals[int(0.025 * len(vals))], vals[min(int(0.975 * len(vals)), len(vals) - 1)]


# ── Aligning vendor output to the target words ─────────────────────────────────

def align(target_words: list[str], vendor_words: list[dict]) -> list[dict | None]:
    """Map each target word to the vendor's entry for it, or None.

    Vendors do not always return one entry per target word: Azure with EnableMiscue adds
    entries for inserted speech. Aligning on the word strings is more robust than
    trusting positions, and a target word the vendor never reported counts as flagged.
    """
    a = [key(w) for w in target_words]
    b = [key(w.get("text", "")) for w in vendor_words]
    out: list[dict | None] = [None] * len(a)
    for op in Levenshtein.opcodes(a, b):
        if op.tag in ("equal", "replace"):
            for k, i in enumerate(range(op.src_start, op.src_end)):
                j = op.dest_start + k
                if j < op.dest_end:
                    out[i] = vendor_words[j]
    return out


def align_spans(text: str, units: list[dict], vendor_words: list[dict],
                pick=None) -> list[dict | None]:
    """Japanese: map each reference unit to the vendor's lowest-scoring overlapping word.

    With no spaces, vendors segment the same target differently (Azure とりあえ|ず|生ビール|を,
    Lingolix とりあえず|生ビールを). Each vendor word is located in the target by character
    position, and every unit ({text,start,end}, from build_tier1_ja) takes the minimum score
    of the words overlapping it, so all vendors are judged at one granularity. Words that
    cannot be located (an insertion pseudo-word "…", text the vendor normalised) are skipped;
    a unit no located word covers counts as flagged, like any unreported target word.
    `pick` reduces the overlapping words to one entry (default: the lowest score).
    """
    spans, cur = [], 0
    for w in vendor_words:
        t = _PUNCT.sub("", w.get("text", ""))
        j = text.find(t, cur) if t else -1
        if j >= 0:
            spans.append((j, j + len(t), w))
            cur = j + len(t)
    out: list[dict | None] = []
    for u in units:
        hits = [w for s, e, w in spans if s < u["end"] and e > u["start"]]
        if not hits:
            out.append(None)
        elif pick:
            out.append(pick(hits))
        else:
            out.append(min(hits, key=lambda w: float(w.get("score", 0.0))))
    return out


# ── Verdicts: each vendor's own yes/no, no threshold of ours ───────────────────

# French: mid-vowel height and length are regional or free, brin/brun merged, schwa
# elides, glides alternate with vowels. Spanish: seseo, yeísmo, spirantised stops, j as
# x or h, s aspirated or dropped before a consonant or at word end.
_MERGERS_FR = {"e": "ɛ", "o": "ɔ", "ø": "œ", "ɑ": "a", "œ̃": "ɛ̃",
               "ɥ": "y", "w": "u", "j": "i"}
_MERGERS_ES = {"θ": "s", "ʎ": "i", "ʝ": "i", "j": "i", "w": "u", "β": "b", "ð": "d",
               "ɣ": "ɡ", "h": "x", "ŋ": "n"}
_VOWELS = set("iyɨʉɯuɪʏʊeøɘɵɤoəɛœɜɝɞʌɔæɐaɶɑɒ")
_FT = None


def _segs(ipa: str) -> tuple[str, ...]:
    global _FT
    if _FT is None:
        import panphon
        _FT = panphon.FeatureTable()
    return tuple(_FT.ipa_segs(ipa))


def _merge(phones: tuple[str, ...], lang: str) -> tuple[str, ...]:
    """Collapse phones that merge in common French / Spanish accents."""
    if lang == "fr":
        return tuple(_MERGERS_FR.get(q, q) for q in (p.replace("ː", "") for p in phones)
                     if q != "ə")
    m = [_MERGERS_ES.get(q, q) for q in phones]
    return tuple(q for i, q in enumerate(m)
                 if not (q == "s" and (i == len(m) - 1 or m[i + 1] not in _VOWELS)))


def _canon(ipa: str, lang: str) -> str:
    """Japanese mora-level spelling of an IPA string, so notation is not an error.

    Expected and detected IPA can write the same sound differently
    (ふ fɯ/ɸɯ, ひ hi/çi, ん as ŋ/ɲ/m/n/ɴ, devoicing, ʲ/j). Comparing in one alphabet per
    sound class keeps what matters -- ː and ʔ stay, so length still counts.
    """
    if lang in ("fr", "es"):
        # Accent variation is not an error (schwa, seseo, yeísmo, ...): see _merge.
        return "".join(_merge(_segs(ipa), lang))
    if lang != "ja":
        return ipa
    s = ipa.replace("\u0325", "").replace("ʲ", "j").replace("ɸ", "f").replace("ç", "h")
    return re.sub("[ɴŋɲm]|n(?![aieoɯ])", "N", s)


def vendor_flags(rec: dict, lang: str) -> list[bool] | None:
    """Per vendor word: did the vendor itself call it an error?

    Azure and others: a non-empty ErrorType. Lingolix: additionally, the phones it heard
    differ from the phones it expected anywhere in the word (compared per word, so a
    phone assigned to the neighbouring mora is not a mismatch). None when undefined:
    phone comparison needs a canonical alphabet, and only Japanese has one so far --
    raw IPA strings differ by notation alone (53% "false alarms" on de).
    """
    words = rec.get("words", [])
    if rec.get("vendor") == "whisper_diff":
        # Its call is "the transcript differs from the target word"; it sets no ErrorType.
        return [float(w.get("score", 0.0)) < 1.0 for w in words]
    raw_words = (rec.get("raw") or {}).get("words")
    parallel = isinstance(raw_words, list) and len(raw_words) == len(words)
    out = []
    for k, w in enumerate(words):
        flag = w.get("error_type") not in (None, "None")
        syl = [x for x in (raw_words[k].get("syllables") or []) if x.get("is_scored", True)] \
            if parallel else []
        if syl and lang not in ("ja", "fr", "es"):
            return None
        if syl:
            exp = _canon("".join(x.get("expected_ipa") or "" for x in syl), lang)
            det = _canon("".join(x.get("detected_ipa") or "" for x in syl), lang)
            flag = flag or exp != det
        out.append(flag)
    return out


def _any_flag(hits: list[dict]) -> dict:
    return {"flag": any(h["flag"] for h in hits)}


def error_signal(entry: dict | None) -> float:
    # A target word the vendor never scored is maximally suspicious, not free.
    return 1.0 if entry is None else 1.0 - float(entry.get("score", 0.0))


# ── Assembly ───────────────────────────────────────────────────────────────────

def utterance_score(rec: dict) -> float | None:
    """The vendor's own sentence score (Lingolix `overall`, Azure PronScore, TonePerfect
    scores.overall); None falls back to the mean of its word scores."""
    raw = rec.get("raw") or {}
    if isinstance(raw.get("overall"), (int, float)):  # Lingolix
        return float(raw["overall"])
    if isinstance(raw.get("accuracy"), (int, float)):
        return float(raw["accuracy"])
    tp = raw.get("scores") or {}
    if isinstance(tp.get("overall"), (int, float)):  # TonePerfect
        return float(tp["overall"]) / 100
    nbest = raw.get("NBest") or [{}]
    pa = nbest[0].get("PronunciationAssessment") or nbest[0]
    return float(pa["PronScore"]) if "PronScore" in pa else None


def collect(manifest: list[dict], responses: dict[str, dict]) -> dict:
    """Per vendor: word-level units, localisation hits, latencies, human-score pairs.

    Tier 2 carries human ratings instead of a synthetic perturbation, so the positive
    label becomes "a human judged this word mispronounced". Every word-level metric then
    applies unchanged, and the correlation pairs are collected alongside.
    """
    units, loc_hit, loc_n, latency = [], 0, 0, []
    word_pairs, utt_pairs = [], []
    for item in manifest:
        rec = responses.get(item["id"])
        if not rec or rec.get("error"):
            continue
        latency.append(rec.get("latency_ms", math.nan))
        vflags = vendor_flags(rec, item.get("lang", ""))
        vwords = [{**w, "flag": f} for w, f in
                  zip(rec.get("words", []), vflags or [False] * len(rec.get("words", [])))]
        if "units" in item:  # Japanese: character-span units, see align_spans
            target = [u["text"] for u in item["units"]]
            aligned = align_spans(item["target"], item["units"], vwords)
            judged = align_spans(item["target"], item["units"], vwords, pick=_any_flag)
        else:
            target = item["target"].split()
            aligned = judged = align(target, vwords)
        scores = [error_signal(e) for e in aligned]
        # A target word the vendor never reported counts as flagged, as for scores.
        flags = None if vflags is None else \
            [True if e is None else bool(e["flag"]) for e in judged]

        human = item.get("human")
        if human:
            hw = human["words"]
            if len(hw) != len(target):
                continue  # tokenisation mismatch; drop rather than mislabel
            labels = [int(w["mispronounced"]) for w in hw]
            for w, e in zip(hw, aligned):
                if e is not None:
                    word_pairs.append((float(e.get("score", 0.0)), float(w["accuracy"])))
            total = (human.get("utterance") or {}).get("total")
            vendor_utt = [1.0 - x for x in scores]
            served = utterance_score(rec)
            if total is not None and (served is not None or vendor_utt):
                utt = served if served is not None else sum(vendor_utt) / len(vendor_utt)
                utt_pairs.append((utt, float(total)))
            units.append((labels, scores, item, flags))
            continue

        label = item.get("label")
        if label and not label.get("word_level_scorable", True):
            continue  # nothing in the target carries the error
        pos = label.get("word_index") if label else None
        # Words the speaker may not have said as written are neither right nor wrong
        # (mark_unverified.py); the planted error stays.
        unverified = set(item.get("unverified", ()))
        keep = [i for i in range(len(target)) if i == pos or i not in unverified]
        labels = [int(i == pos) for i in keep]
        scores = [scores[i] for i in keep]
        flags = None if flags is None else [flags[i] for i in keep]
        units.append((labels, scores, item, flags))
        if pos is not None:
            loc_n += 1
            loc_hit += int(keep[max(range(len(scores)), key=lambda i: scores[i])] == pos)
    return {"units": units, "localisation": (loc_hit, loc_n), "latency": latency,
            "word_pairs": word_pairs, "utt_pairs": utt_pairs}


def _correlations(word_pairs, utt_pairs) -> dict:
    """Agreement with human raters: the standard metric in the MDD literature, so these
    numbers sit next to published work rather than floating on their own."""
    from scipy.stats import pearsonr, spearmanr
    out = {}
    for name, pairs in (("word", word_pairs), ("utterance", utt_pairs)):
        if len(pairs) < 3:
            continue
        a = [p[0] for p in pairs]
        b = [p[1] for p in pairs]
        out[f"pcc_{name}"] = float(pearsonr(a, b).statistic)
        out[f"srcc_{name}"] = float(spearmanr(a, b).statistic)
    return out


def verdict_metrics(units) -> dict:
    """Precision/recall/F1 and false-alarm rate from the vendors' own flags."""
    tp = fp = fn = tn = 0
    by_bucket: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    if any(u[3] is None for u in units):
        return {"recall": math.nan, "precision": math.nan, "f1": math.nan,
                "fa": math.nan, "recall_by_bucket": {}}
    for labels, _, item, flags in units:
        label = item.get("label") or {}
        b = label.get("bucket") or label.get("category") or "human"
        for l, f in zip(labels, flags):
            tp += l and f; fn += l and not f; fp += (not l) and f; tn += (not l) and not f
            if l:
                by_bucket[b][0] += f; by_bucket[b][1] += 1
    rec = tp / (tp + fn) if tp + fn else math.nan
    prec = tp / (tp + fp) if tp + fp else math.nan
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return {"recall": rec, "precision": prec, "f1": f1,
            "fa": fp / (fp + tn) if fp + tn else math.nan,
            "recall_by_bucket": {b: tuple(v) for b, v in by_bucket.items()}}


def flat(units):
    return ([l for u in units for l in u[0]], [s for u in units for s in u[1]])


def pct(x: float) -> str:
    return "  n/a" if math.isnan(x) else f"{x:5.3f}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--responses", type=Path, default=Path("data/responses"))
    ap.add_argument("--json-out", type=Path)
    args = ap.parse_args()

    with open(args.manifest, encoding="utf-8") as f:
        manifest = [json.loads(line) for line in f if line.strip()]

    report = {}
    for path in sorted(args.responses.glob("*.jsonl")):
        vendor = path.stem
        with open(path, encoding="utf-8") as f:
            responses = {}
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    responses[r["id"]] = r
        c = collect(manifest, responses)
        units = c["units"]
        if not units:
            continue
        labels, scores = flat(units)
        pairs = [(u[0], u[1]) for u in units]
        lo, hi = bootstrap_ci(pairs, roc_auc)
        hit, n = c["localisation"]
        lat = sorted(x for x in c["latency"] if not math.isnan(x))
        buckets = defaultdict(lambda: ([], []))
        for ls, ss, item, _ in units:
            b = (item.get("label") or {}).get("bucket") or \
                (item.get("label") or {}).get("category", "control")
            for l, s in zip(ls, ss):
                if l:                     # positives only from this bucket
                    buckets[b][0].append(l)
                    buckets[b][1].append(s)
        for b in buckets:                 # negatives: every correct word in the set
            for l, s in zip(labels, scores):
                if not l:
                    buckets[b][0].append(l)
                    buckets[b][1].append(s)

        report[vendor] = {
            "n_utterances": len(units), "n_words": len(labels), "n_positive": sum(labels),
            "roc_auc": roc_auc(labels, scores), "roc_auc_ci": [lo, hi],
            "pr_auc": average_precision(labels, scores),
            "fpr_at_recall_50": fpr_at_recall(labels, scores, 0.50),
            "fpr_at_recall_80": fpr_at_recall(labels, scores, 0.80),
            "localisation_top1": hit / n if n else math.nan,
            "latency_p50_ms": lat[len(lat) // 2] if lat else math.nan,
            "latency_p95_ms": lat[int(0.95 * len(lat))] if lat else math.nan,
            "by_bucket": {b: roc_auc(*buckets[b]) for b in sorted(buckets)
                          if b != "control"},
        }
        if c["word_pairs"]:
            report[vendor].update(_correlations(c["word_pairs"], c["utt_pairs"]))
        report[vendor]["verdicts"] = verdict_metrics(units)

    print(f"\n{'vendor':<14}{'AUC':>7}{'95% CI':>16}{'PR-AUC':>8}"
          f"{'FA@50':>8}{'FA@80':>8}{'loc@1':>8}{'p50ms':>8}")
    print("-" * 77)
    for v, m in sorted(report.items(), key=lambda kv: -(kv[1]["roc_auc"] or 0)):
        ci = f"[{m['roc_auc_ci'][0]:.3f},{m['roc_auc_ci'][1]:.3f}]"
        print(f"{v:<14}{pct(m['roc_auc']):>7}{ci:>16}{pct(m['pr_auc']):>8}"
              f"{pct(m['fpr_at_recall_50']):>8}{pct(m['fpr_at_recall_80']):>8}"
              f"{pct(m['localisation_top1']):>8}{m['latency_p50_ms']:>8.0f}")

    if any("pcc_word" in m for m in report.values()):
        print(f"\nAgreement with human raters (Tier 2)\n{'vendor':<14}"
              f"{'PCC word':>10}{'SRCC word':>11}{'PCC utt':>10}{'SRCC utt':>11}")
        print("-" * 56)
        for v, m in report.items():
            print(f"{v:<14}{pct(m.get('pcc_word', math.nan)):>10}"
                  f"{pct(m.get('srcc_word', math.nan)):>11}"
                  f"{pct(m.get('pcc_utterance', math.nan)):>10}"
                  f"{pct(m.get('srcc_utterance', math.nan)):>11}")

    print("\nVendor verdicts (each vendor's own error flag; no threshold of ours)\n"
          f"{'vendor':<14}{'recall':>8}{'FA':>8}{'prec':>8}{'F1':>8}   recall by bucket")
    print("-" * 77)
    for v, m in sorted(report.items(),
                       key=lambda kv: -(0 if math.isnan(kv[1]["verdicts"]["f1"])
                                        else kv[1]["verdicts"]["f1"])):
        vd = m["verdicts"]
        rb = "  ".join(f"{b} {h}/{n}" for b, (h, n) in sorted(vd["recall_by_bucket"].items()))
        print(f"{v:<14}{pct(vd['recall']):>8}{pct(vd['fa']):>8}{pct(vd['precision']):>8}"
              f"{pct(vd['f1']):>8}   {rb}")

    all_buckets = sorted({b for m in report.values() for b in m["by_bucket"]})
    if all_buckets:
        print(f"\nAUC by perturbation difficulty\n{'vendor':<14}" +
              "".join(f"{b:>14}" for b in all_buckets))
        print("-" * (14 + 14 * len(all_buckets)))
        for v, m in report.items():
            print(f"{v:<14}" + "".join(pct(m["by_bucket"].get(b, math.nan)).rjust(14)
                                       for b in all_buckets))
    print("\nFA@N = false-alarm rate on correctly-pronounced words at the threshold\n"
          "giving N% recall on perturbed words. Lower is better.", file=sys.stderr)

    if args.json_out:
        args.json_out.write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
