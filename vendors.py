"""One adapter per pronunciation API, all reduced to the same shape.

    score(wav, text, lang) -> {"words": [{"text", "score", "error_type"}], "raw": {...}}

`score` is normalised to 0.0-1.0 (higher = better pronounced), because Lingolix returns
0-1 and everyone else returns 0-100. The word list is the finest granularity every
vendor provides: Lingolix v3's HTTP response stops at the syllable, Azure goes to the
phoneme, so word level is the only fair common denominator. Phoneme-level comparisons
belong in a separate table with coverage stated.

Credentials come from the environment; see README.md. Raw JSON is kept verbatim next to
the reduced shape, so an aggregation can always be rechecked against what the API said.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
from pathlib import Path

import requests

TIMEOUT = 60


class Missing(RuntimeError):
    """Vendor credentials absent; the harness skips this vendor rather than failing."""


def _env(*names: str) -> tuple[str, ...]:
    vals = tuple(os.environ.get(n, "") for n in names)
    if not all(vals):
        raise Missing(f"set {', '.join(names)}")
    return vals


# ── Lingolix v3 ────────────────────────────────────────────────────────────────

API_PREFIX = "/api/pronunciation"


def lingolix(wav: Path, text: str, lang: str) -> dict:
    """Public API: POST /api/pronunciation/v3/check with an API key."""
    key, = _env("LINGOLIX_API_KEY")
    base = os.environ.get("LINGOLIX_BASE_URL", "https://api.lingolix.com")
    return _lingolix_call(f"{base.rstrip('/')}{API_PREFIX}/v3/check",
                          {"Authorization": f"Bearer {key}"}, wav, text, lang)


def _lingolix_call(url: str, headers: dict, wav: Path, text: str, lang: str) -> dict:
    with open(wav, "rb") as f:
        r = requests.post(
            url, headers=headers,
            data={"sentence": text, "language_code": lang},
            files={"speechdata": (wav.name, f, "audio/wav")},
            timeout=TIMEOUT,
        )
    r.raise_for_status()
    raw = r.json()
    return {
        "words": [{"text": w["text"], "score": w["accuracy"],
                   "scored": w.get("is_scored", True),
                   "error_type": _lingolix_error(w)}
                  for w in raw.get("words", [])],
        "raw": raw,
    }


def _lingolix_error(w: dict) -> str | None:
    sylls = w.get("syllables") or []
    if sylls and all(s.get("is_missing") for s in sylls):
        return "Omission"
    if any(s.get("is_extra") for s in sylls):
        return "Insertion"
    return None


# ── Microsoft Azure AI Speech ──────────────────────────────────────────────────

AZURE_LOCALE = {"en": "en-US", "de": "de-DE", "ja": "ja-JP", "fr": "fr-FR", "ko": "ko-KR",
                "es": "es-ES"}


def azure(wav: Path, text: str, lang: str) -> dict:
    """Documented best configuration: phoneme granularity, IPA alphabet, miscue on.

    Getting these wrong understates Azure and invalidates the comparison, so they are
    hard-coded rather than left to a config file.
    """
    key, region = _env("AZURE_SPEECH_KEY", "AZURE_SPEECH_REGION")
    params = base64.b64encode(json.dumps({
        "ReferenceText": text,
        "GradingSystem": "HundredMark",
        "Granularity": "Phoneme",
        "Dimension": "Comprehensive",
        "PhonemeAlphabet": "IPA",
        "EnableMiscue": True,
    }).encode()).decode()
    r = requests.post(
        f"https://{region}.stt.speech.microsoft.com"
        "/speech/recognition/conversation/cognitiveservices/v1",
        params={"language": AZURE_LOCALE.get(lang, lang), "format": "detailed"},
        headers={"Ocp-Apim-Subscription-Key": key,
                 "Content-Type": "audio/wav; codecs=audio/pcm; samplerate=16000",
                 "Pronunciation-Assessment": params},
        data=wav.read_bytes(), timeout=TIMEOUT,
    )
    r.raise_for_status()
    raw = r.json()
    nbest = (raw.get("NBest") or [{}])[0]
    return {
        "words": [{"text": w["Word"],
                   "score": _azure_word_score(w),
                   "scored": True,
                   "error_type": _azure_field(w, "ErrorType")}
                  for w in nbest.get("Words", [])],
        "raw": raw,
    }


def _azure_field(w: dict, name: str):
    """Tolerate both wire shapes: REST puts these on the word, the SDK nests them."""
    if name in w:
        return w[name]
    return (w.get("PronunciationAssessment") or {}).get(name)


def _azure_word_score(w: dict) -> float:
    score = _azure_field(w, "AccuracyScore")
    if score is None:
        # Omitted words carry no accuracy score at all; that is a maximal error,
        # not a missing measurement.
        return 0.0
    return float(score) / 100


# ── TonePerfect ────────────────────────────────────────────────────────────────

TONEPERFECT_LOCALE = {"en": "en-US", "fr": "fr-FR", "es": "es-419", "de": "de-DE",
                      "ko": "ko-KR", "ja": "ja-JP", "zh": "zh-CN"}


def toneperfect(wav: Path, text: str, lang: str) -> dict:
    if lang not in TONEPERFECT_LOCALE:
        raise Missing(f"TonePerfect does not support {lang}")
    key, = _env("TONEPERFECT_API_KEY")
    with open(wav, "rb") as f:
        r = requests.post(
            "https://api.toneperfect.app/v1/assess",
            # same clip + text -> same key, so a retried call is not charged twice
            headers={"Authorization": f"Bearer {key}",
                     "Idempotency-Key": "pronbench-" + hashlib.sha1(
                         f"{wav.name}|{text}|{lang}".encode()).hexdigest()[:24]},
            data={"text": text, "language": TONEPERFECT_LOCALE[lang]},
            files={"audio": (wav.name, f, "audio/wav")},
            timeout=TIMEOUT,
        )
    if r.status_code >= 400:  # keep the reason; the body says why (422: e.g. text it cannot parse)
        raise RuntimeError(f"toneperfect {r.status_code}: {r.text[:300]}")
    raw = r.json()
    if raw.get("test"):
        raise RuntimeError("toneperfect: test key, fixed fixture response (use a live key)")
    return {"words": toneperfect_words(raw), "raw": raw}


def toneperfect_words(raw: dict) -> list[dict]:
    """A word holding a phonetic symbol TonePerfect does not support comes back with
    score null and decision "partial" (listed in phonetic_coverage.unassessed_words).
    That is a coverage gap, not a verdict: take the score its scorer computed anyway
    (raw.raw.words, same order) and do not count "partial" as an error call."""
    inner = (raw.get("raw") or {}).get("words") or []
    words = raw.get("words") or []
    out = []
    for k, w in enumerate(words):
        score = w.get("score")
        if score is None and len(inner) == len(words):
            score = inner[k].get("score")
        out.append({"text": w["word"], "score": (score or 0) / 100, "scored": score is not None,
                    "error_type": None if w.get("decision") in ("no_correction", "partial")
                    else "Mispronunciation"})
    return out


# ── Open baseline: ASR transcribe-and-diff ─────────────────────────────────────

_WHISPER = {}


def whisper_diff(wav: Path, text: str, lang: str) -> dict:
    """"Why not just use ASR?" baseline.

    Expect this to do *well* on Tier 1 and to collapse on Tier 2: "does the transcript
    match the text" is exactly the synthetic task, but real L2 errors frequently leave
    word identity intact. That contrast is the main argument for running both tiers.
    """
    from faster_whisper import WhisperModel
    from rapidfuzz.distance import Levenshtein
    size = os.environ.get("WHISPER_MODEL", "small")
    if size not in _WHISPER:
        _WHISPER[size] = WhisperModel(size, device="auto", compute_type="int8")
    segments, _ = _WHISPER[size].transcribe(str(wav), language=lang, beam_size=5)
    heard = [w.strip(".,!?;:\"'").lower()
             for s in segments for w in s.text.split() if w.strip(".,!?;:\"'")]

    target = text.split()
    keys = [t.strip(".,!?;:\"'").lower() for t in target]
    scores = [0.0] * len(target)
    # Align heard words to target words, then score each target word by how closely its
    # aligned counterpart matches. Unaligned target words score 0 (apparent omission).
    for op in Levenshtein.opcodes(keys, heard):
        if op.tag == "equal":
            for i in range(op.src_start, op.src_end):
                scores[i] = 1.0
        elif op.tag == "replace":
            for k, i in enumerate(range(op.src_start, op.src_end)):
                j = op.dest_start + k
                if j < op.dest_end:
                    scores[i] = Levenshtein.normalized_similarity(keys[i], heard[j])
    return {
        "words": [{"text": t, "score": s, "scored": True, "error_type": None}
                  for t, s in zip(target, scores)],
        "raw": {"transcript": " ".join(heard)},
    }


VENDORS = {
    "lingolix": lingolix,
    "azure": azure,
    "toneperfect": toneperfect,
    "whisper_diff": whisper_diff,
}
