# Pronunciation Benchmark

An open benchmark for **pronunciation assessment APIs**: does the API notice a
mispronounced word, and does its score agree with human judges?

- **6 languages:** English, French, Spanish, German, Japanese, Korean
- **1,700 recordings:** 1,200 native clips with planted errors + 500 rated learner clips
- **Adapters for:** Lingolix, Microsoft Azure Pronunciation Assessment, TonePerfect, and
  an open "transcribe with Whisper and diff" baseline

Results and discussion: **[Best pronunciation assessment API in 2026? An open
benchmark](https://lingolix.com/en/blog/best-pronunciation-assessment-api-2026)**.

## The two tests

**Test 1 — native speech, planted errors (6 languages, 200 clips each).**
Common Voice recordings whose transcript was checked with Whisper. We then edit the
*reference text* so it no longer matches the audio by one word: a swapped sound (graded
subtle / moderate / gross by articulatory features), an added or dropped sound, a dropped
word, and in Japanese a vowel or consonant length change (おばさん / おばあさん). The
replacement is always a real word, so the text stays valid. A good system flags the
edited word, and only that word.

**Test 2 — real learners, human judges (English, 500 clips).**
The test split of [speechocean762](https://www.openslr.org/101/): Mandarin speakers
reading English sentences, each word and sentence scored by five trained raters. A word
with human accuracy ≤ 6/10 counts as mispronounced.

Test 1 measures confident mismatches in every language; Test 2 measures agreement with
people on real learner speech, in one language. Read them together.

## Layout

```
data/tier1/manifest_<lang>.jsonl     test 1: target text, original prompt, label, Common Voice clip id
data/tier1/audio/                    test 1 audio (download, see below)
data/tier2/manifest_speechocean762.jsonl   test 2: target text and human scores
data/tier2/audio/                    test 2 audio (built by build_tier2.py)
vendors.py     one adapter per API, each reduced to the same word-level shape
run.py         calls an API for every clip; resumable, keeps raw responses
report.py      all metrics from the raw responses
build_tier2.py downloads speechocean762 and builds test 2
```

A test 1 manifest line:

```json
{"id": "en-0001", "lang": "en", "audio": "audio/en-0001.wav",
 "prompt": "For the past few years, his focus has been fiction.",
 "target": "For the past few tears, his focus has been fiction.",
 "label": {"category": "sub", "bucket": "sub_gross", "word_index": 4,
           "original": "years", "replacement": "tears", ...}}
```

`prompt` is what was said, `target` is the reference text sent to the API, and
`label.word_index` is the word that no longer matches.

## Reproduce

Requires Python 3.10+, [uv](https://docs.astral.sh/uv/) and ffmpeg.

```bash
git clone https://github.com/lingolix/pronunciation-benchmark && cd pronunciation-benchmark

# Test 1 audio (138 MB, from the release)
curl -LO https://github.com/lingolix/pronunciation-benchmark/releases/download/v2026.10/pronbench-tier1-audio-v2026.10.zip
unzip pronbench-tier1-audio-v2026.10.zip -d data/tier1

# Test 2 audio (downloads ~520 MB from OpenSLR, converts 500 clips)
uv run build_tier2.py --out data/tier2 --n 500
```

Set the keys for the APIs you want to run:

| API | Environment |
| --- | --- |
| Lingolix | `LINGOLIX_API_KEY` |
| Microsoft Azure | `AZURE_SPEECH_KEY`, `AZURE_SPEECH_REGION` |
| TonePerfect | `TONEPERFECT_API_KEY` (a live key; test keys return a fixed sample) |
| Whisper baseline | none; `WHISPER_MODEL` (default `small`), needs `uv run --with faster-whisper` |

Run and report, per test:

```bash
uv run run.py --manifest data/tier1/manifest_en.jsonl --vendors lingolix azure toneperfect \
              --out data/responses/tier1_en
uv run report.py --manifest data/tier1/manifest_en.jsonl --responses data/responses/tier1_en

uv run run.py --manifest data/tier2/manifest_speechocean762.jsonl --vendors lingolix azure toneperfect \
              --out data/responses/tier2
uv run report.py --manifest data/tier2/manifest_speechocean762.jsonl --responses data/responses/tier2
```

`run.py` appends to `<out>/<vendor>.jsonl` and skips clips that already succeeded, so an
interrupted run resumes where it stopped and failed calls are retried on the next run.
A full run is 1,700 calls per API (TonePerfect: 1 credit per call).

### What the report shows

| Metric | Meaning |
| --- | --- |
| `AUC` | How well the word score separates mispronounced from correct words (0.5 = chance), with a 95% bootstrap interval over clips |
| `FA@80` | Share of correct words flagged when the threshold catches 80% of errors. The number a learner feels |
| `loc@1` | Test 1: the edited word is the lowest-scoring word in its sentence |
| `PCC word / utt` | Test 2: Pearson correlation with human word accuracy / sentence total |
| verdicts | Recall and false alarms of each API's own error flag, no threshold of ours |

Word level is the finest granularity every API returns. Japanese has no spaces, so each
API's words are mapped onto the manifest's `units` by character position.

## Configuration notes

- **Azure:** `Granularity=Phoneme`, `PhonemeAlphabet=IPA`, `EnableMiscue=true`,
  `GradingSystem=HundredMark` — its documented best setup. Spanish uses `es-ES`.
- **TonePerfect:** a word containing a phonetic symbol it does not support comes back
  with `score: null` and decision `partial`. That is a coverage gap, not a verdict: the
  adapter uses the score TonePerfect computed anyway and does not count `partial` as an
  error. Five Japanese sentences containing 々 are rejected with HTTP 422.
- **Lingolix:** the public `/api/pronunciation/v3/check` endpoint with an API key.

## Limitations

- Test 1 errors are planted in the text, not made by learners: a confident mismatch,
  not a learner's fuzzy vowel.
- Test 2 is one language and one group of learners.
- Systems may have seen Common Voice or speechocean762 in training. Test 1 sentences are
  held out from Lingolix training.

## Licenses

- **Code:** MIT, see [LICENSE](LICENSE).
- **Test 1 audio and text:** from [Mozilla Common Voice](https://commonvoice.mozilla.org),
  CC0. Do not attempt to identify the speakers.
- **Test 2:** [speechocean762](https://www.openslr.org/101/), CC BY 4.0. Please cite:
  Zhang et al., *speechocean762: An Open-Source Non-native English Speech Corpus for
  Pronunciation Assessment*, Interspeech 2021.

## Cite

```bibtex
@misc{lingolix_pronunciation_benchmark_2026,
  title  = {An open benchmark for pronunciation assessment APIs},
  author = {{Lingolix}},
  year   = {2026},
  url    = {https://github.com/lingolix/pronunciation-benchmark}
}
```

Issues and pull requests welcome, especially human-rated learner data for languages
other than English.
