#!/usr/bin/env python3
"""Run a manifest against the vendor APIs, caching every raw response.

Resumable and append-only: these calls cost money, so an interrupted run must never
repeat work, and the raw JSON is kept verbatim so every number can be rechecked.

Usage:
    python run.py --manifest data/tier1/manifest_en.jsonl --vendors lingolix azure
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from vendors import VENDORS, Missing  # noqa: E402


def done_ids(path: Path) -> set[str]:
    """IDs already answered *successfully*.

    A failed record stays on disk as evidence but is not treated as done, so a transient
    503 gets retried on the next run instead of being baked in permanently. Retries
    append a second record with the same id; readers take the last one, so the success
    wins without anything being rewritten in place.
    """
    if not path.exists():
        return set()
    ok, failed = set(), set()
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            (failed if r.get("error") else ok).add(r["id"])
    if failed - ok:
        print(f"  {len(failed - ok)} previously failed, will retry", file=sys.stderr)
    return ok


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--vendors", nargs="+", default=list(VENDORS))
    ap.add_argument("--out", type=Path, default=Path("data/responses"))
    ap.add_argument("--limit", type=int)
    args = ap.parse_args()

    root = args.manifest.parent
    with open(args.manifest, encoding="utf-8") as f:
        items = [json.loads(line) for line in f if line.strip()]
    if args.limit:
        items = items[: args.limit]
    args.out.mkdir(parents=True, exist_ok=True)

    for name in args.vendors:
        fn = VENDORS[name]
        path = args.out / f"{name}.jsonl"
        skip = done_ids(path)
        todo = [it for it in items if it["id"] not in skip]
        print(f"\n{name}: {len(todo)} to do, {len(skip)} cached", file=sys.stderr)
        with open(path, "a", encoding="utf-8") as out:
            for i, it in enumerate(todo, 1):
                rec = {"id": it["id"], "vendor": name}
                t0 = time.perf_counter()
                try:
                    lang = it.get("vendor_lang", {}).get(name, it["lang"])
                    rec.update(fn(root / it["audio"], it["target"], lang))
                except Missing as e:
                    print(f"  skipping {name}: {e}", file=sys.stderr)
                    break
                except Exception as e:  # keep going; a dead utterance is not a dead run
                    rec["error"] = f"{type(e).__name__}: {e}"
                rec["latency_ms"] = round((time.perf_counter() - t0) * 1000, 1)
                out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                out.flush()
                if i % 20 == 0:
                    print(f"  {i}/{len(todo)}", file=sys.stderr)


if __name__ == "__main__":
    main()
