#!/usr/bin/env python
"""Run an OpenRouter-hosted LLM on the EvaLatin 2024 parsing task.

Two modes:

* **Full bench** (default): predicts every sentence of each split, writes
  `predictions/<slug>/{split}_pred.conllu` + `scores.json`, prints UAS/LAS/CLAS.
  This is the real run you commit.

      .venv/bin/python scripts/run_openrouter.py
      .venv/bin/python scripts/run_openrouter.py --splits prose --k-shot 2

* **Smoke test** (`--limit N`): a cheap end-to-end check on the first N
  sentences of one split. Fires one live diagnostic call (so API/auth/schema
  errors surface verbatim), then predicts + scores the N-sentence slice without
  touching the committed `predictions/` tree. A few cents at most.

      .venv/bin/python scripts/run_openrouter.py --split prose --limit 3

The API key comes from `OPENROUTER_API_KEY`; this script also loads a gitignored
`.env` at the repo root if present, so you don't have to export it each time.
"""
from __future__ import annotations

import argparse
import os
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))


def _load_dotenv(path: Path) -> None:
    """Minimal .env loader (KEY=VALUE per line) — no dependency, no override of
    variables already set in the environment."""
    if not path.exists():
        return
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        if key.startswith("export "):   # tolerate `export KEY=VALUE`
            key = key[len("export "):].strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


def _slice_conllu(src: Path, dst: Path, n: int) -> int:
    """Write the first `n` sentences (blank-line-separated blocks) of `src` to
    `dst`. Returns how many were actually written."""
    blocks = [b for b in src.read_text().split("\n\n") if b.strip()]
    take = blocks[:n]
    dst.write_text("\n\n".join(take) + "\n\n")
    return len(take)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="google/gemini-3-flash-preview")
    ap.add_argument("--splits", nargs="+", default=["poetry", "prose"],
                    choices=["poetry", "prose"])
    ap.add_argument("--k-shot", type=int, default=0)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--temperature", type=float, default=0.3)
    ap.add_argument("--max-tokens", type=int, default=None,
                    help="output token cap (raise for pretty-printing models like gemini-2.5-flash)")
    ap.add_argument("--no-temperature", action="store_true",
                    help="omit temperature (required for reasoning models e.g. gpt-5-mini)")
    ap.add_argument("--reasoning-effort", default=None,
                    choices=["low", "medium", "high"],
                    help="reasoning effort for reasoning models (bounds hidden-reasoning token spend)")
    ap.add_argument("--force", action="store_true",
                    help="ignore cached predictions/scores and re-run")
    ap.add_argument("--limit", type=int, default=None,
                    help="smoke mode: only the first N sentences of one split")
    ap.add_argument("--split", default="prose", choices=["poetry", "prose"],
                    help="which split to use in smoke mode (--limit)")
    args = ap.parse_args()

    _load_dotenv(REPO_ROOT / ".env")

    from latinbench import Bench
    from latinbench.data import gold_path, test_path
    from latinbench.models.openrouter_llm import OpenRouterModel
    from latinbench.score import score

    if not os.environ.get("OPENROUTER_API_KEY"):
        print("ERROR: OPENROUTER_API_KEY not set (env or .env). Aborting.",
              file=sys.stderr)
        return 2

    extra = {}
    if args.max_tokens is not None:
        extra["max_tokens"] = args.max_tokens
    model = OpenRouterModel(
        model_id=args.model,
        num_workers=args.workers,
        temperature=args.temperature,
        k_shot=args.k_shot,
        send_temperature=not args.no_temperature,
        reasoning_effort=args.reasoning_effort,
        **extra,
    )
    print(f"Model: {args.model}  ->  cache slug '{model.name}'")

    if args.limit is None:
        scores = Bench(splits=tuple(args.splits)).run(model, force=args.force)
        print("\n=== scores ===")
        for split, by_metric in scores.items():
            for metric in ("UAS", "LAS", "CLAS"):
                if metric in by_metric:
                    print(f"  {split:6s} {metric:4s}  F1={by_metric[metric]['F1']:.2f}")
        return 0

    # --- smoke mode -------------------------------------------------------
    split = args.split
    print(f"\nSMOKE: first {args.limit} sentence(s) of '{split}' "
          f"with {model.name}\n")

    # 1) one live diagnostic call so auth/schema errors are visible verbatim.
    import conllu
    first = conllu.parse(test_path(split).read_text())[0]
    single = [t for t in first if isinstance(t["id"], int)]
    print(f"[diagnostic] calling OpenRouter on 1 sentence ({len(single)} tokens)…")
    try:
        resp = model._call_llm(single)
        n = len(resp.get("tokens", []))
        print(f"[diagnostic] OK — got {n} token predictions back "
              f"(expected {len(single)}).")
        print(f"[diagnostic] sample: {resp.get('tokens', [])[:3]}")
    except Exception as e:  # noqa: BLE001 — diagnostic: show whatever broke
        print(f"[diagnostic] FAILED: {type(e).__name__}: {e}")
        return 1

    # 2) predict + score the N-sentence slice in a temp dir (predictions/ untouched).
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        test_slice = tmp / "test.conllu"
        gold_slice = tmp / "gold.conllu"
        pred_slice = tmp / "pred.conllu"
        nt = _slice_conllu(test_path(split), test_slice, args.limit)
        _slice_conllu(gold_path(split), gold_slice, args.limit)
        print(f"\n[predict] running model on {nt} sentence(s)…")
        model.predict(test_slice, pred_slice)
        print("\n[score] scoring against gold slice…")
        s = score(gold_slice, pred_slice)
        print("\n=== smoke scores (tiny sample — indicative only) ===")
        for metric in ("UAS", "LAS", "CLAS"):
            if metric in s:
                print(f"  {metric:4s}  F1={s[metric]['F1']:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
