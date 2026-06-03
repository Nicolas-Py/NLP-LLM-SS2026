# Design: few-shot demonstrations from training data (UD_Latin-Perseus)

**Date:** 2026-06-03
**Status:** Implemented; bench run pending (LM Studio). Numbers land in
`docs/01_findings.md` §"Key findings" #8 once run.
**Related:** extends `2026-05-27-few-shot-llm-benchmark-design.md` (the k-shot
machinery this reuses) and `docs/01_findings.md` #7 (the hand-curated 2-shot
result this is the follow-up to).

---

## Goal

Run the few-shot LLM parser again, but with the in-context demonstrations drawn
from **real training data** (UD_Latin-Perseus) instead of the six hand-curated
toy sentences. Everything else — k=2, the prompt, the JSON schema, the
`Bench.compare` reporting path — stays the same, so the only thing that changes
between the existing `…-2shot` run and this one is **where the examples come
from**.

## Approach — keep it simple

The injection machinery already exists: `ExamplePool(path=...)` reads any
CoNLL-U file and `LMStudioModel(k_shot=2, example_pool=...)` injects those
sentences as chat history. So this is just:

1. **A new committed pool file** — `src/latinbench/few_shot_examples_perseus.conllu`,
   **6 real Perseus train sentences** (so it's a drop-in sibling of the
   6-sentence hand-curated pool). Each sentence keeps its Perseus
   `# source_sent_id` for provenance.

2. **Punctuation stripped + re-indexed.** EvaLatin's test/gold data is
   punctuation-free (0 `punct`/`PUNCT` tokens), and the JSON-schema label enum
   therefore has no `punct`. Real Perseus sentences all carry punctuation, so
   each picked sentence has its punctuation tokens removed and the remaining
   tokens renumbered `1..n` (heads remapped). This makes the demonstrations
   format-identical to what the model sees for every target — not a hack, the
   same normalization EvaLatin applied to its own data.

3. **One tiny code change for cache naming.** `ExamplePool` gains a `tag`
   (derived from the filename: the default pool → `""`, `…_perseus.conllu` →
   `"perseus"`), and `LMStudioModel` appends it to its slug. So the Perseus run
   caches to `predictions/…-2shot-perseus/` and sits **beside** the hand-curated
   `…-2shot` run instead of overwriting it.

The picked sentences are short (5–12 tokens), form valid single-root trees, and
use only deprels in the EvaLatin gold enum — selected once from Perseus r2.13
train and committed. (The raw treebank is downloaded to a gitignored
`data/ud_treebanks/` and is not committed.)

## Files

```
src/latinbench/few_shot_examples_perseus.conllu   # new: 6 Perseus demos (punct-stripped)
src/latinbench/few_shot.py                         # tweak: ExamplePool gains `tag`
src/latinbench/models/lmstudio_llm.py              # tweak: slug appends pool tag
notebooks/02_compare_models.ipynb                  # one cell: run the Perseus variants
notebooks/03_explore_past_runs.ipynb               # parse_slug aware of the -perseus tag
```

No change to the prompt, schema, scoring, or the predict/repair path.

## Cache & naming

| construction | slug |
|---|---|
| `LMStudioModel("qwen3-0.6b-mlx", k_shot=2)` (default pool) | `qwen3-0.6b-mlx-2shot` *(unchanged)* |
| `LMStudioModel("qwen3-0.6b-mlx", k_shot=2, example_pool=ExamplePool(PERSEUS))` | `qwen3-0.6b-mlx-2shot-perseus` |
| `LMStudioModel("qwen3-vl-8b-instruct-mlx", k_shot=2, example_pool=ExamplePool(PERSEUS))` | `qwen3-vl-8b-instruct-mlx-2shot-perseus` |

## Scope

`qwen3-0.6b-mlx` and `qwen3-vl-8b-instruct-mlx`, k=2 — an apples-to-apples
comparison vs the hand-curated `…-2shot` numbers in findings #7.

## Testing (network-free)

Mirrors the hand-curated pool's checks: the Perseus pool loads, every token has
an integer head + non-empty deprel and no MWT rows, and every deprel is in the
EvaLatin gold enum. Plus slug naming: `…-2shot-perseus` for the Perseus pool,
unchanged `…-2shot` for the default pool.

## Note

Perseus FEATS are barer than EvaLatin's (no `InflClass`/`NameType`), so the
FEATS column of a demo differs in granularity from a target's. The prompt
*structure* (columns, formatter, schema) is identical; this is inherent to using
real training data and doesn't affect the k=0-vs-k=2 comparison.
