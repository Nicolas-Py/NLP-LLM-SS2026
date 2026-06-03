# Findings — LLM dependency parsing on EvaLatin 2024

Living research log. Add new sections as experiments land; keep older entries
intact so we can trace what we tried and what it cost. Detailed per-experiment
analysis lives in [notebooks/02_compare_models.ipynb](../notebooks/02_compare_models.ipynb);
this doc is the distilled version.

_Last updated: 2026-06-03._

## Goal

Beat the EvaLatin 2024 winner (LatinPipe, 7-model ensemble) on dependency
parsing of Latin: **~78 LAS poetry, ~83 LAS prose**.

The 1× LatinPipe checkpoint we ship as a reference is a step down from the
ensemble; both are listed below as targets.

## Reference scores (committed parsers)

Gold tokenization, so token/sentence/word F1 = 100 for every system. We track
**LAS** (labeled attachment) and **CLAS** (content-word labeled attachment) as
the primary metrics, with **UAS** as a structure-only sanity check.

| split | system | UAS | LAS | CLAS |
|---|---|---:|---:|---:|
| poetry | LatinPipe (1× ckpt) | 78.4 | **72.3** | 71.3 |
| poetry | UDPipe 2 (`latin-perseus-ud-2.17`) | 69.0 | 61.2 | 59.9 |
| prose  | LatinPipe (1× ckpt) | 80.4 | **75.1** | 70.9 |
| prose  | UDPipe 2 (`latin-perseus-ud-2.17`) | 69.3 | 62.4 | 57.5 |

Ensemble published numbers (not run locally): ~78 LAS poetry, ~83 LAS prose.

## LLM-as-parser approach

Each sentence goes to a local LLM hosted by **LM Studio** (MLX runtime on
Apple Silicon). The prompt presents the tokens in a tab-separated table; the
response is constrained to a JSON schema that fixes the label set to the UD
relations seen in gold training data and the head field to an integer in
`[0, n]`. One LM Studio server hosts all models; predictions cache to
`predictions/<slug>/`.

When the per-token JSON doesn't form a valid rooted tree (cycles, multi-root),
we repair with the minimum number of head mutations rather than wiping the
sentence — see `_repair_tree` in [src/latinbench/models/lmstudio_llm.py](../src/latinbench/models/lmstudio_llm.py).

## Results so far

All LLM runs below use the same prompt, schema, and minimal tree-repair
fallback. Fallback rate is the fraction of tokens whose head we had to
mutate to make the sentence form a valid tree.

| split | system | UAS | LAS | CLAS | fallback |
|---|---|---:|---:|---:|---|
| poetry | LatinPipe | 78.4 | **72.3** | 71.3 | — |
| poetry | UDPipe 2 | 69.0 | 61.2 | 59.9 | — |
| poetry | qwen3-vl-8b-instruct-mlx          | 35.4 | **18.2** | 17.4 | 4.2% tok, 29% of sents |
| poetry | qwen3-vl-8b-instruct-mlx (2-shot) | 35.9 | 18.4 | 17.4 | 9.5% tok, 32% of sents |
| poetry | qwen3-0.6b-mlx                    |  9.6 | 2.7 | 2.7 | 42.3% tok, 44% of sents |
| poetry | qwen3-0.6b-mlx (2-shot)           | 19.5 | 2.1 | 2.2 | 57.9% tok, 74% of sents |
| poetry | qwen3-0.6b-mlx (2-shot Perseus)   | 11.6 | 2.3 | 2.4 | 23.2% tok, 72% of sents |
| prose  | LatinPipe | 80.4 | **75.1** | 70.9 | — |
| prose  | UDPipe 2 | 69.3 | 62.4 | 57.5 | — |
| prose  | qwen3-vl-8b-instruct-mlx          | 33.3 | 17.8 | 14.5 | 3.4% tok, 41% of sents |
| prose  | qwen3-vl-8b-instruct-mlx (2-shot) | 36.5 | **20.2** | 16.7 | 3.8% tok, 38% of sents |
| prose  | qwen3-0.6b-mlx                    |  7.6 | 1.6 | 1.5 | 40.1% tok, 53% of sents |
| prose  | qwen3-0.6b-mlx (2-shot)           | 23.5 | 1.3 | 1.5 | 59.7% tok, 86% of sents |
| prose  | qwen3-0.6b-mlx (2-shot Perseus)   | 12.8 | 2.0 | 2.1 | 20.8% tok, 63% of sents |

## Key findings

1. **Scale dominates.** Going from 0.6B → 8B (same Qwen3 family) lifts LAS
   from ~2 to ~18 on both splits — roughly **7–11×**, depending on split.
   UAS jumps even more sharply (10 → 35 poetry, 8 → 33 prose) because the
   8B produces a valid tree on 96% of tokens vs the 0.6B's 58%. Same
   prompt, same schema, same tree-repair logic; only the model id changes.

2. **The 0.6B fails tree validity on roughly half its sentences.** After
   minimal repair, 44–53% of sentences still required at least one head
   mutation; 40–42% of all tokens needed re-pointing. The 8B drops that
   to 3–4% of tokens (29–41% of sentences see *any* repair, but most of
   those only need one or two head mutations).

3. **8B is still ~4× short of the LatinPipe baseline** (LAS 18 vs 72 on
   poetry). Closing that gap won't come from prompt tweaks alone; see
   "Next experiments" below.

4. **Tree-repair strategy is a UAS/LAS tradeoff for weak models.** When
   we switched from "wipe broken sentences to right-branching defaults"
   to "minimal head mutation", 0.6B LAS went *up* (1.9 → 2.7 poetry,
   1.1 → 1.6 prose) but UAS went *down* (17 → 10 poetry, 21 → 8 prose).
   Right-branching was a decent structural prior that helped UAS at the
   cost of all labels reading `dep`; minimal repair preserves model
   labels (LAS) but lets the model's bad heads stand (UAS). For the 8B
   the choice barely moves either metric — its trees are valid most of
   the time. **Implication:** for low-skill LLMs, the apparent UAS
   number is more about the fallback than the model.

5. **At 0.6B the model produces nicely-formatted nonsense.** Detailed
   error analysis (notebook cells 17–18) shows label collapse onto a
   handful of English-flavored relations (`det`, `nsubj`), 0% accuracy
   on many real Latin relations (`amod`, `conj`, `case`, `nmod`, `cop`,
   `acl`, `mark`), and ~80% of tokens with *both* head and label wrong.
   Constrained decoding fixes formatting, not knowledge. (Numbers in
   the notebook reflect the pre-repair run; the qualitative picture is
   unchanged after re-grading.)

6. **No structural pocket of competence at 0.6B.** Per-relation head
   accuracy is 5–25% across the board; no relation type is "easy". For
   comparison LatinPipe sits at 60–90% per relation.

7. **Few-shot (2-shot) helps the 8B on prose (+2.4 LAS) but not poetry;
   hurts the 0.6B on both splits.** Two hand-curated Latin sentences with
   full UD annotations injected as chat history (user/assistant turns
   before the target). The 8B's prose LAS climbed 17.80 → 20.16 and CLAS
   14.53 → 16.68; poetry barely moved (LAS +0.20). The 0.6B's LAS dropped
   on both splits (2.67 → 2.06 poetry; 1.62 → 1.32 prose) while UAS jumped
   sharply (9.6 → 19.5, 7.6 → 23.5) — same UAS/LAS tradeoff documented in
   key finding #4: more invalid output → more minimal-repair → right-
   branching prior boosts UAS while erasing model labels. **The 8B prose
   result restores the prose > poetry gap** (20.2 vs 18.4) that
   trained parsers show but 0-shot 8B didn't, partly answering the open
   question below — suggests 0-shot 8B wasn't really doing Latin syntax;
   2 demonstrations are enough to pull it into the data. See the
   [few-shot design spec](superpowers/specs/2026-05-27-few-shot-llm-benchmark-design.md)
   for methodology (disjoint hand-curated pool, static selection,
   deterministic seed, identical prompt scaffolding across k).

8. **Few-shot from training data (Perseus) beats the hand-curated pool on the
   0.6B; 8B not yet run.** Same k=2 setup as #7, but the two demonstrations are
   real UD_Latin-Perseus training sentences (punctuation-stripped) instead of
   hand-curated toy ones (run 2026-06-03; see the 0.6B `(2-shot Perseus)` rows in
   the results table above). Perseus beats hand-curated on LAS and CLAS on both
   splits (poetry LAS 2.31 vs 2.06, prose 1.99 vs 1.32), and unlike the
   hand-curated pool it **helps prose** over zero-shot (1.99 vs 1.62). Its bigger
   win is tree validity: token-fallback drops to 23.2% / 20.8% (poetry/prose) vs
   the hand-curated pool's 57.9% / 59.7% — real treebank demos make even the weak
   model emit far more valid trees. (UAS is correspondingly lower, 11.6 / 12.8 vs
   19.5 / 23.5 — the fallback-driven UAS/LAS tradeoff of #4.) Absolute LAS is
   still ~2; the **8B — the model that actually benefits from few-shot (#7) — is
   not run yet** (only the 0.6B was loaded). Pool:
   `src/latinbench/few_shot_examples_perseus.conllu` (6 sentences); cache
   `predictions/<model>-2shot-perseus/`. Methodology in the
   [training-data pool spec](superpowers/specs/2026-06-03-few-shot-training-data-pool-design.md).

## Engineering wins

- **Minimal tree repair** (commit `9c230c9`) preserves the model's
  per-token labels when only the head pointers are inconsistent. Old
  behavior wiped the whole sentence to right-branching defaults; new
  behavior re-points the smallest set of heads to make the tree legal.
  Net effect on 0.6B: +0.7 LAS poetry, +0.5 LAS prose; UAS drops (see
  key finding #4). Net effect on 8B: negligible — its trees are mostly
  valid anyway.
- **Per-sentence partial cache** (pre-existing). A `.partial.json`
  sidecar updates after each sentence so a crash mid-bench resumes
  exactly where it left off — important since 8B runs take hours.
  Caveat: the partial only stores *post-repair* head/deprel, so changing
  the repair logic invalidates cached runs (which is why we re-ran 0.6B
  from scratch on 2026-05-18).

## Next experiments

Ordered by expected impact ÷ effort:

1. **Gemma-3-12B (cross-family check).** Registry entry already exists
   (`gemma-3-12b-lmstudio`); the notebook has a stub cell. Tells us
   whether the 8B result is Qwen-specific or a general scale effect.
2. ~~**Few-shot Latin parses in the prompt.**~~ Done 2026-05-27 (key
   finding #7). 2-shot helps the 8B on prose, flat on poetry, hurts the
   0.6B. Worth following up: scaling k (4, 8) and multi-seed variance
   for the 8B — the +2.4 LAS prose gain is the kind of effect size that
   could vary meaningfully with example choice. Training-data (Perseus)
   demonstrations are now implemented (finding #8); run pending.
3. **Chu-Liu-Edmonds over candidate heads.** `ufal.chu-liu-edmonds` is
   already installed. Have the LLM score each token-pair candidacy
   instead of committing to one head, then extract the maximum-spanning
   tree. Turns N independent decisions into one global decision and
   removes the need for `_repair_tree` entirely.
4. **Larger Qwen (32B / 72B), if it fits on the box.** Test whether
   the scale curve flattens out by 12B or keeps climbing.

## Open questions

- **Why is poetry so close to prose for the 8B?** LatinPipe and UDPipe
  both show a clear poetry < prose gap (~3 LAS); the 8B shows essentially
  no gap (18.2 vs 17.8). Suggests it's not actually doing Latin syntax —
  more like applying a generic dependency prior that's split-agnostic.
  *Partly answered by finding #7:* with 2-shot the 8B's prose > poetry
  gap re-emerges (20.2 vs 18.4), matching the trained-parser pattern —
  consistent with "0-shot 8B applies a generic prior; 2-shot pulls it
  into the actual treebank."
- **CLAS gap on prose.** The 8B prose UAS/LAS look comparable to poetry,
  but CLAS drops 3 points (17.4 → 14.5). Worth checking which content-word
  relations specifically degrade.
- **Is the tree-repair count a useful per-model quality signal?**
  Could be a cheap proxy for "model coherence on this task" without
  needing the scorer. The 8B's 96% valid-token rate vs the 0.6B's 58%
  correlates with the LAS gap, but with only two data points we can't
  tell whether the relationship is linear, saturating, or coincidental.
- **Does a stronger UAS-favoring fallback help weak models?** The 0.6B
  data hints that right-branching is a meaningful prior for unparseable
  output. A hybrid (preserve labels but use right-branching heads when
  the model's head graph is broken) might give the best of both. Easy
  experiment.

## Changelog

- **2026-06-03** — Few-shot demonstrations sourced from UD_Latin-Perseus
  training data (punctuation-stripped) as an alternative to the hand-curated
  pool. `ExamplePool(path=…_perseus.conllu)` + a `-2shot-perseus` cache slug so
  it sits beside the hand-curated run. 0.6B run: Perseus beats hand-curated on
  LAS/CLAS both splits and cuts token-fallback to ~21–23% (hand-curated ~58–60%);
  8B run still pending (not loaded). See key finding #8.
- **2026-05-27** — Few-shot (2-shot) hand-curated Latin demonstrations
  injected as chat history. 0.6B LAS hurt on both splits; 8B LAS flat
  on poetry (+0.20), +2.36 on prose. Restores 8B's prose > poetry gap.
  Implementation: `LMStudioModel(k_shot=2, ...)`. Predictions cached at
  `predictions/<slug>-2shot/`. 12B run pending.
- **2026-05-18** — Re-ran 0.6B under the new minimal repair for
  apples-to-apples comparison with 8B (LAS 2.7 / 1.6). Pre-repair
  predictions archived at `predictions/qwen3-0.6b-mlx.pre-repair/`.
  Findings doc started.
- **2026-05-17** — Qwen3-VL-8B-Instruct-MLX run (LAS 18.2 poetry, 17.8 prose);
  switched to minimal tree repair (commit `9c230c9`).
- **2026-05-14** — Qwen3-0.6B-MLX baseline + error analysis under the
  old whole-sentence right-branching fallback (LAS 1.9 / 1.1, UAS 17.4 / 20.8).
- Earlier — Ported from Ollama to LM Studio; LatinPipe + UDPipe reference
  numbers locked in.
