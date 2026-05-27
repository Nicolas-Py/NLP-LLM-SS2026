# Design: Few-shot prompting for LM Studio LLMs in `latinbench`

**Date:** 2026-05-27
**Status:** Draft (pending user review of this written spec).
**Related:** `2026-05-14-qwen3-ollama-llm-model-design.md` (introduces the LLM
model class that this design extends); `docs/01_findings.md` §"Next experiments"
item 2 (motivates this work).

---

## Goal

Add few-shot ("k-shot") in-context demonstrations to the existing LM Studio
LLM parser so we can benchmark it head-to-head against the current zero-shot
runs, across the same three models (qwen3-0.6B, qwen3-VL-8B, gemma-3-12B), with
results that drop seamlessly into the existing `Bench.compare(...)` DataFrame
and plot. `k` is a constructor argument with a default of 2 but free to vary.

User intent: test whether in-context examples help small *and* large LLMs on
Latin UD parsing, and produce results that sit next to the existing zero-shot
LLM and trained-parser numbers in the same table — no parallel reporting path.

## Non-goals (v1)

- **Retrieval-based or per-sentence example selection.** Static selection only:
  the same k examples are shown for every test sentence in a run. This isolates
  "does few-shot help?" from "does retrieval quality help?".
- **Multi-seed variance reporting.** Few-shot results are known to be sensitive
  to example choice. We expose `shot_seed` as a knob and run a single seed for
  headline numbers; aggregating multiple seeds is a future extension.
- **Few-shot on `UdpipeModel` / `LatinpipeModel`.** Trained parsers don't
  consume demonstrations; the abstraction stays LLM-local.
- **Adding k-shot entries to the `MODELS` registry.** Registry would balloon
  (one entry per (model, k, seed)). Users construct k-shot variants directly
  in the notebook; the registry keeps just the canonical zero-shot entries.
- **Auto-downloaded external example corpora.** The default example pool ships
  as a small committed CoNLL-U file; users can point at any other CoNLL-U file
  via `ExamplePool(path=...)`.
- **Larger k by default.** k=2 is the default and is what the headline numbers
  use; higher k is supported (capped by pool size) but not benchmarked in v1.

## Methodology — best practices baked in

These are the controls the design enforces; they're listed here so the
experimental contract is explicit.

1. **Demonstrations are disjoint from the EvaLatin test corpus by construction.**
   The bundled pool is hand-curated Latin sentences that are not in
   `data/EvaLatin_2024_Syntactic_Parsing_test_data*`. No leakage.
2. **Static selection across the run.** The k demonstrations are sampled once
   at `LMStudioModel.__init__` and reused for every test sentence. Any score
   delta between k=0 and k=2 isolates the effect of demonstrations themselves.
3. **Deterministic sampling.** `random.Random(seed).sample(...)` — same
   `(pool, k, shot_seed)` → identical demonstrations across machines and runs.
4. **Identical prompt scaffolding across k.** The user-turn formatter is shared
   between demonstrations and the target sentence — what the model sees for a
   demo is bit-for-bit what it sees for the target. Same system prompt, same
   JSON schema on the final completion. Only the chat history differs.
5. **Multi-turn chat format, not packed user message.** Demonstrations are
   presented as alternating `user` / `assistant` turns, which matches how
   modern instruction-tuned models (Qwen3, Gemma 3) were post-trained on chat.
6. **Per-(model, k, seed) cache isolation.** Each variant gets its own
   `predictions/<slug>/` so 0-shot and k-shot results coexist and can be
   compared in the same DataFrame without re-running.
7. **Documented pool stability.** Changing the bundled example file silently
   invalidates cached k-shot scores under that slug. The pool file is committed
   and considered stable; the standard `force=True` / `rm -rf predictions/<slug>/`
   workflow handles invalidation when it does change.

## File layout

Two new files, one tweak:

```
src/latinbench/few_shot.py                 # new: ExamplePool class
src/latinbench/few_shot_examples.conllu    # new: 6 hand-curated demonstrations
src/latinbench/models/lmstudio_llm.py      # tweak: k_shot / example_pool / shot_seed args
```

No changes to `core.py`, `score.py`, `data.py`, `__init__.py`,
`models/__init__.py`, `pyproject.toml`. No new runtime dependencies.

## `ExamplePool`

`src/latinbench/few_shot.py`, ~30 lines:

```python
DEFAULT_EXAMPLES_PATH = Path(__file__).parent / "few_shot_examples.conllu"

class ExamplePool:
    def __init__(self, path: Path | None = None) -> None:
        self.path = Path(path) if path else DEFAULT_EXAMPLES_PATH
        self._sentences = conllu.parse(self.path.read_text())
        if not self._sentences:
            raise ValueError(f"Empty example pool: {self.path}")

    def __len__(self) -> int: ...
    def sample(self, k: int, seed: int = 0) -> list[conllu.TokenList]: ...
```

Semantics:

- Loads + parses the CoNLL-U file once at construction. `conllu.parse` returns
  `TokenList` objects identical in shape to gold sentences, so downstream
  formatters can treat them uniformly.
- `sample(k, seed)`:
  - `k == 0` → `[]` (lets the call-site code iterate unconditionally).
  - `k < 0` → `ValueError("k must be >= 0, got -1")`.
  - `k > len(self)` → `ValueError("requested k=8 but pool only has 6 sentences at <path>")`.
  - Otherwise: `random.Random(seed).sample(self._sentences, k)`. Sampling is
    without replacement (no duplicate demos) and deterministic.
- `len(pool)` returns the pool size — useful for the error message above and
  for tests.

## Bundled `few_shot_examples.conllu`

6 hand-curated classical Latin sentences, each 5–10 single-word tokens, with
all five UD columns filled (form, lemma, UPOS, feats, head, deprel). Coverage:

| # | construction targeted              | deprels exercised                          |
|---|------------------------------------|--------------------------------------------|
| 1 | SVO with adjective modifier        | `nsubj`, `obj`, `amod`, `root`             |
| 2 | Copular clause with predicate      | `cop`, `nsubj`, `root`                     |
| 3 | Coordinated noun phrases           | `cc`, `conj`, `nmod`                       |
| 4 | Verb + prepositional phrase        | `case`, `obl`, `det`                       |
| 5 | Subordinate clause (relative or cum) | `acl:relcl` or `advcl`, `mark`           |
| 6 | Participial / ablative absolute    | `advcl:pred`, `nmod`                       |

Constraints on the bundled file:

- **Deprels drawn from the EvaLatin gold label set only** (the union computed
  by `_collect_deprels_from_gold` in `lmstudio_llm.py`). Not strictly required
  — assistant turns aren't schema-validated — but keeps demonstrations in
  distribution with what the schema then forces the model to output.
- **No multi-word token rows** in the demos (e.g. no `19-20 locumque ...`).
  Keeps the demo formatter simple; multi-word tokens would never appear in an
  assistant JSON anyway.
- **Sentences composed from canonical Latin idiom** (Caesar/Cicero style),
  not lifted verbatim from any single UD treebank we might later use as an
  alternate pool source.

## `LMStudioModel` changes

`src/latinbench/models/lmstudio_llm.py`. New constructor signature:

```python
def __init__(
    self,
    model_id: str = DEFAULT_MODEL_ID,
    host: str = DEFAULT_HOST,
    num_workers: int = 8,
    max_tokens: int = 4096,
    temperature: float = 0.3,
    k_shot: int = 0,                                   # NEW
    example_pool: ExamplePool | None = None,           # NEW
    shot_seed: int = 0,                                # NEW
) -> None:
```

Init-time logic:

```python
self.k_shot = k_shot
self.shot_seed = shot_seed
self._pool = example_pool if example_pool is not None else (
    ExamplePool() if k_shot > 0 else None
)
self._demonstrations = (
    self._pool.sample(k_shot, shot_seed) if self._pool else []
)

slug = model_id.replace(":", "-").replace("/", "-")
if k_shot > 0:
    slug += f"-{k_shot}shot"
    if shot_seed != 0:
        slug += f"-s{shot_seed}"
self.name = slug
```

Properties:

- **Zero-shot path unchanged.** `k_shot=0` → `self._demonstrations = []` →
  `_call_llm` builds the same `[system, user]` messages it does today.
- **Default seed (0) gets a clean `-{k}shot` suffix.** Only non-default seeds
  add `-s{n}`.
- **Pool is loaded lazily — only when `k_shot > 0` and the caller didn't pass
  one in.** Zero-shot runs don't touch the example file at all.

`_call_llm` change — inject demonstration turns before the target user message:

```python
def _call_llm(self, single: list) -> dict:
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for demo in self._demonstrations:
        demo_single = [t for t in demo if isinstance(t["id"], int)]
        messages.append({"role": "user",
                         "content": _format_user_message(demo_single)})
        messages.append({"role": "assistant",
                         "content": _format_assistant_response(demo_single)})
    messages.append({"role": "user", "content": _format_user_message(single)})
    body = {
        "model": self.model_id,
        "messages": messages,
        "response_format": {...},   # unchanged
        ...
    }
    ...
```

`_format_assistant_response(single)` is a new module-level helper:

```python
def _format_assistant_response(single: list) -> str:
    tokens = [
        {"id": t["id"], "head": t["head"], "deprel": t["deprel"]}
        for t in single
    ]
    return json.dumps({"tokens": tokens}, ensure_ascii=False)
```

Notes:

- The user-turn formatter `_format_user_message` is **reused unchanged** — a
  demonstration's token table is byte-identical in shape to a target's. This
  is what makes "k=0 vs k=2 isolates the effect of demonstrations" actually
  true.
- LM Studio's `response_format: {type: "json_schema"}` only constrains the
  final assistant generation. Historical assistant turns are plain JSON
  strings; no schema validation on them. (Documented LM Studio behaviour.)
- All parallelism, partial-resume, tree-repair, and fallback logic in
  `predict` / `_parse_one` is **unchanged**. Demonstrations are sampled once
  per `LMStudioModel` instance and shared across all parallel worker calls.

## Cache & naming

Existing cache convention (`predictions/<model.name>/{scores.json,*_pred.conllu,*.partial.json}`)
is reused. Variants:

| construction                                                          | `name` slug                                  |
|------------------------------------------------------------------------|----------------------------------------------|
| `LMStudioModel("qwen3-vl-8b-instruct-mlx")`                           | `qwen3-vl-8b-instruct-mlx`                   |
| `LMStudioModel("qwen3-vl-8b-instruct-mlx", k_shot=2)`                 | `qwen3-vl-8b-instruct-mlx-2shot`             |
| `LMStudioModel("qwen3-vl-8b-instruct-mlx", k_shot=2, shot_seed=7)`    | `qwen3-vl-8b-instruct-mlx-2shot-s7`          |
| `LMStudioModel("qwen3-vl-8b-instruct-mlx", k_shot=4)`                 | `qwen3-vl-8b-instruct-mlx-4shot`             |

Existing committed zero-shot predictions are untouched; their cache dirs and
slugs are unchanged.

## Notebook integration

`notebooks/02_compare_models.ipynb` gains one cell that constructs k-shot
variants and runs them through the existing `compare`:

```python
from latinbench.models.lmstudio_llm import LMStudioModel

few_shot_models = [
    LMStudioModel("qwen3-0.6b-mlx",             k_shot=2),
    LMStudioModel("qwen3-vl-8b-instruct-mlx",   k_shot=2),
    LMStudioModel("google/gemma-3-12b",         k_shot=2),
]

df_all = bench.compare([
    MODELS["udpipe"], MODELS["latinpipe"],
    MODELS["qwen3-lmstudio"],
    MODELS["qwen3-vl-8b-lmstudio"],
    MODELS["gemma-3-12b-lmstudio"],
    *few_shot_models,
])
bench.plot(df_all, metric="LAS")
```

Output: the same tidy `(system, split, metric, P, R, F1)` DataFrame the
existing `compare` produces, with three additional `system` rows per split
for the k=2 variants. The grouped bar plot shows 0-shot and 2-shot bars next
to each other within each model family.

## Error handling

Deliberately minimal, matching the existing file's style:

| condition                                | behaviour                                            |
|------------------------------------------|------------------------------------------------------|
| `ExamplePool(Path("missing.conllu"))`    | `FileNotFoundError` from `Path.read_text` (unwrapped)|
| empty pool file                          | `ValueError("Empty example pool: <path>")`           |
| `pool.sample(k=-1)`                      | `ValueError("k must be >= 0, got -1")`               |
| `pool.sample(k > len(pool))`             | `ValueError("requested k=N but pool only has M ...")`|
| LM Studio HTTP error / malformed JSON    | **unchanged** — `_parse_one` falls back per sentence |
| demonstration assistant turn malformed   | not possible — we produce them via `json.dumps`      |

## Testing

`tests/test_few_shot.py` (new):

- `ExamplePool.sample(k, seed)` is deterministic across calls (same return
  for same args).
- `sample(0)` returns `[]`.
- `sample(k > len(pool))` raises `ValueError`.
- Bundled pool loads at least 6 sentences; every single-word token in every
  pool sentence has integer `head` and non-empty `deprel` (sanity check on
  the committed file).
- Every deprel used in the bundled pool is in the EvaLatin gold label set.

`tests/test_lmstudio_llm.py` (extend):

- `LMStudioModel("foo").name == "foo"`.
- `LMStudioModel("foo", k_shot=2).name == "foo-2shot"`.
- `LMStudioModel("foo", k_shot=2, shot_seed=7).name == "foo-2shot-s7"`.
- A network-free unit test for message construction: stub
  `requests.post` (or refactor message building into a pure helper) and
  assert the message list has the expected `[system, user, assistant, user,
  assistant, user]` structure for k=2, with the demonstrations slotted in
  by index.

Smoke test (manual, run from the notebook on the user's box):

- One 0.6B `k_shot=2` run on the `poetry` split.
- Confirm: scores file lands at `predictions/qwen3-0.6b-mlx-2shot/scores.json`;
  `partial.json` sidecar updates as sentences complete; predictions differ
  from the zero-shot cache; the bench `compare` table shows both 0-shot and
  2-shot rows for `qwen3-0.6b-mlx`.

## Risks and open questions

- **Schema enum mismatch.** If a deprel I bake into the bundled examples
  later disappears from the EvaLatin gold corpus (e.g. annotation guidelines
  change), the demonstration would use a label the schema then forbids the
  model from emitting. Mitigation: the testing step that asserts pool
  deprels ⊆ gold-derived enum will catch this; addressing it is a one-line
  fix to the pool file.
- **`response_format` interaction with chat history on non-LM-Studio backends.**
  We rely on LM Studio applying the JSON schema only to the final completion.
  Other OpenAI-compatible servers may behave differently — out of scope for
  v1 (we only target LM Studio), but worth a note if/when we add a backend.
- **Context length at higher k.** k=2 with ~10-token examples adds ~300 input
  tokens. k=8 would add ~1200, still well inside the 4096 `max_tokens` budget,
  but worth re-checking if the pool ever grows or the examples get longer.
- **Demonstration "leakage" of UD style.** Even though the bundled examples
  are not from EvaLatin, they use the same UD scheme and the same EvaLatin-
  derived deprel inventory. That's intentional (we want demos to actually
  help, not confuse) but means the demonstrations *are* somewhat tailored to
  the eval. Document this in `01_findings.md` when reporting results.
