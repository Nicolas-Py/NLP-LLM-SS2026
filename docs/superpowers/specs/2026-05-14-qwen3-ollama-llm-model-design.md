# Design: `OllamaLLMModel` — local-LLM dependency parser for `latinbench`

**Date:** 2026-05-14
**Status:** Draft (pending user review of this written spec).
**Related:** `.specs/2026-05-07-nlp-llm-ss2026-design.md` (original `latinbench` design).

---

## Goal

Add a new `Model` to `latinbench` that uses a local LLM (initially Qwen3-0.6B
served via Ollama on macOS) to attempt the EvaLatin 2024 dependency parsing
task with constrained structured output. Realistic expectation: a 0.6B model
will score poorly on this task — the goal is the plumbing, not the score.

User intent: pull a local LLM into the same bench as UDPipe and LatinPipe so we
can compare numbers and iterate on prompting. The class is parameterized on
`model_id` so swapping to `qwen3:4b`, `llama3.2:3b`, etc. is a one-liner with
its own cache dir, mirroring how `UdpipeModel` is parameterized on LINDAT
model id.

## Non-goals (v1)

- **MLX / non-Ollama backends.** A sibling class can be added later when
  there's a second backend to actually design against. Backend abstraction
  ahead of that is premature.
- **Few-shot prompting beyond one worked example.** One example in the system
  prompt is the v1 baseline; richer prompting is a follow-up.
- **Retry on malformed output.** A failed sentence falls back deterministically
  (see §5); the user can re-run with `force=True`.
- **Streaming or batched inference.** One HTTP call per sentence is fine for
  a few hundred test sentences per split.
- **Different quantizations.** Ollama's default for `qwen3:0.6b` is used as-is.
- **Hyperparameter sweeps** (temperature, `num_ctx`). Defaults only.

## File layout

One new file:

```
src/latinbench/models/ollama_llm.py
```

Imported and registered in:

```
src/latinbench/models/__init__.py     # adds "qwen3-ollama" entry
```

No changes to `core.py`, `score.py`, `data.py`, or `pyproject.toml`. No new
runtime dependencies (`requests` and `conllu` are already in `pyproject.toml`).

## Registration

`_make_registry()` in `src/latinbench/models/__init__.py` gets one new entry:

```python
return {
    "udpipe": UdpipeModel(),
    "latinpipe": LatinpipeModel(),
    "qwen3-ollama": OllamaLLMModel(),   # new
}
```

The existing `_LazyRegistry` already makes this construction lazy — Ollama
not running won't break unrelated `MODELS["udpipe"]` access. Construction
itself does no I/O; failures fire in `predict()`.

## The `Model` class API

```python
# src/latinbench/models/ollama_llm.py
from __future__ import annotations
from pathlib import Path

import conllu
import requests

from ..core import Model


DEFAULT_HOST = "http://localhost:11434"
DEFAULT_MODEL_ID = "qwen3:0.6b"


class OllamaLLMModel(Model):
    def __init__(
        self,
        model_id: str = DEFAULT_MODEL_ID,
        host: str = DEFAULT_HOST,
        num_ctx: int = 8192,
    ) -> None:
        self.model_id = model_id
        self.host = host.rstrip("/")
        self.num_ctx = num_ctx
        # filesystem-safe slug for predictions/<name>/: qwen3:0.6b -> qwen3-0.6b
        self.name = model_id.replace(":", "-").replace("/", "-")

    def predict(self, test_path: Path, out_path: Path) -> None:
        sentences = conllu.parse(test_path.read_text())
        total_toks = 0
        total_fallback = 0
        fallback_sents = 0
        for i, sent in enumerate(sentences):
            n_toks, n_fb = self._parse_one(sent)
            total_toks += n_toks
            total_fallback += n_fb
            if n_fb > 0:
                fallback_sents += 1
            if i % 25 == 0:
                print(f"[{self.name}] {i}/{len(sentences)} sentences")
        out_path.write_text("".join(s.serialize() for s in sentences))
        pct = (100.0 * total_fallback / total_toks) if total_toks else 0.0
        print(
            f"[{self.name}] {len(sentences)} sentences, {total_toks} tokens; "
            f"{total_fallback} fallback tokens ({pct:.1f}%) across "
            f"{fallback_sents} sentences"
        )
```

### Key choices

- Defaults make `OllamaLLMModel()` Just Work for Qwen3-0.6B against a local
  Ollama on the default port.
- `self.name` is derived from `model_id`, mirroring `UdpipeModel.name =
  self.model_id`. Cache lives at `predictions/qwen3-0.6b/`.
- No constructor-time network probe (matches `UdpipeModel`). Failures fire
  in `predict()` with a useful traceback (HTTP error or `ConnectionError`).
- Progress print every 25 sentences gives visible heartbeat without spam.

## Prompt + JSON schema

For each sentence, one `POST {host}/api/chat`. Multi-word tokens (CoNLL-U IDs
like `5-6`) are skipped — they don't get HEAD/DEPREL.

### Request body

```python
{
    "model": self.model_id,
    "messages": [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": _format_sentence(single)},
    ],
    "format": SCHEMA,                # JSON schema, structured output
    "stream": False,
    "options": {"num_ctx": self.num_ctx, "temperature": 0},
}
```

### `SYSTEM_PROMPT`

```
You are a Universal Dependencies parser for Latin. Given a list of tokens
with their lemma, UPOS, and morphological features, output the syntactic
head ID and dependency relation for each token, in input order.

Rules:
- `head` is the parent token's ID, or 0 if this token is the root.
- Exactly one token in the sentence has head=0 with deprel="root".
- `deprel` must be from the provided vocabulary.

Example input:
1  Marcus    PROPN  Case=Nom|Number=Sing
2  puellam   NOUN   Case=Acc|Number=Sing
3  amat      VERB   Mood=Ind|Person=3|VerbForm=Fin

Example output:
{"tokens": [
  {"id": 1, "head": 3, "deprel": "nsubj"},
  {"id": 2, "head": 3, "deprel": "obj"},
  {"id": 3, "head": 0, "deprel": "root"}
]}
```

### `_format_sentence(single)`

Tab-separated rows, one per single-word token:

```
1   Germania     germania     PROPN   Case=Nom|Gender=Fem|...
2   omnis        omnis        DET     Case=Nom|Gender=Fem|...
...
```

Columns: id, form, lemma, UPOS, FEATS. Empty FEATS rendered as `_`.

### `SCHEMA`

Computed once at module import:

```python
DEPREL_LABELS = _collect_deprels_from_gold()   # union from data/.../*gold/*.conllu

SCHEMA = {
    "type": "object",
    "properties": {
        "tokens": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "integer", "minimum": 1},
                    "head": {"type": "integer", "minimum": 0},
                    "deprel": {"type": "string", "enum": DEPREL_LABELS},
                },
                "required": ["id", "head", "deprel"],
            },
        },
    },
    "required": ["tokens"],
}
```

`_collect_deprels_from_gold()` walks the two gold files in
`data/EvaLatin_2024_Syntactic_Parsing_test_data_gold/`, collects the union
of seen `deprel` values from single-word tokens, sorts and returns them. This
is the UD-Latin label space — not per-token leakage; the vocabulary is
public from training treebanks. Pre-computed once at module import.

## Validation + fallback

After parsing the Ollama response JSON, walk the sentence's single-word
tokens and assign head/deprel with per-token fallback:

```python
def _parse_one(self, sent: conllu.TokenList) -> tuple[int, int]:
    single = [t for t in sent if isinstance(t["id"], int)]
    valid_ids = {t["id"] for t in single}

    try:
        pred_by_id = {
            p["id"]: p for p in self._call_ollama(single).get("tokens", [])
            if isinstance(p, dict) and isinstance(p.get("id"), int)
        }
    except (requests.RequestException, ValueError):
        pred_by_id = {}   # whole-sentence fallback

    n_fallback = 0
    for i, tok in enumerate(single):
        pred = pred_by_id.get(tok["id"])
        head = pred.get("head") if pred else None
        deprel = pred.get("deprel") if pred else None

        valid = (
            pred is not None
            and isinstance(head, int)
            and (head == 0 or head in valid_ids)
            and isinstance(deprel, str)
        )
        if not valid:
            head, deprel = _right_branching_default(single, i)
            n_fallback += 1
        tok["head"] = head
        tok["deprel"] = deprel
    return len(single), n_fallback


def _right_branching_default(single, i):
    n = len(single)
    if i == n - 1:
        return 0, "root"
    return single[i + 1]["id"], "dep"
```

### Failure modes covered

| Failure | Behavior |
|---|---|
| HTTP error / timeout / model not pulled | `_call_ollama` raises; caught; whole sentence falls back to right-branching default. |
| JSON missing a token id | That token only falls back. |
| `head` out of range or wrong type | That token only falls back. |
| `deprel` invalid type | That token only falls back. (Bad values are blocked by the enum schema; this is the bug-safety branch.) |

### End-of-run summary

`predict()` prints, after writing the output file:

```
[qwen3-0.6b] 100 sentences, 1840 tokens; 73 fallback tokens (3.9%) across 21 sentences
```

So the user sees the model's reliability separately from LAS/CLAS.

## Setup expectations

User installs on macOS (one-time):

```bash
brew install ollama
ollama serve                    # or use the menubar app
ollama pull qwen3:0.6b          # ~520MB
```

Sanity check:

```bash
curl http://localhost:11434/api/tags     # should list qwen3:0.6b
```

Then from a notebook or Python:

```python
from latinbench import Bench, MODELS
Bench().run(MODELS["qwen3-ollama"])
```

### Note on the gaianet GGUF link

The user-referenced `https://huggingface.co/gaianet/Qwen3-0.6B-GGUF` ships
the same Qwen3-0.6B weights packaged as GGUF. Ollama's `qwen3:0.6b` in its
registry is the same model and is one command to install. The gaianet GGUF
path would require writing a Modelfile and `ollama create`-ing it — same
end result, more steps. v1 uses `qwen3:0.6b` from Ollama's registry.
Switching to gaianet later is just a different `model_id` after the user
runs `ollama create my-qwen3 -f Modelfile`.

## Docs to update

- **README.md** — new subsection after *Reference scores*:
  *Trying a local LLM via Ollama* — install / serve / pull commands and a
  three-line Python example.
- **notebooks/02_compare_models.ipynb** — add `MODELS["qwen3-ollama"]` to the
  `compare([...])` cell. (Optional; doesn't block the new model from being
  runnable from a one-liner.)

## Open questions / future work

- **Few-shot prompting.** Adding 3-5 hand-picked examples (mix of prose +
  poetry) might lift the score a few points on a 0.6B model. Easy add via
  flag/constructor arg.
- **MLX backend.** Once it exists, factor out a `LLMBackend` ABC; until then
  YAGNI.
- **Retry on partial output.** If fallback rates are high in practice, a
  one-retry-with-stricter-prompt pass could help.
- **Larger Qwen3 sizes.** `qwen3:4b`, `qwen3:8b` — one-liner change, separate
  cache dir, expected to score meaningfully higher.
- **Constraining `head` to per-sentence range in the schema.** Currently the
  schema allows any non-negative int; we filter post-hoc. A per-sentence
  schema with `maximum: N` would push the constraint into the decoder.
  Skipped for v1 to keep the schema static (computed once).

## Implementation order

1. Add `_collect_deprels_from_gold` helper + static `DEPREL_LABELS` /
   `SCHEMA` at module top.
2. Add `OllamaLLMModel` class with constructor + `predict()` skeleton.
3. Add `_format_sentence`, `_call_ollama`, `_parse_one`,
   `_right_branching_default`.
4. Register in `models/__init__.py`.
5. Smoke-test against one short test sentence (manual `predict()` call to a
   one-sentence file).
6. Run end-to-end via `Bench().run(MODELS["qwen3-ollama"])`. Confirm output
   file is valid CoNLL-U and `scores.json` lands.
7. Update README with the Ollama section.

Tests are not specced here — `latinbench` itself doesn't have a test suite,
and the integration test is "run the bench and inspect the scores file".
The writing-plans step will revisit this.
