# Few-shot LM Studio LLM benchmark — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add k-shot in-context prompting to `LMStudioModel` so we can benchmark 2-shot (configurable) variants of the existing qwen3-0.6B / qwen3-VL-8B / gemma-3-12B LLM parsers alongside the current 0-shot LLM runs and the trained baselines, all in one `Bench.compare(...)` DataFrame.

**Architecture:** A new `ExamplePool` class (`src/latinbench/few_shot.py`) loads a bundled CoNLL-U file of 6 hand-curated Latin sentences and supports deterministic `sample(k, seed)`. `LMStudioModel` gains three constructor args (`k_shot`, `example_pool`, `shot_seed`); when `k_shot > 0` the chat-completion call interleaves `(user demo, assistant gold JSON)` turns before the target sentence's user turn. Cache slug auto-suffixes with `-{k}shot[-s{seed}]` so 0-shot results are preserved untouched.

**Tech Stack:** Python 3.12, `conllu` library, `requests`, `pytest`, LM Studio (OpenAI-compatible API).

**Spec:** [docs/superpowers/specs/2026-05-27-few-shot-llm-benchmark-design.md](../specs/2026-05-27-few-shot-llm-benchmark-design.md)

---

## File layout summary

```
Create: src/latinbench/few_shot.py                    # ExamplePool class
Create: src/latinbench/few_shot_examples.conllu       # 6 hand-curated demos
Modify: src/latinbench/models/lmstudio_llm.py         # k_shot args + chat history
Create: tests/test_few_shot.py                        # ExamplePool + bundle tests
Modify: tests/test_lmstudio_llm.py                    # few-shot name + messages tests
Modify: notebooks/02_compare_models.ipynb             # comparison cell (manual)
```

No changes to `core.py`, `score.py`, `data.py`, `__init__.py`, `models/__init__.py`, `pyproject.toml`.

---

## Task 1: `ExamplePool` class (no bundled file yet)

Build the pool abstraction first using `tmp_path` fixtures in the tests. The bundled CoNLL-U file lands in Task 2 — keeping these two tasks separate keeps each commit small and lets the file content get its own focused review.

**Files:**
- Create: `src/latinbench/few_shot.py`
- Create: `tests/test_few_shot.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_few_shot.py` with these contents:

```python
from __future__ import annotations

import pytest

from latinbench.few_shot import ExamplePool


_TWO_SENT_POOL = (
    "# sent_id = ex-1\n"
    "# text = a b\n"
    "1\ta\ta\tX\t_\t_\t2\tnsubj\t_\t_\n"
    "2\tb\tb\tX\t_\t_\t0\troot\t_\t_\n"
    "\n"
    "# sent_id = ex-2\n"
    "# text = c d\n"
    "1\tc\tc\tX\t_\t_\t2\tnsubj\t_\t_\n"
    "2\td\td\tX\t_\t_\t0\troot\t_\t_\n"
)


def test_pool_loads_from_explicit_path(tmp_path):
    p = tmp_path / "pool.conllu"
    p.write_text(_TWO_SENT_POOL)
    pool = ExamplePool(p)
    assert len(pool) == 2


def test_pool_empty_file_raises(tmp_path):
    p = tmp_path / "empty.conllu"
    p.write_text("")
    with pytest.raises(ValueError, match="Empty example pool"):
        ExamplePool(p)


def test_pool_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        ExamplePool(tmp_path / "nope.conllu")


def test_sample_zero_returns_empty(tmp_path):
    p = tmp_path / "pool.conllu"
    p.write_text(_TWO_SENT_POOL)
    pool = ExamplePool(p)
    assert pool.sample(0) == []


def test_sample_negative_k_raises(tmp_path):
    p = tmp_path / "pool.conllu"
    p.write_text(_TWO_SENT_POOL)
    pool = ExamplePool(p)
    with pytest.raises(ValueError, match="k must be >= 0"):
        pool.sample(-1)


def test_sample_k_larger_than_pool_raises(tmp_path):
    p = tmp_path / "pool.conllu"
    p.write_text(_TWO_SENT_POOL)
    pool = ExamplePool(p)
    with pytest.raises(ValueError, match="pool only has 2"):
        pool.sample(5)


def test_sample_is_deterministic_given_same_seed(tmp_path):
    p = tmp_path / "pool.conllu"
    p.write_text(_TWO_SENT_POOL)
    pool = ExamplePool(p)
    a = pool.sample(2, seed=0)
    b = pool.sample(2, seed=0)
    # Same seed → identical order of returned sentences
    assert [s.metadata["sent_id"] for s in a] == [s.metadata["sent_id"] for s in b]


def test_sample_returns_k_distinct_sentences(tmp_path):
    p = tmp_path / "pool.conllu"
    p.write_text(_TWO_SENT_POOL)
    pool = ExamplePool(p)
    sampled = pool.sample(2, seed=0)
    sent_ids = [s.metadata["sent_id"] for s in sampled]
    assert len(set(sent_ids)) == 2  # sampling is without replacement
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/pytest tests/test_few_shot.py -v`
Expected: All fail with `ModuleNotFoundError: No module named 'latinbench.few_shot'`

- [ ] **Step 3: Implement `ExamplePool`**

Create `src/latinbench/few_shot.py`:

```python
"""Example pool for few-shot prompting of LLM-backed parsers.

Loads a CoNLL-U file of demonstration sentences and exposes deterministic
sampling. Used by `LMStudioModel(k_shot=k, ...)` to inject in-context
demonstrations into the chat history.

The default pool ships in `few_shot_examples.conllu` alongside this module.
"""
from __future__ import annotations

import random
from pathlib import Path

import conllu


DEFAULT_EXAMPLES_PATH = Path(__file__).parent / "few_shot_examples.conllu"


class ExamplePool:
    """A bag of demonstration sentences, sampled deterministically.

    Sentences are parsed once at construction. `sample(k, seed)` is pure
    in `(k, seed)` — same args, same returned sentences, no shared state.
    """

    def __init__(self, path: Path | None = None) -> None:
        self.path = Path(path) if path else DEFAULT_EXAMPLES_PATH
        self._sentences = conllu.parse(self.path.read_text())
        if not self._sentences:
            raise ValueError(f"Empty example pool: {self.path}")

    def __len__(self) -> int:
        return len(self._sentences)

    def sample(self, k: int, seed: int = 0) -> list[conllu.TokenList]:
        if k < 0:
            raise ValueError(f"k must be >= 0, got {k}")
        if k == 0:
            return []
        if k > len(self._sentences):
            raise ValueError(
                f"requested k={k} but pool only has {len(self._sentences)} "
                f"sentences at {self.path}"
            )
        return random.Random(seed).sample(self._sentences, k)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/test_few_shot.py -v`
Expected: 8 passed.

- [ ] **Step 5: Commit**

```bash
git add src/latinbench/few_shot.py tests/test_few_shot.py
git commit -m "$(cat <<'EOF'
ExamplePool: deterministic few-shot demonstration sampling

Loads a CoNLL-U file of demonstration sentences and exposes
sample(k, seed) for in-context examples in LLM prompts. Pure
in (k, seed) — same args yield identical samples across calls.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 2: Bundled `few_shot_examples.conllu` + sanity tests

Hand-curate 6 Latin sentences covering the major UD construction types listed in the spec. Add tests that catch annotation regressions on the committed file.

**Files:**
- Create: `src/latinbench/few_shot_examples.conllu`
- Modify: `tests/test_few_shot.py` (append new tests)

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_few_shot.py`:

```python
import conllu

from latinbench.few_shot import DEFAULT_EXAMPLES_PATH, ExamplePool
from latinbench.models.lmstudio_llm import DEPREL_LABELS


def test_default_pool_loads():
    pool = ExamplePool()
    assert len(pool) >= 6


def test_default_pool_every_token_has_head_and_deprel():
    pool = ExamplePool()
    for sent in pool._sentences:
        for tok in sent:
            if isinstance(tok["id"], int):
                assert isinstance(tok["head"], int), (
                    f"sent {sent.metadata.get('sent_id')} token id={tok['id']} "
                    f"head not int: {tok['head']!r}"
                )
                assert isinstance(tok["deprel"], str) and tok["deprel"], (
                    f"sent {sent.metadata.get('sent_id')} token id={tok['id']} "
                    f"missing deprel"
                )


def test_default_pool_deprels_are_in_evalatin_label_set():
    """Every deprel used in the bundled pool must be in the EvaLatin gold
    label inventory. Otherwise the JSON-schema enum in lmstudio_llm.py would
    forbid the model from ever emitting it, which makes the demonstration
    inconsistent with what the model can produce at test time."""
    pool = ExamplePool()
    used = {
        tok["deprel"]
        for sent in pool._sentences
        for tok in sent
        if isinstance(tok["id"], int)
    }
    extra = used - set(DEPREL_LABELS)
    assert not extra, (
        f"pool uses deprels not in EvaLatin gold inventory: {sorted(extra)}"
    )


def test_default_pool_has_no_multi_word_tokens():
    """Demonstrations should not contain MWT rows (e.g. `2-3 puellaeque ...`),
    since the prompt's user-message formatter ignores those and the assistant
    JSON output never references them."""
    pool = ExamplePool()
    for sent in pool._sentences:
        for tok in sent:
            assert isinstance(tok["id"], int), (
                f"sent {sent.metadata.get('sent_id')} has MWT row id={tok['id']!r}"
            )
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/pytest tests/test_few_shot.py::test_default_pool_loads -v`
Expected: FAIL with `FileNotFoundError: ... few_shot_examples.conllu`

- [ ] **Step 3: Create the bundled CoNLL-U file**

Create `src/latinbench/few_shot_examples.conllu` with exactly this content:

```
# sent_id = fs-001
# text = puella bonum librum legit
1	puella	puella	NOUN	A1	Case=Nom|Gender=Fem|InflClass=IndEurA|Number=Sing	4	nsubj	_	_
2	bonum	bonus	ADJ	C1	Case=Acc|Degree=Pos|Gender=Masc|InflClass=IndEurO|Number=Sing	3	amod	_	_
3	librum	liber	NOUN	A2	Case=Acc|Gender=Masc|InflClass=IndEurO|Number=Sing	4	obj	_	_
4	legit	lego	VERB	B3	Aspect=Imp|InflClass=LatX|Mood=Ind|Number=Sing|Person=3|Tense=Pres|VerbForm=Fin|Voice=Act	0	root	_	_

# sent_id = fs-002
# text = Marcus poeta est
1	Marcus	marcus	PROPN	A2	Case=Nom|Gender=Masc|InflClass=IndEurO|Number=Sing	2	nsubj	_	_
2	poeta	poeta	NOUN	A1	Case=Nom|Gender=Masc|InflClass=IndEurA|Number=Sing	0	root	_	_
3	est	sum	AUX	Z3	Aspect=Imp|InflClass=LatAnom|Mood=Ind|Number=Sing|Person=3|Tense=Pres|VerbForm=Fin	2	cop	_	_

# sent_id = fs-003
# text = pueri et puellae cantant
1	pueri	puer	NOUN	A2	Case=Nom|Gender=Masc|InflClass=IndEurO|Number=Plur	4	nsubj	_	_
2	et	et	CCONJ	S	_	3	cc	_	_
3	puellae	puella	NOUN	A1	Case=Nom|Gender=Fem|InflClass=IndEurA|Number=Plur	1	conj	_	_
4	cantant	canto	VERB	B1	Aspect=Imp|InflClass=LatA|Mood=Ind|Number=Plur|Person=3|Tense=Pres|VerbForm=Fin|Voice=Act	0	root	_	_

# sent_id = fs-004
# text = agricola in agro laborat
1	agricola	agricola	NOUN	A1	Case=Nom|Gender=Masc|InflClass=IndEurA|Number=Sing	4	nsubj	_	_
2	in	in	ADP	R	_	3	case	_	_
3	agro	ager	NOUN	A2	Case=Abl|Gender=Masc|InflClass=IndEurO|Number=Sing	4	obl	_	_
4	laborat	laboro	VERB	B1	Aspect=Imp|InflClass=LatA|Mood=Ind|Number=Sing|Person=3|Tense=Pres|VerbForm=Fin|Voice=Act	0	root	_	_

# sent_id = fs-005
# text = video puerum qui currit
1	video	video	VERB	B2	Aspect=Imp|InflClass=LatE|Mood=Ind|Number=Sing|Person=1|Tense=Pres|VerbForm=Fin|Voice=Act	0	root	_	_
2	puerum	puer	NOUN	A2	Case=Acc|Gender=Masc|InflClass=IndEurO|Number=Sing	1	obj	_	_
3	qui	qui	PRON	I	Case=Nom|Gender=Masc|InflClass=LatPron|Number=Sing|PronType=Rel	4	nsubj	_	_
4	currit	curro	VERB	B3	Aspect=Imp|InflClass=LatX|Mood=Ind|Number=Sing|Person=3|Tense=Pres|VerbForm=Fin|Voice=Act	2	acl:relcl	_	_

# sent_id = fs-006
# text = Caesare interfecto bellum exarsit
1	Caesare	caesar	PROPN	A3	Case=Abl|Gender=Masc|InflClass=IndEurX|Number=Sing	2	nsubj	_	_
2	interfecto	interficio	VERB	Y3	Aspect=Perf|Case=Abl|Degree=Pos|Gender=Masc|InflClass=LatX|InflClass[nominal]=IndEurO|Number=Sing|VerbForm=Part|Voice=Pass	4	advcl:pred	_	_
3	bellum	bellum	NOUN	A2	Case=Nom|Gender=Neut|InflClass=IndEurO|Number=Sing	4	nsubj	_	_
4	exarsit	exardesco	VERB	B3	Aspect=Perf|InflClass=LatX|Mood=Ind|Number=Sing|Person=3|Tense=Past|VerbForm=Fin|Voice=Act	0	root	_	_

```

The trailing blank line is required by the CoNLL-U format and by `conllu.parse`.

**Note on tab characters:** every `	` in the file content above is a single TAB (`\t`). When editing in PyCharm, set the indent style to "tab" for this file or use the file's existing tab characters as reference — CoNLL-U requires literal tabs between columns, not spaces.

- [ ] **Step 4: Run all `test_few_shot.py` tests to verify they pass**

Run: `.venv/bin/pytest tests/test_few_shot.py -v`
Expected: 12 passed (8 from Task 1 + 4 new bundle tests).

If `test_default_pool_deprels_are_in_evalatin_label_set` fails, the test message names the offending deprels. If `cop` or `acl:relcl` is rejected, those labels are absent from the EvaLatin gold inventory — swap the offending sentence in `few_shot_examples.conllu` for one that uses a deprel from the gold set (run `python -c "from latinbench.models.lmstudio_llm import DEPREL_LABELS; print(DEPREL_LABELS)"` to see the full inventory) and re-run.

- [ ] **Step 5: Commit**

```bash
git add src/latinbench/few_shot_examples.conllu tests/test_few_shot.py
git commit -m "$(cat <<'EOF'
few-shot: bundle 6 hand-curated Latin demonstrations

Six short classical-Latin sentences with full UD annotations,
covering SVO+amod, copula, coordination, prep phrase, relative
clause, and ablative absolute. Deprels constrained to the
EvaLatin gold label inventory so the in-context demonstrations
stay in-distribution with what the JSON-schema decoder allows
the model to emit at test time.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 3: `LMStudioModel` constructor — `k_shot` / `example_pool` / `shot_seed`

Add the three new constructor args and the cache-slug suffix logic. This task does **not** yet inject demonstrations into the chat — it just gets the storage and naming right. Task 4 wires demonstrations into the prompt.

**Files:**
- Modify: `src/latinbench/models/lmstudio_llm.py:37-51`
- Modify: `tests/test_lmstudio_llm.py:1-18` (imports), then append new tests

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_lmstudio_llm.py` (after the existing `test_name_slug_*` block, around line 122):

```python
# ---------- few-shot constructor / name slug ----------

def test_name_slug_zero_shot_default_unchanged():
    """Existing behaviour: k_shot defaults to 0, no slug suffix."""
    assert LMStudioModel(model_id="qwen3-0.6b-mlx").name == "qwen3-0.6b-mlx"


def test_name_slug_k_shot_2_appends_2shot():
    m = LMStudioModel(model_id="qwen3-0.6b-mlx", k_shot=2)
    assert m.name == "qwen3-0.6b-mlx-2shot"


def test_name_slug_k_shot_4_appends_4shot():
    m = LMStudioModel(model_id="qwen3-0.6b-mlx", k_shot=4)
    assert m.name == "qwen3-0.6b-mlx-4shot"


def test_name_slug_non_default_seed_appends_seed_suffix():
    m = LMStudioModel(model_id="qwen3-0.6b-mlx", k_shot=2, shot_seed=7)
    assert m.name == "qwen3-0.6b-mlx-2shot-s7"


def test_name_slug_default_seed_does_not_add_seed_suffix():
    """Common case stays clean: shot_seed=0 → no -s0 suffix."""
    m = LMStudioModel(model_id="qwen3-0.6b-mlx", k_shot=2, shot_seed=0)
    assert m.name == "qwen3-0.6b-mlx-2shot"


def test_k_shot_zero_does_not_load_pool():
    """Zero-shot construction must not touch the example file at all."""
    m = LMStudioModel(model_id="qwen3-0.6b-mlx", k_shot=0)
    assert m._demonstrations == []
    assert m._pool is None


def test_k_shot_positive_uses_default_pool():
    m = LMStudioModel(model_id="qwen3-0.6b-mlx", k_shot=2)
    assert m._pool is not None
    assert len(m._demonstrations) == 2


def test_k_shot_accepts_explicit_pool(tmp_path):
    p = tmp_path / "custom.conllu"
    p.write_text(
        "# sent_id = c-1\n"
        "1\tx\tx\tX\t_\t_\t0\troot\t_\t_\n"
        "\n"
        "# sent_id = c-2\n"
        "1\ty\ty\tX\t_\t_\t0\troot\t_\t_\n"
    )
    from latinbench.few_shot import ExamplePool
    pool = ExamplePool(p)
    m = LMStudioModel(model_id="qwen3-0.6b-mlx", k_shot=2, example_pool=pool)
    assert m._pool is pool
    assert len(m._demonstrations) == 2
    ids = {s.metadata["sent_id"] for s in m._demonstrations}
    assert ids == {"c-1", "c-2"}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/pytest tests/test_lmstudio_llm.py -v -k "k_shot or 2shot or shot_seed or seed_suffix or zero_shot"`
Expected: All new tests fail with `TypeError: __init__() got an unexpected keyword argument 'k_shot'`.

- [ ] **Step 3: Update `LMStudioModel.__init__`**

Edit `src/latinbench/models/lmstudio_llm.py`. Replace the existing `__init__` (lines 38–51) with:

```python
    def __init__(
        self,
        model_id: str = DEFAULT_MODEL_ID,
        host: str = DEFAULT_HOST,
        num_workers: int = 8,
        max_tokens: int = 4096,
        temperature: float = 0.3,
        k_shot: int = 0,
        example_pool: "ExamplePool | None" = None,
        shot_seed: int = 0,
    ) -> None:
        self.model_id = model_id
        self.host = host.rstrip("/")
        self.num_workers = num_workers
        self.max_tokens = max_tokens
        self.temperature = temperature
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

Also add the import near the top of the file (after the existing imports around line 15):

```python
from ..few_shot import ExamplePool
```

The full top-of-file imports block should now be:

```python
from __future__ import annotations
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock

import conllu
import requests

from ..core import Model
from ..data import gold_path
from ..few_shot import ExamplePool
```

- [ ] **Step 4: Run all tests to verify they pass and nothing regressed**

Run: `.venv/bin/pytest tests/ -v`
Expected: All tests pass — the new few-shot tests, the existing `test_name_slug_*` tests, and the existing `test_call_llm_posts_correct_body` (which still applies to k=0 default behaviour, since this task didn't touch `_call_llm`).

- [ ] **Step 5: Commit**

```bash
git add src/latinbench/models/lmstudio_llm.py tests/test_lmstudio_llm.py
git commit -m "$(cat <<'EOF'
LMStudioModel: k_shot/example_pool/shot_seed constructor args

Adds the few-shot configuration knobs to LMStudioModel:
  - k_shot (default 0 — zero-shot, no behaviour change)
  - example_pool (default = bundled ExamplePool when k_shot > 0)
  - shot_seed (default 0)

Cache slug suffixes with -{k}shot[-s{seed}] when k_shot > 0, so
few-shot runs land in their own predictions/<slug>/ dir and the
existing zero-shot cache is preserved untouched.

Demonstrations are sampled once at __init__ and stored on the
instance; the prompt-injection itself lands in the next commit.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 4: Inject demonstrations into the chat messages

Refactor message building out of `_call_llm` into a pure `_build_messages` helper that can be unit-tested without HTTP, then wire the demonstrations through it. Also add the `_format_assistant_response` helper.

**Files:**
- Modify: `src/latinbench/models/lmstudio_llm.py:125-147` (`_call_llm` body)
- Modify: `tests/test_lmstudio_llm.py` (append new tests)

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_lmstudio_llm.py`:

```python
# ---------- _format_assistant_response ----------

def test_format_assistant_response_emits_compact_json():
    from latinbench.models.lmstudio_llm import _format_assistant_response

    sent = conllu.parse(
        "# sent_id = t\n"
        "1\tMarcus\tmarcus\tPROPN\t_\t_\t2\tnsubj\t_\t_\n"
        "2\tpoeta\tpoeta\tNOUN\t_\t_\t0\troot\t_\t_\n"
        "3\test\tsum\tAUX\t_\t_\t2\tcop\t_\t_\n"
    )[0]
    single = [t for t in sent if isinstance(t["id"], int)]
    out = _format_assistant_response(single)
    parsed = json.loads(out)
    assert parsed == {"tokens": [
        {"id": 1, "head": 2, "deprel": "nsubj"},
        {"id": 2, "head": 0, "deprel": "root"},
        {"id": 3, "head": 2, "deprel": "cop"},
    ]}


# ---------- _build_messages (few-shot chat history) ----------

def _target_single():
    return [
        {"id": 1, "form": "a", "lemma": "_", "upos": "X", "feats": None},
        {"id": 2, "form": "b", "lemma": "_", "upos": "X", "feats": None},
    ]


def test_build_messages_zero_shot_is_system_then_user():
    m = LMStudioModel(model_id="qwen3-0.6b-mlx")  # k_shot=0
    messages = m._build_messages(_target_single())
    assert [msg["role"] for msg in messages] == ["system", "user"]
    assert messages[0]["content"] == SYSTEM_PROMPT


def test_build_messages_k_shot_2_interleaves_user_assistant_demos():
    m = LMStudioModel(model_id="qwen3-0.6b-mlx", k_shot=2)
    messages = m._build_messages(_target_single())
    # system + (user, assistant) × 2 demos + final user = 6 messages
    assert [msg["role"] for msg in messages] == [
        "system", "user", "assistant", "user", "assistant", "user",
    ]
    # System prompt unchanged
    assert messages[0]["content"] == SYSTEM_PROMPT
    # Demonstration assistant turns parse as JSON with a "tokens" key
    for demo_idx in (2, 4):
        parsed = json.loads(messages[demo_idx]["content"])
        assert "tokens" in parsed and isinstance(parsed["tokens"], list)
        for entry in parsed["tokens"]:
            assert set(entry.keys()) == {"id", "head", "deprel"}
    # Target sentence is the LAST user message and uses the same formatter
    # as demonstration user messages (no special framing).
    final_user = messages[-1]["content"]
    assert "2 tokens" in final_user
    assert "[1, 2]" in final_user


def test_build_messages_k_shot_4_has_4_demo_pairs():
    m = LMStudioModel(model_id="qwen3-0.6b-mlx", k_shot=4)
    messages = m._build_messages(_target_single())
    roles = [msg["role"] for msg in messages]
    # system + 4 × (user, assistant) + user = 10 messages
    assert len(roles) == 10
    assert roles[0] == "system"
    assert roles[-1] == "user"
    # Interleaving: positions 1,3,5,7 are user demos; 2,4,6,8 are assistant demos
    for i in (1, 3, 5, 7):
        assert roles[i] == "user"
    for i in (2, 4, 6, 8):
        assert roles[i] == "assistant"


def test_call_llm_posts_few_shot_body_with_chat_history():
    """End-to-end: the HTTP body for a k=2 call has the expected interleaved
    chat history, and the final assistant generation still gets the JSON
    schema constraint applied."""
    m = LMStudioModel(model_id="qwen3-0.6b-mlx", k_shot=2)
    payload = {"tokens": [
        {"id": 1, "head": 2, "deprel": "nsubj"},
        {"id": 2, "head": 0, "deprel": "root"},
    ]}
    with patch("requests.post", return_value=_lmstudio_response(payload)) as mock_post:
        m._call_llm(_target_single())

    body = mock_post.call_args.kwargs["json"]
    roles = [msg["role"] for msg in body["messages"]]
    assert roles == ["system", "user", "assistant", "user", "assistant", "user"]
    # Schema still applies to the final completion only
    assert body["response_format"]["type"] == "json_schema"
    assert body["response_format"]["json_schema"]["strict"] is True
```

- [ ] **Step 2: Run the new tests to verify they fail**

Run: `.venv/bin/pytest tests/test_lmstudio_llm.py -v -k "build_messages or format_assistant_response or few_shot_body"`
Expected: All new tests fail — `_build_messages` and `_format_assistant_response` don't exist yet.

- [ ] **Step 3: Add `_format_assistant_response` helper**

In `src/latinbench/models/lmstudio_llm.py`, add this module-level function next to `_format_user_message` (around line 341):

```python
def _format_assistant_response(single: list) -> str:
    """Render a gold-annotated sentence as the JSON assistant response it
    represents — i.e. exactly what we want a few-shot demonstration's
    assistant turn to contain.
    """
    tokens = [
        {"id": t["id"], "head": t["head"], "deprel": t["deprel"]}
        for t in single
    ]
    return json.dumps({"tokens": tokens}, ensure_ascii=False)
```

- [ ] **Step 4: Refactor `_call_llm` to use a `_build_messages` helper**

Replace the existing `_call_llm` (lines 125–147 of `src/latinbench/models/lmstudio_llm.py`) with:

```python
    def _build_messages(self, single: list) -> list[dict]:
        """Build the chat messages list: system + k demo pairs + target user."""
        messages: list[dict] = [
            {"role": "system", "content": SYSTEM_PROMPT}
        ]
        for demo in self._demonstrations:
            demo_single = [t for t in demo if isinstance(t["id"], int)]
            messages.append({
                "role": "user",
                "content": _format_user_message(demo_single),
            })
            messages.append({
                "role": "assistant",
                "content": _format_assistant_response(demo_single),
            })
        messages.append({
            "role": "user",
            "content": _format_user_message(single),
        })
        return messages

    def _call_llm(self, single: list) -> dict:
        body = {
            "model": self.model_id,
            "messages": self._build_messages(single),
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "ud_parse",
                    "strict": True,
                    "schema": SCHEMA,
                },
            },
            "stream": False,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        r = requests.post(f"{self.host}/v1/chat/completions", json=body, timeout=180)
        r.raise_for_status()
        content = r.json()["choices"][0]["message"]["content"]
        return json.loads(content)
```

- [ ] **Step 5: Run all tests to verify pass + no regressions**

Run: `.venv/bin/pytest tests/ -v`
Expected: All tests pass — new few-shot tests, the existing `test_call_llm_posts_correct_body` (k=0 still gives `["system", "user"]`), and every other test in the suite.

- [ ] **Step 6: Commit**

```bash
git add src/latinbench/models/lmstudio_llm.py tests/test_lmstudio_llm.py
git commit -m "$(cat <<'EOF'
LMStudioModel: inject few-shot demos as chat history

When k_shot > 0, the chat-completion call now interleaves
(user demo, assistant gold JSON) turns between the system prompt
and the target sentence's user turn. The user-message formatter
is reused for both demos and the target, so the demonstration's
input shape is bit-for-bit identical to what the model sees at
test time.

LM Studio's response_format JSON-schema constraint applies only
to the final assistant generation; historical assistant turns
are plain JSON strings produced via json.dumps.

Refactor: message construction extracted into _build_messages
so it's testable without HTTP.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 5: Notebook integration + smoke test (manual)

This task involves running a real LLM end-to-end through LM Studio. It produces the headline 0-shot vs 2-shot comparison numbers. The notebook step itself is small; the runtime is hours on the 8B/12B models and requires LM Studio to be live on `localhost:1234`.

**Files:**
- Modify: `notebooks/02_compare_models.ipynb`

- [ ] **Step 1: Confirm LM Studio is running and the three models are downloaded**

Run: `curl -s http://localhost:1234/v1/models | python -m json.tool`
Expected: a JSON response listing `qwen3-0.6b-mlx`, `qwen3-vl-8b-instruct-mlx`, and `google/gemma-3-12b`. If any are missing, download them in LM Studio's Discover tab and reload (see the README's "Trying a local LLM via LM Studio" section).

- [ ] **Step 2: Add a new cell to `notebooks/02_compare_models.ipynb`**

Open the notebook (`source .venv/bin/activate && jupyter lab notebooks/02_compare_models.ipynb`). Add a new code cell at the bottom (or wherever the comparison section ends) with:

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
df_all.query("metric == 'LAS'").pivot_table(
    index="system", columns="split", values="F1"
).round(2)
```

- [ ] **Step 3: Run the smoke test (0.6B first — cheapest)**

In the notebook, run a smaller-scoped cell first to verify wiring before committing to the full run:

```python
from latinbench import Bench
from latinbench.models.lmstudio_llm import LMStudioModel
scores = Bench().run(LMStudioModel("qwen3-0.6b-mlx", k_shot=2))
print(scores)
```

Expected:
- A new directory `predictions/qwen3-0.6b-mlx-2shot/` appears with `scores.json`, `poetry_pred.conllu`, `prose_pred.conllu`, and (briefly during the run) a `*.partial.json` sidecar.
- LAS scores print for both splits and differ from the existing 0-shot scores in `predictions/qwen3-0.6b-mlx/scores.json`.
- The existing `predictions/qwen3-0.6b-mlx/` directory is untouched.

If LM Studio's auto-swap is enabled, the 0.6B will load on first request. The run takes ~10–20 min on M1.

- [ ] **Step 4: Run the full comparison cell**

Run the full comparison cell from Step 2. The 8B and 12B 2-shot runs are slow (hours each). The bench caches per-split, so if it gets interrupted, re-running picks up from the partial sidecar.

Expected: a tidy `(system × split → LAS F1)` table with all 8 model rows (2 trained + 3 zero-shot LLMs + 3 two-shot LLMs).

- [ ] **Step 5: Commit the notebook + new predictions**

```bash
git add notebooks/02_compare_models.ipynb \
        predictions/qwen3-0.6b-mlx-2shot/ \
        predictions/qwen3-vl-8b-instruct-mlx-2shot/ \
        predictions/google-gemma-3-12b-2shot/
git commit -m "$(cat <<'EOF'
notebook: 2-shot LM Studio runs alongside 0-shot baselines

Adds the few-shot comparison cell to 02_compare_models and
commits the 2-shot prediction outputs and scores for the three
LM Studio models. The 0-shot prediction caches are unchanged
(they live under predictions/<slug>/ without the -2shot suffix).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

- [ ] **Step 6: Update `docs/01_findings.md` (optional, recommended)**

After the runs finish, add a "Few-shot prompting" section to `docs/01_findings.md` summarising the 0-shot vs 2-shot deltas per model. Note in the writeup that the demonstrations are hand-curated and disjoint from the EvaLatin test corpus (the methodology controls from the design doc). This is bookkeeping for the research log; commit separately.

---

## Self-review

**Spec coverage** — every section of the design doc maps to a task:

| Spec section                  | Implementing task |
|-------------------------------|-------------------|
| Goal                          | Task 5 (end-to-end run)       |
| Methodology controls          | Task 1 (deterministic sample), Task 2 (disjoint pool, deprel ⊆ gold), Task 4 (identical formatter), Task 4 (chat format) |
| File layout                   | All tasks combined |
| `ExamplePool`                 | Task 1            |
| Bundled examples file         | Task 2            |
| `LMStudioModel` changes — args + slug | Task 3    |
| `LMStudioModel` changes — chat history | Task 4   |
| Cache & naming                | Task 3 (slug logic), Task 5 (verify on disk) |
| Notebook integration          | Task 5            |
| Error handling                | Task 1 (k<0, empty pool, k>len) |
| Testing                       | All tasks (TDD)  |

**Placeholder scan** — none. Every code block contains the literal content the engineer types or pastes. Tests include actual assertions. Commands include actual paths.

**Type / signature consistency**:
- `ExamplePool(path)` — same signature used in Task 1 (`ExamplePool(p)`), Task 3 (`ExamplePool(p)` in the explicit-pool test), and `LMStudioModel.__init__` (default `ExamplePool()`). ✓
- `pool.sample(k, seed=0)` — same call site in Task 1 tests and `LMStudioModel.__init__`. ✓
- `_build_messages(single)` — defined as method in Task 4; tested via `m._build_messages(_target_single())`. ✓
- `_format_assistant_response(single)` — module-level helper in Task 4; imported via `from latinbench.models.lmstudio_llm import _format_assistant_response` in tests. ✓
- `_demonstrations` and `_pool` attribute names match between Task 3 (define) and Task 4 (use). ✓
- Name slug format `{model}-{k}shot[-s{seed}]` consistent between Task 3 implementation and Task 3 tests. ✓

No issues found.
