# NLP-LLM-SS2026 — `latinbench`

A small bench for evaluating dependency parsers on the EvaLatin 2024 test data,
with the goal of beating the published winners.

Comes with two reference models (UDPipe 2, ÚFAL LatinPipe), a local-LLM
template via LM Studio, and a stub for your own model. Add a new model =
write a 30-line Python file.

> Two ways to use this repo:
> - **Just want to read what's been tried?** See [Explore the repo](#explore-the-repo) — nothing to install.
> - **Want to run the bench or add a model?** See [Get started](#get-started).

## Explore the repo

The repo ships with the actual parse outputs of every system we've run, so you
can inspect results without installing anything. Start here if you're just
reading along.

### Results so far

LAS = labeled attachment score, CLAS = content-word LAS. Gold tokenization, so
the system contribution is purely the parse. See
[docs/01_findings.md](docs/01_findings.md) for the full breakdown (UAS,
fallback rates, per-finding analysis).

| split | system | LAS | CLAS |
|---|---|---:|---:|
| poetry | LatinPipe (1× checkpoint)                       | **72.27** | **71.28** |
| poetry | UDPipe 2 (`latin-perseus-ud-2.17`)              | 61.19 | 59.90 |
| poetry | qwen3-vl-8b-instruct-mlx (LM Studio, 2-shot Perseus, packed) | 18.77 | 18.31 |
| poetry | qwen3-vl-8b-instruct-mlx (LM Studio, 2-shot)    | 18.41 | 17.43 |
| poetry | qwen3-vl-8b-instruct-mlx (LM Studio, 0-shot)    | 18.21 | 17.42 |
| poetry | qwen3-0.6b-mlx (LM Studio, 0-shot)              |  2.67 |  2.72 |
| poetry | qwen3-0.6b-mlx (LM Studio, 2-shot Perseus)      |  2.31 |  2.42 |
| poetry | qwen3-0.6b-mlx (LM Studio, 2-shot)              |  2.06 |  2.23 |
| prose  | LatinPipe (1× checkpoint)                       | **75.06** | **70.90** |
| prose  | UDPipe 2 (`latin-perseus-ud-2.17`)              | 62.43 | 57.46 |
| prose  | qwen3-vl-8b-instruct-mlx (LM Studio, 2-shot)    | 20.16 | 16.68 |
| prose  | qwen3-vl-8b-instruct-mlx (LM Studio, 2-shot Perseus, packed) | 18.98 | 16.26 |
| prose  | qwen3-vl-8b-instruct-mlx (LM Studio, 0-shot)    | 17.80 | 14.53 |
| prose  | qwen3-0.6b-mlx (LM Studio, 2-shot Perseus)      |  1.99 |  2.08 |
| prose  | qwen3-0.6b-mlx (LM Studio, 0-shot)              |  1.62 |  1.51 |
| prose  | qwen3-0.6b-mlx (LM Studio, 2-shot)              |  1.32 |  1.52 |

The bar to beat is the published LatinPipe 7-model ensemble: ~78 LAS poetry,
~83 LAS prose. The 1× checkpoint above is one model from that ensemble and is
what we ship as the local reference.

### Where to look

- [docs/01_findings.md](docs/01_findings.md) — distilled research log: what
  we tried, what worked, what's next.
- [docs/00_task_explained.md](docs/00_task_explained.md) — plain-English
  walkthrough of dependency parsing on Latin.
- [notebooks/02_compare_models.ipynb](notebooks/02_compare_models.ipynb) — the
  bench end-to-end: running the comparison, the LLM error analysis, the
  0.6B → 8B scale-up. Renders directly on GitHub.
- [predictions/](predictions/) — each system's actual parse output:
  - `predictions/<system>/scores.json` — full CoNLL-18 scorer output
    (UAS / LAS / CLAS / MLAS / BLEX, plus token/sent/word F1).
  - `predictions/<system>/{poetry,prose}_pred.conllu` — the predicted UD
    trees, sentence-by-sentence. Diff against `data/EvaLatin-2024-test/`
    gold to see where each system goes wrong.
- [src/latinbench/](src/latinbench/) — the small Python package: `Model` ABC
  + `Bench` orchestrator (`core.py`), scorer wrapper (`score.py`), one
  file per model in `models/`.

## Get started

Skip this section if you only want to read results. Everything below assumes
you'll be running models locally.

### Setup (one time)

```bash
git clone https://github.com/Nicolas-Py/NLP-LLM-SS2026
cd NLP-LLM-SS2026

# Main env (notebook kernel + Python API)
python3 -m venv .venv && source .venv/bin/activate
pip install -e .

# Subprocess env for LatinPipe (Keras + PyTorch backend)
python3 -m venv third_party/latinpipe/.venv
third_party/latinpipe/.venv/bin/pip install -e .

# Download the LatinPipe checkpoint (≈700 MB) from
# https://hdl.handle.net/11234/1-5671 and extract its contents
# (model.weights.h5, mappings.pkl, la_evalatin24.tokenizer, options.json)
# into checkpoints/latinpipe-evalatin24-240520/
```

Dependencies live in `pyproject.toml` — there's no `requirements.txt` (it
would just duplicate). `pip install -e .` is the modern equivalent.

### Running the existing models

The two reference models (`udpipe`, `latinpipe`) come pre-registered. Pick
whichever interface you prefer:

#### PyCharm (commands in the built-in Terminal)

Open PyCharm's Terminal tab (**View → Tool Windows → Terminal**, or `⌥F12`)
and paste:

```bash
# One-time setup (skip if already done)
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
python3 -m venv third_party/latinpipe/.venv
third_party/latinpipe/.venv/bin/pip install -e .
# Then drop the LatinPipe checkpoint files into
# checkpoints/latinpipe-evalatin24-240520/

# Every PyCharm session — activate the venv first
source .venv/bin/activate

# Smoke-test both reference models
python -c "
from latinbench import Bench, MODELS
print(Bench().compare([MODELS['udpipe'], MODELS['latinpipe']]).to_string(index=False))
"
```

To make PyCharm itself (its notebook runner, autocomplete, "Run" buttons) use
this venv, point its project interpreter at `<repo>/.venv/bin/python` once.

#### Jupyter Lab / Notebook

```bash
source .venv/bin/activate
jupyter lab    # or jupyter notebook
```

Open `notebooks/02_compare_models.ipynb` → Run All. Same flow as PyCharm;
just a browser instead of the IDE.

#### VS Code / Cursor

Open the `NLP-LLM-SS2026/` folder, pick the kernel `<repo>/.venv/bin/python`
in the top-right of the notebook, then Run All.

#### One-liner (no notebook)

```bash
.venv/bin/python -c "
from latinbench import Bench, MODELS
print(Bench().compare([MODELS['udpipe'], MODELS['latinpipe']]).to_string(index=False))
"
```

### Forcing a fresh run

Results cache at `predictions/<model_name>/scores.json`. The committed
`predictions/` tree means `Bench().run(...)` will short-circuit to cached
scores on a fresh checkout. To genuinely re-run:

```bash
# Re-run everything from scratch
rm -rf predictions/*/scores.json

# Re-run just one model
rm -rf predictions/latinpipe

# Or pass force=True from Python:
.venv/bin/python -c "from latinbench import Bench, MODELS; Bench().run(MODELS['latinpipe'], force=True)"
```

LatinPipe inference takes ~1–2 min per split on M1 CPU; UDPipe takes
~5–10 sec (REST API).

### Common gotchas

- **`ModuleNotFoundError: No module named 'latinbench'`** — wrong Python
  interpreter. In PyCharm, fix via Settings → Interpreter. From the shell,
  `which python` should point at `<repo>/.venv/bin/python`.
- **LatinPipe fails with TensorFlow import error** — the subprocess venv
  exists but `KERAS_BACKEND=torch` wasn't set. The bench sets it
  automatically; if you're invoking `latinpipe_evalatin24.py` by hand,
  prefix with `KERAS_BACKEND=torch`.
- **LatinPipe checkpoint not found** — verify
  `checkpoints/latinpipe-evalatin24-240520/model.weights.h5` exists
  (≈663 MB). Re-download from the LINDAT link above.

### Adding a new model

1. Open `src/latinbench/models/template.py`.
2. Subclass `Model`, set `name`, implement `predict(test_path, out_path)`.
3. Import it in `notebooks/02_compare_models.ipynb` and add it to the
   `bench.compare([...])` call.

That's it. The bench handles writing predictions, calling the official
scorer, parsing results, and plotting.

## Layout

```
src/latinbench/         # the Python package
├── core.py             # Model ABC + Bench orchestrator
├── score.py            # scorer subprocess wrapper
├── data.py             # canonical paths
└── models/             # one file per model
data/                   # EvaLatin 2024 test + gold (committed)
third_party/
├── scorer/             # CoNLL-18 official scorer
└── latinpipe/          # vendored ÚFAL LatinPipe (no .git, no venv)
checkpoints/            # gitignored; LatinPipe weights live here
predictions/            # tracked; one subdir per system (see Explore above)
notebooks/              # 01_explore_data, 02_compare_models
docs/
├── 00_task_explained.md   # what dependency parsing is, in plain English
└── 01_findings.md         # research log: what's been tried, what worked
```

## Reference details

### Trying a different UDPipe version

```python
from latinbench import Bench
from latinbench.models.udpipe import UdpipeModel, list_perseus_models

print(list_perseus_models())                              # all available LINDAT ids
Bench().run(UdpipeModel('latin-perseus-ud-2.6-200830'))   # try an older one
Bench().run(UdpipeModel(model_id='latest'))               # auto-pick newest
```

Each id gets its own `predictions/<id>/` cache dir, so swapping versions
doesn't clobber prior results.

### Trying a local LLM via LM Studio

Three LM Studio entries are registered (one 0.6B baseline, one 8B Qwen3-VL,
one Gemma-3-12B). They share a single running LM Studio server; only one
model is hot in memory at a time but LM Studio auto-swaps on request, so
the same workflow handles all three.

One-time setup:

1. Install [LM Studio](https://lmstudio.ai/).
2. **Discover** tab → search "Qwen3 0.6B MLX" → download
   `lmstudio-community/Qwen3-0.6B-MLX-4bit` (~400 MB). Repeat for any other
   model you want to bench.
3. **Developer** tab → load the model → **Start Server** (defaults to
   port 1234).
4. Recommended server settings: max parallel requests **8**, Flash Attention
   **on**, KV cache **q8_0**.
5. Sanity check: `curl http://localhost:1234/v1/models` should list the
   loaded model.

Run any registered LM Studio model:

```python
from latinbench import Bench, MODELS
Bench().run(MODELS["qwen3-lmstudio"])          # 0.6B baseline
Bench().run(MODELS["qwen3-vl-8b-lmstudio"])    # 8B
Bench().run(MODELS["gemma-3-12b-lmstudio"])    # 12B, different family
```

Swap to any other model loaded in LM Studio by constructing directly (pass
the exact id LM Studio reports for `GET /v1/models`):

```python
from latinbench import Bench
from latinbench.models.lmstudio_llm import LMStudioModel
Bench().run(LMStudioModel("qwen/qwen3-4b"))               # different size
Bench().run(LMStudioModel("meta-llama/llama-3.2-3b"))     # different family
```

Each `model_id` gets its own `predictions/<slug>/` cache dir (`:` and `/`
are sanitized to `-`). Expect ~5 s per short sentence on a 0.6B model — full
splits take many minutes; the bench caches per-model so re-runs are instant.
A `<pred>.partial.json` sidecar updates as each sentence completes so a
crash mid-predict resumes from where it left off.

#### Few-shot prompting

Pass `k_shot=N` to inject `N` hand-curated Latin demonstration sentences
as chat history before the target sentence (default `k_shot=0` is the
zero-shot path above). Demonstrations are sampled deterministically from
the bundled
[`src/latinbench/few_shot_examples.conllu`](src/latinbench/few_shot_examples.conllu)
pool — disjoint from the EvaLatin test corpus by construction.

```python
from latinbench import Bench
from latinbench.models.lmstudio_llm import LMStudioModel
Bench().run(LMStudioModel("qwen3-vl-8b-instruct-mlx", k_shot=2))
```

The cache slug auto-suffixes with `-{k}shot` (e.g.
`predictions/qwen3-vl-8b-instruct-mlx-2shot/`), so 0-shot and k-shot
results coexist and can be compared in the same `Bench.compare(...)`
DataFrame. Pass `example_pool=ExamplePool(my_conllu)` to swap in your
own pool; `shot_seed=N` to pick a different sample from the same pool.

A second bundled pool drawn from **UD_Latin-Perseus training data**
(punctuation-stripped to match the punct-free EvaLatin format) ships alongside
the hand-curated one:

```python
from latinbench.few_shot import DEFAULT_EXAMPLES_PATH, ExamplePool
perseus = ExamplePool(DEFAULT_EXAMPLES_PATH.parent / "few_shot_examples_perseus.conllu")
Bench().run(LMStudioModel("qwen3-vl-8b-instruct-mlx", k_shot=2, example_pool=perseus))
# caches to predictions/qwen3-vl-8b-instruct-mlx-2shot-perseus/
```

The pool's `tag` (`perseus`, from the filename) is appended to the slug so it
sits beside the hand-curated `-2shot` run instead of overwriting it.

For the single-prompt ("packed") few-shot style — all examples inlined in one
user turn instead of multi-turn chat — pass `pack_demos=True`; the slug then
ends in `-packed` (e.g. `…-2shot-perseus-packed`).

Methodology, results, and the comparisons (0-shot vs 2-shot, chat vs packed)
live in [docs/01_findings.md](docs/01_findings.md) key findings #7–#9.
