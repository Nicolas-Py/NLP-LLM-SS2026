# NLP-LLM-SS2026 — `latinbench`

A small bench for evaluating dependency parsers on the EvaLatin 2024 test data, with the goal of beating the published winners.

Comes with two reference models (UDPipe 2, ÚFAL LatinPipe) and a template stub. Add a new model = write a 30-line Python file.

## Setup (one time)

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

Dependencies live in `pyproject.toml` — there's no `requirements.txt` (it would just duplicate). `pip install -e .` is the modern equivalent.

## Running the existing models

The two reference models (`udpipe`, `latinpipe`) come pre-registered. Pick whichever interface you prefer:

### PyCharm (commands in the built-in Terminal)

Open PyCharm's Terminal tab (**View → Tool Windows → Terminal**, or `⌥F12`) and paste:

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

To make PyCharm itself (its notebook runner, autocomplete, "Run" buttons) use this venv, point its project interpreter at `<repo>/.venv/bin/python` once.

### Jupyter Lab / Notebook

```bash
source .venv/bin/activate
jupyter lab    # or jupyter notebook
```

Open `notebooks/02_compare_models.ipynb` → Run All. Same flow as PyCharm; just a browser instead of the IDE.

### VS Code / Cursor

Open the `NLP-LLM-SS2026/` folder, pick the kernel `<repo>/.venv/bin/python` in the top-right of the notebook, then Run All.

### One-liner (no notebook)

```bash
.venv/bin/python -c "
from latinbench import Bench, MODELS
print(Bench().compare([MODELS['udpipe'], MODELS['latinpipe']]).to_string(index=False))
"
```

## Forcing a fresh run

Results cache at `predictions/<model_name>/scores.json`. To re-run:

```bash
# Re-run everything from scratch
rm -rf predictions/

# Re-run just one model
rm -rf predictions/latinpipe

# Or pass force=True from Python:
.venv/bin/python -c "from latinbench import Bench, MODELS; Bench().run(MODELS['latinpipe'], force=True)"
```

LatinPipe inference takes ~1–2 min per split on M1 CPU; UDPipe takes ~5–10 sec (REST API).

## Common gotchas

- **`ModuleNotFoundError: No module named 'latinbench'`** — wrong Python interpreter. In PyCharm, fix via Settings → Interpreter. From the shell, `which python` should point at `<repo>/.venv/bin/python`.
- **LatinPipe fails with TensorFlow import error** — the subprocess venv exists but `KERAS_BACKEND=torch` wasn't set. The bench sets it automatically; if you're invoking `latinpipe_evalatin24.py` by hand, prefix with `KERAS_BACKEND=torch`.
- **LatinPipe checkpoint not found** — verify `checkpoints/latinpipe-evalatin24-240520/model.weights.h5` exists (≈663 MB). Re-download from the LINDAT link above.

## Adding a new model

1. Open `src/latinbench/models/template.py`.
2. Subclass `Model`, set `name`, implement `predict(test_path, out_path)`.
3. Import it in `notebooks/02_compare_models.ipynb` and add it to the `bench.compare([...])` call.

That's it. The bench handles writing predictions, calling the official scorer, parsing results, and plotting.

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
predictions/            # gitignored; per-model output
notebooks/              # 01_explore_data, 02_compare_models
docs/00_task_explained.md
```

## What's the task?

See [docs/00_task_explained.md](docs/00_task_explained.md) for a plain-English walkthrough of dependency parsing on Latin.

## Reference scores

Run on the LINDAT REST API (UDPipe 2, model `latin-perseus-ud-2.17-251125`) and the released LatinPipe single-model checkpoint (`latinpipe-evalatin24-240520`). Both are pinned in code so these numbers stay reproducible.

| split | system | LAS | CLAS |
|---|---|---|---|
| poetry | UDPipe | 61.19 | 59.90 |
| poetry | LatinPipe (1× ckpt) | **72.27** | **71.28** |
| prose | UDPipe | 62.43 | 57.46 |
| prose | LatinPipe (1× ckpt) | **75.06** | **70.90** |

The published LatinPipe paper used a 7-model ensemble — that's the bar to beat (~78 LAS poetry, ~83 LAS prose).

### Trying a different UDPipe version

```python
from latinbench import Bench
from latinbench.models.udpipe import UdpipeModel, list_perseus_models

print(list_perseus_models())                              # all available LINDAT ids
Bench().run(UdpipeModel('latin-perseus-ud-2.6-200830'))   # try an older one
Bench().run(UdpipeModel(model_id='latest'))               # auto-pick newest
```

Each id gets its own `predictions/<id>/` cache dir, so swapping versions doesn't clobber prior results.
