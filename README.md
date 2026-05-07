# NLP-LLM-SS2026 — `latinbench`

A small bench for evaluating dependency parsers on the EvaLatin 2024 test data, with the goal of beating the published winners.

Comes with two reference models (UDPipe 2 baseline, ÚFAL LatinPipe) and a template stub. Add a new model = write a 30-line Python file.

## Quick start

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e .

# LatinPipe runs in its own venv (Keras + PyTorch backend);
# we reuse the same requirements but in a separate interpreter.
python3 -m venv third_party/latinpipe/.venv
third_party/latinpipe/.venv/bin/pip install -e .

jupyter lab    # or open notebooks/ in VS Code / Cursor
```

Then run, in order:

1. **`notebooks/01_explore_data.ipynb`** — see what's in the test data, distributions, an example tree.
2. **`notebooks/02_compare_models.ipynb`** — runs all registered models, scores them, plots a comparison.

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

Run on the LINDAT REST API (UDPipe 2 + Latin-Perseus, 2026 model) and the released LatinPipe single-model checkpoint:

| split | system | LAS | CLAS |
|---|---|---|---|
| poetry | UDPipe baseline | 61.19 | 59.90 |
| poetry | LatinPipe (1× ckpt) | **72.27** | **71.28** |
| prose | UDPipe baseline | 62.43 | 57.46 |
| prose | LatinPipe (1× ckpt) | **75.06** | **70.90** |

The published LatinPipe paper used a 7-model ensemble — that's the bar to beat (~78 LAS poetry, ~83 LAS prose).
