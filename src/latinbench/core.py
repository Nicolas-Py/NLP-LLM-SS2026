"""Model ABC and Bench orchestrator.

A Model knows how to read a test CoNLL-U and write predictions.
A Bench runs one or more Models on the canonical splits, scores the predictions,
and produces a tidy comparison DataFrame.
"""
from __future__ import annotations

import json
from abc import ABC, abstractmethod
from pathlib import Path

import pandas as pd

from .data import PREDICTIONS_DIR, gold_path, test_path
from .score import score as _score


class Model(ABC):
    """Subclass this and implement `predict`."""

    name: str  # short slug; used as folder under predictions/

    @abstractmethod
    def predict(self, test_path: Path, out_path: Path) -> None:
        """Read test conllu, write predicted conllu to out_path."""


class Bench:
    """Runs models, scores them, returns/plots comparisons."""

    def __init__(self, splits: tuple[str, ...] = ("poetry", "prose")) -> None:
        self.splits = splits

    def run(self, model: Model, force: bool = False) -> dict:
        """Run model on all splits, score each, cache results.

        Returns: {split: {metric: {P, R, F1}}}.
        Cache file `predictions/<model.name>/scores.json` is updated after
        every split, so a crash on later splits doesn't lose earlier scores.
        Pass `force=True` to ignore the cache and rebuild from scratch.
        """
        out_dir = PREDICTIONS_DIR / model.name
        out_dir.mkdir(parents=True, exist_ok=True)
        cache = out_dir / "scores.json"

        scores: dict = {}
        if cache.exists() and not force:
            scores = json.loads(cache.read_text())
            if all(s in scores for s in self.splits):
                return scores

        for split in self.splits:
            if split in scores and not force:
                continue
            pred = out_dir / f"{split}_pred.conllu"
            if force or not pred.exists():
                print(f"[{model.name}] predicting {split}…")
                model.predict(test_path(split), pred)
            print(f"[{model.name}] scoring {split}…")
            scores[split] = _score(gold_path(split), pred)
            cache.write_text(json.dumps(scores, indent=2))

        return scores

    def compare(
        self,
        models: list[Model],
        force: bool = False,
        metrics: tuple[str, ...] = ("UAS", "LAS", "CLAS"),
    ) -> pd.DataFrame:
        """Run each model, return a tidy DataFrame.

        Columns: system, split, metric, P, R, F1.
        """
        rows = []
        for m in models:
            scores = self.run(m, force=force)
            for split, by_metric in scores.items():
                for metric in metrics:
                    if metric in by_metric:
                        rows.append({
                            "system": m.name,
                            "split": split,
                            "metric": metric,
                            **by_metric[metric],
                        })
        return pd.DataFrame(rows)

    def plot(self, df: pd.DataFrame, metric: str = "LAS") -> None:
        """Grouped bar chart of `metric` F1 by (split, system)."""
        import matplotlib.pyplot as plt

        sub = df[df["metric"] == metric]
        pivot = sub.pivot_table(index="split", columns="system", values="F1")
        ax = pivot.plot(kind="bar", figsize=(8, 4))
        ax.set_ylabel(f"{metric} F1")
        ax.set_title(f"{metric} by split / system")
        ax.set_ylim(0, 100)
        plt.xticks(rotation=0)
        plt.tight_layout()
        plt.show()
