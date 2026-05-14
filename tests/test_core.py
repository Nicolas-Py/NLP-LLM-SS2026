from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from latinbench.core import Bench, Model


class _StubModel(Model):
    """Records which splits .predict() was called for, writes a stub pred file."""

    def __init__(self, name: str = "stub"):
        self.name = name
        self.calls: list[str] = []

    def predict(self, test_path: Path, out_path: Path) -> None:
        self.calls.append(out_path.name)
        out_path.write_text("stub\n")


def _fake_scores(_gold, _pred):
    return {"LAS": {"P": 50.0, "R": 50.0, "F1": 50.0}}


def test_run_writes_cache_after_each_split(tmp_path, monkeypatch):
    monkeypatch.setattr("latinbench.core.PREDICTIONS_DIR", tmp_path)
    monkeypatch.setattr("latinbench.core.test_path", lambda s: tmp_path / f"{s}_test.conllu")
    monkeypatch.setattr("latinbench.core.gold_path", lambda s: tmp_path / f"{s}_gold.conllu")
    # Touch the input files so paths exist
    for s in ("poetry", "prose"):
        (tmp_path / f"{s}_test.conllu").write_text("\n")
        (tmp_path / f"{s}_gold.conllu").write_text("\n")

    model = _StubModel()
    written: list[dict] = []

    real_write = Path.write_text

    def spy_write(self, data, *args, **kwargs):
        if self.name == "scores.json":
            written.append(json.loads(data))
        return real_write(self, data, *args, **kwargs)

    with patch("latinbench.core._score", side_effect=_fake_scores), \
         patch.object(Path, "write_text", spy_write):
        Bench().run(model)

    # scores.json was written twice — once after poetry, once after prose
    assert len(written) == 2
    assert set(written[0].keys()) == {"poetry"}
    assert set(written[1].keys()) == {"poetry", "prose"}


def test_run_resumes_when_one_split_already_cached(tmp_path, monkeypatch):
    """If scores.json has only poetry, prose runs but poetry is skipped."""
    monkeypatch.setattr("latinbench.core.PREDICTIONS_DIR", tmp_path)
    monkeypatch.setattr("latinbench.core.test_path", lambda s: tmp_path / f"{s}_test.conllu")
    monkeypatch.setattr("latinbench.core.gold_path", lambda s: tmp_path / f"{s}_gold.conllu")
    for s in ("poetry", "prose"):
        (tmp_path / f"{s}_test.conllu").write_text("\n")
        (tmp_path / f"{s}_gold.conllu").write_text("\n")

    model = _StubModel()
    out_dir = tmp_path / model.name
    out_dir.mkdir()
    # Pre-populate poetry's scores + prediction file as if last run was interrupted
    # after poetry but before prose.
    (out_dir / "scores.json").write_text(json.dumps(
        {"poetry": {"LAS": {"P": 99.0, "R": 99.0, "F1": 99.0}}}
    ))
    (out_dir / "poetry_pred.conllu").write_text("stub\n")

    with patch("latinbench.core._score", side_effect=_fake_scores):
        scores = Bench().run(model)

    # Only prose was predicted this run
    assert model.calls == ["prose_pred.conllu"]
    # Poetry's cached score (99.0) survives; prose got the new fake (50.0)
    assert scores["poetry"]["LAS"]["F1"] == 99.0
    assert scores["prose"]["LAS"]["F1"] == 50.0


def test_run_fully_cached_returns_immediately(tmp_path, monkeypatch):
    monkeypatch.setattr("latinbench.core.PREDICTIONS_DIR", tmp_path)
    monkeypatch.setattr("latinbench.core.test_path", lambda s: tmp_path / f"{s}_test.conllu")
    monkeypatch.setattr("latinbench.core.gold_path", lambda s: tmp_path / f"{s}_gold.conllu")

    model = _StubModel()
    out_dir = tmp_path / model.name
    out_dir.mkdir()
    (out_dir / "scores.json").write_text(json.dumps({
        "poetry": {"LAS": {"P": 1.0, "R": 2.0, "F1": 3.0}},
        "prose": {"LAS": {"P": 4.0, "R": 5.0, "F1": 6.0}},
    }))

    with patch("latinbench.core._score") as score_fn:
        scores = Bench().run(model)
        assert not score_fn.called
    assert model.calls == []
    assert scores["poetry"]["LAS"]["F1"] == 3.0
    assert scores["prose"]["LAS"]["F1"] == 6.0
