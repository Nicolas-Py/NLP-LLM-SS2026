"""Wrapper around the official CoNLL-18 UD scorer."""
from __future__ import annotations
import re
import subprocess
import sys
from pathlib import Path

from .data import SCORER_PATH


_METRIC_RE = re.compile(r"^(\w+)\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)")


def score(gold_path: Path, pred_path: Path) -> dict[str, dict[str, float]]:
    """Run the official scorer and return {metric: {'P','R','F1'}}.

    Raises CalledProcessError if the scorer fails.
    """
    out = subprocess.run(
        [sys.executable, str(SCORER_PATH), "-v", str(gold_path), str(pred_path)],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    return _parse(out)


def _parse(s: str) -> dict[str, dict[str, float]]:
    rows: dict[str, dict[str, float]] = {}
    for line in s.splitlines():
        m = _METRIC_RE.match(line)
        if m:
            metric, p, r, f1 = m.groups()
            rows[metric] = {"P": float(p), "R": float(r), "F1": float(f1)}
    return rows
