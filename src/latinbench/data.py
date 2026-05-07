"""Canonical paths into the repo's data/, checkpoints/, predictions/ folders.

Resolved relative to the package install location, so notebooks work regardless
of which directory the kernel was started from.
"""
from __future__ import annotations
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = REPO_ROOT / "data"
TEST_DIR = DATA_DIR / "EvaLatin_2024_Syntactic_Parsing_test_data"
GOLD_DIR = DATA_DIR / "EvaLatin_2024_Syntactic_Parsing_test_data_gold"

CHECKPOINTS_DIR = REPO_ROOT / "checkpoints"
PREDICTIONS_DIR = REPO_ROOT / "predictions"

THIRD_PARTY = REPO_ROOT / "third_party"
SCORER_PATH = THIRD_PARTY / "scorer" / "conll18_ud_eval.py"
LATINPIPE_DIR = THIRD_PARTY / "latinpipe"


def test_path(split: str) -> Path:
    """Path to the test conllu file for `split` ('poetry' or 'prose')."""
    name = {
        "poetry": "EvaLatin_2024_poetry_test_data.conllu",
        "prose": "EvaLatin_2024_prose-test-data.conllu",
    }[split]
    return TEST_DIR / name


def gold_path(split: str) -> Path:
    name = {
        "poetry": "EvaLatin_2024_poetry_gold.conllu",
        "prose": "EvaLatin_2024_prose_gold.conllu",
    }[split]
    return GOLD_DIR / name
