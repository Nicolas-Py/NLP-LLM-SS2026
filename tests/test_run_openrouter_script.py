"""Unit tests for the standalone runner script (scripts/run_openrouter.py).

The script isn't an installed package, so we load it by file path.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "run_openrouter.py"


def _load_runner():
    spec = importlib.util.spec_from_file_location("run_openrouter_under_test", _SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


runner = _load_runner()


# ---------- _load_dotenv ----------

def test_load_dotenv_plain_key(tmp_path, monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    env = tmp_path / ".env"
    env.write_text("OPENROUTER_API_KEY=sk-plain\n")
    runner._load_dotenv(env)
    import os
    assert os.environ["OPENROUTER_API_KEY"] == "sk-plain"


def test_load_dotenv_export_prefix(tmp_path, monkeypatch):
    """Regression: `export KEY=VALUE` must set KEY, not 'export KEY'."""
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    env = tmp_path / ".env"
    env.write_text("export OPENROUTER_API_KEY=sk-exported\n")
    runner._load_dotenv(env)
    import os
    assert os.environ.get("OPENROUTER_API_KEY") == "sk-exported"
    assert "export OPENROUTER_API_KEY" not in os.environ


def test_load_dotenv_quotes_comments_and_value_with_equals(tmp_path, monkeypatch):
    monkeypatch.delenv("FOO", raising=False)
    monkeypatch.delenv("BAR", raising=False)
    env = tmp_path / ".env"
    env.write_text(
        "# a comment\n"
        "\n"
        'FOO="quoted-value"\n'
        "BAR=a=b=c\n"        # '=' inside the value is preserved
    )
    runner._load_dotenv(env)
    import os
    assert os.environ["FOO"] == "quoted-value"
    assert os.environ["BAR"] == "a=b=c"


def test_load_dotenv_does_not_override_existing(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-from-env")
    env = tmp_path / ".env"
    env.write_text("OPENROUTER_API_KEY=sk-from-file\n")
    runner._load_dotenv(env)
    import os
    assert os.environ["OPENROUTER_API_KEY"] == "sk-from-env"  # env wins


def test_load_dotenv_missing_file_is_noop(tmp_path):
    runner._load_dotenv(tmp_path / "does-not-exist.env")  # must not raise


# ---------- _slice_conllu ----------

_THREE_SENTS = (
    "# sent_id = s1\n"
    "1\ta\ta\tX\t_\t_\t_\t_\t_\t_\n"
    "2\tb\tb\tX\t_\t_\t_\t_\t_\t_\n"
    "\n"
    "# sent_id = s2\n"
    "2-3\tcd\t_\t_\t_\t_\t_\t_\t_\t_\n"   # multi-word token line
    "2\tc\tc\tX\t_\t_\t_\t_\t_\t_\n"
    "3\td\td\tX\t_\t_\t_\t_\t_\t_\n"
    "\n"
    "# sent_id = s3\n"
    "1\te\te\tX\t_\t_\t_\t_\t_\t_\n"
)


def test_slice_conllu_takes_first_n_blocks(tmp_path):
    import conllu
    src = tmp_path / "src.conllu"
    src.write_text(_THREE_SENTS)
    dst = tmp_path / "dst.conllu"
    n = runner._slice_conllu(src, dst, 2)
    assert n == 2
    parsed = conllu.parse(dst.read_text())
    assert [s.metadata["sent_id"] for s in parsed] == ["s1", "s2"]
    # multi-word token block survived intact in the second sentence
    assert any(not isinstance(t["id"], int) for t in parsed[1])


def test_slice_conllu_clamps_to_available(tmp_path):
    src = tmp_path / "src.conllu"
    src.write_text(_THREE_SENTS)
    dst = tmp_path / "dst.conllu"
    assert runner._slice_conllu(src, dst, 99) == 3


def test_slice_test_and_gold_stay_aligned(tmp_path):
    """The same N applied to test and gold yields matching sent_ids — required
    by the scorer."""
    import conllu
    test = tmp_path / "t.conllu"
    gold = tmp_path / "g.conllu"
    test.write_text(_THREE_SENTS)
    gold.write_text(_THREE_SENTS)
    ts, gs = tmp_path / "ts.conllu", tmp_path / "gs.conllu"
    runner._slice_conllu(test, ts, 2)
    runner._slice_conllu(gold, gs, 2)
    tids = [s.metadata["sent_id"] for s in conllu.parse(ts.read_text())]
    gids = [s.metadata["sent_id"] for s in conllu.parse(gs.read_text())]
    assert tids == gids
