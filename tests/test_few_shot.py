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
