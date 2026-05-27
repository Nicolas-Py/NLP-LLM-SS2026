from __future__ import annotations

import pytest
import conllu

from latinbench.few_shot import DEFAULT_EXAMPLES_PATH, ExamplePool
from latinbench.models.lmstudio_llm import DEPREL_LABELS


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
