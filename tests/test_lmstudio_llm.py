from __future__ import annotations

import json
from unittest.mock import patch, MagicMock

import conllu
import requests

from latinbench.models.lmstudio_llm import (
    LMStudioModel,
    SCHEMA,
    SYSTEM_PROMPT,
    _collect_deprels_from_gold,
    _format_sentence,
    _is_valid_tree,
    _repair_tree,
    _right_branching_default,
)


class _FakeModel(LMStudioModel):
    """LMStudioModel with `_call_llm` swapped for a fixture."""
    def __init__(self, fake):
        super().__init__()
        self._fake = fake

    def _call_llm(self, single):
        if isinstance(self._fake, Exception):
            raise self._fake
        return self._fake


def _toks(ids):
    """Build a minimal TokenList-like list of dicts with the given ids."""
    return [{"id": i, "form": f"w{i}", "lemma": "_", "upos": "X", "feats": None}
            for i in ids]


# ---------- _right_branching_default ----------

def test_right_branching_last_token_is_root():
    single = _toks([1, 2, 3])
    assert _right_branching_default(single, 2) == (0, "root")


def test_right_branching_non_last_attaches_to_next():
    single = _toks([1, 2, 3])
    assert _right_branching_default(single, 0) == (2, "dep")
    assert _right_branching_default(single, 1) == (3, "dep")


def test_right_branching_handles_non_contiguous_ids():
    # ids 1, 3, 7 — non-contiguous because of MWT skipping etc.
    single = _toks([1, 3, 7])
    assert _right_branching_default(single, 0) == (3, "dep")
    assert _right_branching_default(single, 1) == (7, "dep")
    assert _right_branching_default(single, 2) == (0, "root")


# ---------- _format_sentence ----------

def test_format_sentence_basic_columns():
    single = [
        {"id": 1, "form": "Marcus", "lemma": "marcus", "upos": "PROPN",
         "feats": {"Case": "Nom", "Number": "Sing"}},
        {"id": 2, "form": "amat", "lemma": "amo", "upos": "VERB",
         "feats": {"Mood": "Ind"}},
    ]
    out = _format_sentence(single)
    lines = out.split("\n")
    assert len(lines) == 2
    assert lines[0] == "1\tMarcus\tmarcus\tPROPN\tCase=Nom|Number=Sing"
    assert lines[1] == "2\tamat\tamo\tVERB\tMood=Ind"


def test_format_sentence_empty_feats_renders_underscore():
    single = [{"id": 1, "form": "a", "lemma": "_", "upos": "X", "feats": None}]
    assert _format_sentence(single).endswith("\t_")


# ---------- _collect_deprels_from_gold ----------

def test_collect_deprels_from_tmp_gold(tmp_path):
    p = tmp_path / "g.conllu"
    p.write_text(
        "# sent_id = 1\n"
        "1\ta\ta\tX\t_\t_\t2\tnsubj\t_\t_\n"
        "2\tb\tb\tX\t_\t_\t0\troot\t_\t_\n"
        "\n"
        "# sent_id = 2\n"
        "1\tc\tc\tX\t_\t_\t0\troot\t_\t_\n"
        "2\td\td\tX\t_\t_\t1\tobj\t_\t_\n"
    )
    labels = _collect_deprels_from_gold([p])
    assert labels == sorted(["nsubj", "root", "obj"])


def test_collect_deprels_from_real_gold_files_contains_common_relations():
    labels = _collect_deprels_from_gold()
    for r in ("root", "nsubj", "obj", "det", "amod"):
        assert r in labels


# ---------- model naming ----------

def test_name_slug_replaces_colon():
    assert LMStudioModel(model_id="vendor:foo").name == "vendor-foo"


def test_name_slug_default_model_is_qwen3():
    # Default LM Studio model id is qwen3-0.6b-mlx (no : or /)
    assert LMStudioModel().name == "qwen3-0.6b-mlx"


def test_name_slug_replaces_slash_too():
    assert LMStudioModel(model_id="hf.co/foo:Q4_K_M").name == "hf.co-foo-Q4_K_M"


def test_host_is_trimmed_of_trailing_slash():
    m = LMStudioModel(host="http://localhost:1234/")
    assert m.host == "http://localhost:1234"


# ---------- few-shot constructor / name slug ----------

def test_name_slug_zero_shot_default_unchanged():
    """Existing behaviour: k_shot defaults to 0, no slug suffix."""
    assert LMStudioModel(model_id="qwen3-0.6b-mlx").name == "qwen3-0.6b-mlx"


def test_name_slug_k_shot_2_appends_2shot():
    m = LMStudioModel(model_id="qwen3-0.6b-mlx", k_shot=2)
    assert m.name == "qwen3-0.6b-mlx-2shot"


def test_name_slug_k_shot_4_appends_4shot():
    m = LMStudioModel(model_id="qwen3-0.6b-mlx", k_shot=4)
    assert m.name == "qwen3-0.6b-mlx-4shot"


def test_name_slug_non_default_seed_appends_seed_suffix():
    m = LMStudioModel(model_id="qwen3-0.6b-mlx", k_shot=2, shot_seed=7)
    assert m.name == "qwen3-0.6b-mlx-2shot-s7"


def test_name_slug_default_seed_does_not_add_seed_suffix():
    """Common case stays clean: shot_seed=0 → no -s0 suffix."""
    m = LMStudioModel(model_id="qwen3-0.6b-mlx", k_shot=2, shot_seed=0)
    assert m.name == "qwen3-0.6b-mlx-2shot"


def test_k_shot_zero_does_not_load_pool():
    """Zero-shot construction must not touch the example file at all."""
    m = LMStudioModel(model_id="qwen3-0.6b-mlx", k_shot=0)
    assert m._demonstrations == []
    assert m._pool is None


def test_k_shot_positive_uses_default_pool():
    m = LMStudioModel(model_id="qwen3-0.6b-mlx", k_shot=2)
    assert m._pool is not None
    assert len(m._demonstrations) == 2


def test_k_shot_accepts_explicit_pool(tmp_path):
    p = tmp_path / "custom.conllu"
    p.write_text(
        "# sent_id = c-1\n"
        "1\tx\tx\tX\t_\t_\t0\troot\t_\t_\n"
        "\n"
        "# sent_id = c-2\n"
        "1\ty\ty\tX\t_\t_\t0\troot\t_\t_\n"
    )
    from latinbench.few_shot import ExamplePool
    pool = ExamplePool(p)
    m = LMStudioModel(model_id="qwen3-0.6b-mlx", k_shot=2, example_pool=pool)
    assert m._pool is pool
    assert len(m._demonstrations) == 2
    ids = {s.metadata["sent_id"] for s in m._demonstrations}
    assert ids == {"c-1", "c-2"}


# ---------- _parse_one ----------

THREE_TOKEN_SENT = (
    "# sent_id = 1\n"
    "# text = Marcus puellam amat\n"
    "1\tMarcus\tmarcus\tPROPN\t_\tCase=Nom|Number=Sing\t_\t_\t_\t_\n"
    "2\tpuellam\tpuella\tNOUN\t_\tCase=Acc|Number=Sing\t_\t_\t_\t_\n"
    "3\tamat\tamo\tVERB\t_\tMood=Ind|Person=3\t_\t_\t_\t_\n"
)


def test_parse_one_happy_path_assigns_predictions():
    sent = conllu.parse(THREE_TOKEN_SENT)[0]
    fake = {"tokens": [
        {"id": 1, "head": 3, "deprel": "nsubj"},
        {"id": 2, "head": 3, "deprel": "obj"},
        {"id": 3, "head": 0, "deprel": "root"},
    ]}
    n_toks, n_fb = _FakeModel(fake)._parse_one(sent)
    assert n_toks == 3
    assert n_fb == 0
    assert [(t["head"], t["deprel"]) for t in sent] == [
        (3, "nsubj"), (3, "obj"), (0, "root"),
    ]


def test_parse_one_missing_token_falls_back():
    sent = conllu.parse(THREE_TOKEN_SENT)[0]
    fake = {"tokens": [
        {"id": 1, "head": 3, "deprel": "nsubj"},
        # id=2 missing -> right-branching fallback (head=3, deprel="dep")
        {"id": 3, "head": 0, "deprel": "root"},
    ]}
    n_toks, n_fb = _FakeModel(fake)._parse_one(sent)
    assert n_toks == 3
    assert n_fb == 1
    assert (sent[1]["head"], sent[1]["deprel"]) == (3, "dep")


def test_parse_one_out_of_range_head_falls_back():
    sent = conllu.parse(THREE_TOKEN_SENT)[0]
    fake = {"tokens": [
        {"id": 1, "head": 99, "deprel": "nsubj"},   # 99 not in {1,2,3}
        {"id": 2, "head": 3, "deprel": "obj"},
        {"id": 3, "head": 0, "deprel": "root"},
    ]}
    n_toks, n_fb = _FakeModel(fake)._parse_one(sent)
    assert n_fb == 1
    # token 1 fell back to right-branching: next id is 2, deprel "dep"
    assert (sent[0]["head"], sent[0]["deprel"]) == (2, "dep")


def test_parse_one_request_exception_fully_falls_back():
    sent = conllu.parse(THREE_TOKEN_SENT)[0]
    n_toks, n_fb = _FakeModel(requests.ConnectionError("boom"))._parse_one(sent)
    assert n_toks == 3
    assert n_fb == 3
    # right-branching: 1->2, 2->3, 3->root
    assert [(t["head"], t["deprel"]) for t in sent] == [
        (2, "dep"), (3, "dep"), (0, "root"),
    ]


def test_parse_one_invalid_json_fully_falls_back():
    sent = conllu.parse(THREE_TOKEN_SENT)[0]
    n_toks, n_fb = _FakeModel(ValueError("bad json"))._parse_one(sent)
    assert n_fb == 3


def test_parse_one_skips_multi_word_tokens():
    snippet = (
        "# sent_id = 1\n"
        "# text = a Raetisque b\n"
        "1\ta\ta\tX\t_\t_\t_\t_\t_\t_\n"
        "2-3\tRaetisque\t_\t_\t_\t_\t_\t_\t_\t_\n"
        "2\tRaetis\traetis\tX\t_\t_\t_\t_\t_\t_\n"
        "3\tque\tque\tX\t_\t_\t_\t_\t_\t_\n"
        "4\tb\tb\tX\t_\t_\t_\t_\t_\t_\n"
    )
    sent = conllu.parse(snippet)[0]
    fake = {"tokens": [
        {"id": 1, "head": 4, "deprel": "nsubj"},
        {"id": 2, "head": 4, "deprel": "obj"},
        {"id": 3, "head": 2, "deprel": "cc"},
        {"id": 4, "head": 0, "deprel": "root"},
    ]}
    n_toks, n_fb = _FakeModel(fake)._parse_one(sent)
    assert n_toks == 4
    assert n_fb == 0
    mwt = next(t for t in sent if not isinstance(t["id"], int))
    assert mwt["head"] is None or mwt["head"] == "_"


# ---------- _is_valid_tree + tree-level fallback ----------

def _toks_with_heads(triples):
    """[(id, head, deprel), ...] -> list of dict tokens."""
    return [{"id": i, "head": h, "deprel": d} for i, h, d in triples]


def test_is_valid_tree_accepts_well_formed_tree():
    assert _is_valid_tree(_toks_with_heads([(1, 3, "nsubj"), (2, 3, "obj"), (3, 0, "root")]))


def test_is_valid_tree_rejects_two_roots():
    assert not _is_valid_tree(_toks_with_heads([(1, 0, "root"), (2, 0, "root"), (3, 1, "dep")]))


def test_is_valid_tree_rejects_no_root():
    assert not _is_valid_tree(_toks_with_heads([(1, 2, "dep"), (2, 1, "dep")]))


def test_is_valid_tree_rejects_cycle():
    # 1 -> 2 -> 3 -> 1 cycle, with 4 as root (one root, but cycle exists)
    assert not _is_valid_tree(
        _toks_with_heads([(1, 2, "dep"), (2, 3, "dep"), (3, 1, "dep"), (4, 0, "root")])
    )


def test_parse_one_cycle_is_repaired_not_wiped():
    """A cycle in the model output should be REPAIRED (re-point the highest-id
    member of the cycle to the root) — not wiped to right-branching defaults.

    The old behavior threw away all the model's labels for any sentence with
    a structurally invalid tree, which discarded perfectly good per-token
    predictions for ~half the sentences on real bench runs.
    """
    sent = conllu.parse(THREE_TOKEN_SENT)[0]
    # 1 -> 2, 2 -> 1, 3 root → cycle between 1 and 2; highest-id member is 2
    fake = {"tokens": [
        {"id": 1, "head": 2, "deprel": "obj"},
        {"id": 2, "head": 1, "deprel": "nsubj"},
        {"id": 3, "head": 0, "deprel": "root"},
    ]}
    n_toks, n_fb = _FakeModel(fake)._parse_one(sent)
    assert n_toks == 3
    # Cycle broken by re-pointing id=2 (highest cycle member) to root (id=3).
    # Deprels are preserved; only id=2's head changed.
    assert (sent[0]["head"], sent[0]["deprel"]) == (2, "obj")     # unchanged
    assert (sent[1]["head"], sent[1]["deprel"]) == (3, "nsubj")   # head repaired
    assert (sent[2]["head"], sent[2]["deprel"]) == (0, "root")    # unchanged
    # n_fb counts heads we had to mutate vs what the model said: 1 here.
    assert n_fb == 1


def test_parse_one_multi_root_is_repaired_not_wiped():
    """Two head=0 tokens should be reduced to one root, with the extras
    re-pointed to the first root. Model deprels are preserved."""
    sent = conllu.parse(THREE_TOKEN_SENT)[0]
    fake = {"tokens": [
        {"id": 1, "head": 0, "deprel": "root"},
        {"id": 2, "head": 1, "deprel": "obj"},
        {"id": 3, "head": 0, "deprel": "root"},
    ]}
    n_toks, n_fb = _FakeModel(fake)._parse_one(sent)
    assert n_toks == 3
    assert (sent[0]["head"], sent[0]["deprel"]) == (0, "root")
    assert (sent[1]["head"], sent[1]["deprel"]) == (1, "obj")
    # id=3 re-pointed to id=1; deprel preserved
    assert sent[2]["head"] == 1
    assert sent[2]["deprel"] == "root"
    assert n_fb == 1


# ---------- _repair_tree ----------

def test_repair_tree_noop_on_valid_tree():
    tokens = _toks_with_heads([(1, 3, "nsubj"), (2, 3, "obj"), (3, 0, "root")])
    n_changed = _repair_tree(tokens)
    assert n_changed == 0
    assert [(t["head"], t["deprel"]) for t in tokens] == [
        (1, "nsubj"), (2, "obj"), (3, "root"),
    ][:0] or True  # placeholder; real check below
    # Use the same triples
    assert tokens[0]["head"] == 3 and tokens[0]["deprel"] == "nsubj"
    assert tokens[1]["head"] == 3 and tokens[1]["deprel"] == "obj"
    assert tokens[2]["head"] == 0 and tokens[2]["deprel"] == "root"


def test_repair_tree_multi_root_keeps_first_repoints_rest():
    tokens = _toks_with_heads([
        (1, 0, "root"),
        (2, 0, "root"),
        (3, 0, "root"),
        (4, 2, "obj"),
    ])
    n_changed = _repair_tree(tokens)
    # ids 2 and 3 were re-pointed to id 1 → 2 changes
    assert n_changed == 2
    assert tokens[0]["head"] == 0           # id=1 stays root
    assert tokens[1]["head"] == 1           # id=2 → 1
    assert tokens[2]["head"] == 1           # id=3 → 1
    assert tokens[3]["head"] == 2           # id=4 unchanged (still child of id=2)
    # Deprels untouched
    assert [t["deprel"] for t in tokens] == ["root", "root", "root", "obj"]
    assert _is_valid_tree(tokens)


def test_repair_tree_no_root_promotes_last_token():
    """If no token has head=0, promote the last token to root."""
    tokens = _toks_with_heads([(1, 2, "obj"), (2, 1, "nsubj")])  # cycle, no root
    n_changed = _repair_tree(tokens)
    assert tokens[-1]["head"] == 0
    assert tokens[-1]["deprel"] == "root"
    assert _is_valid_tree(tokens)
    # We changed at least the promoted token's head
    assert n_changed >= 1


def test_repair_tree_breaks_two_node_cycle():
    """Cycle 1<->2 with id=3 as root; cycle is broken at the highest-id member."""
    tokens = _toks_with_heads([
        (1, 2, "obj"),
        (2, 1, "nsubj"),
        (3, 0, "root"),
    ])
    n_changed = _repair_tree(tokens)
    assert n_changed == 1
    # id=2 (highest in cycle) re-pointed to root (id=3); deprels preserved
    assert tokens[1]["head"] == 3
    assert tokens[1]["deprel"] == "nsubj"
    assert tokens[0]["head"] == 2  # unchanged
    assert tokens[0]["deprel"] == "obj"
    assert _is_valid_tree(tokens)


def test_repair_tree_breaks_three_node_cycle():
    """1 -> 2 -> 3 -> 1 cycle, id=4 root. Highest cycle member (id=3) repaired."""
    tokens = _toks_with_heads([
        (1, 2, "obj"),
        (2, 3, "nmod"),
        (3, 1, "nsubj"),
        (4, 0, "root"),
    ])
    n_changed = _repair_tree(tokens)
    assert n_changed == 1
    assert tokens[2]["head"] == 4  # id=3 re-pointed to root
    assert tokens[2]["deprel"] == "nsubj"
    assert _is_valid_tree(tokens)


def test_repair_tree_breaks_disjoint_cycles():
    """Two independent cycles must both be broken."""
    tokens = _toks_with_heads([
        (1, 2, "obj"),
        (2, 1, "nsubj"),     # cycle A: {1, 2}
        (3, 4, "obj"),
        (4, 3, "nsubj"),     # cycle B: {3, 4}
        (5, 0, "root"),
    ])
    n_changed = _repair_tree(tokens)
    assert n_changed == 2  # one head per cycle
    assert _is_valid_tree(tokens)
    # Highest in each cycle re-pointed to root (id=5)
    assert tokens[1]["head"] == 5
    assert tokens[3]["head"] == 5


def test_repair_tree_preserves_all_deprels_except_promoted_root():
    """Heads may change; deprels are only touched when promoting a new root."""
    tokens = _toks_with_heads([
        (1, 0, "root"),
        (2, 0, "root"),
        (3, 1, "obj"),
    ])
    _repair_tree(tokens)
    # All deprels intact — repair only mutates heads in this case
    assert [t["deprel"] for t in tokens] == ["root", "root", "obj"]


def test_repair_tree_empty_input_no_op():
    tokens = []
    n_changed = _repair_tree(tokens)
    assert n_changed == 0
    assert tokens == []


def test_repair_tree_single_token_promotes_to_root():
    tokens = _toks_with_heads([(1, 99, "nsubj")])  # head points nowhere valid
    _repair_tree(tokens)
    assert tokens[0]["head"] == 0
    assert tokens[0]["deprel"] == "root"
    assert _is_valid_tree(tokens)


def test_repair_tree_single_token_already_root_is_noop():
    tokens = _toks_with_heads([(1, 0, "root")])
    n_changed = _repair_tree(tokens)
    assert n_changed == 0
    assert tokens[0]["head"] == 0


# ---------- _call_llm (HTTP wiring) ----------

def _lmstudio_response(payload: dict) -> MagicMock:
    """MagicMock that mimics an LM Studio (OpenAI-compatible) HTTP response."""
    r = MagicMock()
    r.raise_for_status = MagicMock(return_value=None)
    r.json = MagicMock(return_value={
        "choices": [{"message": {"content": json.dumps(payload)}}]
    })
    return r


def test_call_llm_posts_correct_body():
    m = LMStudioModel(model_id="qwen3-0.6b-mlx", max_tokens=2048)
    single = [
        {"id": 1, "form": "a", "lemma": "_", "upos": "X", "feats": None},
        {"id": 2, "form": "b", "lemma": "_", "upos": "X", "feats": None},
    ]
    payload = {"tokens": [
        {"id": 1, "head": 2, "deprel": "nsubj"},
        {"id": 2, "head": 0, "deprel": "root"},
    ]}
    with patch("requests.post", return_value=_lmstudio_response(payload)) as mock_post:
        out = m._call_llm(single)

    mock_post.assert_called_once()
    args, kwargs = mock_post.call_args
    assert args[0] == "http://localhost:1234/v1/chat/completions"
    body = kwargs["json"]
    assert body["model"] == "qwen3-0.6b-mlx"
    assert body["stream"] is False
    assert body["temperature"] == 0.3
    assert body["max_tokens"] == 2048
    # OpenAI structured-output shape
    rf = body["response_format"]
    assert rf["type"] == "json_schema"
    assert rf["json_schema"]["strict"] is True
    assert rf["json_schema"]["schema"] == SCHEMA
    # system + user messages
    roles = [msg["role"] for msg in body["messages"]]
    assert roles == ["system", "user"]
    assert body["messages"][0]["content"] == SYSTEM_PROMPT
    # User message contains the token table + the explicit count/id hint
    user_msg = body["messages"][1]["content"]
    assert "1\ta\t_\tX\t_" in user_msg
    assert "2 tokens" in user_msg
    assert "[1, 2]" in user_msg
    assert out == payload


def test_call_llm_parses_json_string_from_message_content():
    m = LMStudioModel()
    payload = {"tokens": [{"id": 1, "head": 0, "deprel": "root"}]}
    with patch("requests.post", return_value=_lmstudio_response(payload)):
        out = m._call_llm([
            {"id": 1, "form": "x", "lemma": "_", "upos": "X", "feats": None}
        ])
    assert out == payload


# ---------- predict (orchestrator) ----------

def test_predict_writes_valid_conllu_with_predictions(tmp_path, capsys):
    test_file = tmp_path / "test.conllu"
    test_file.write_text(
        "# sent_id = s1\n"
        "# text = Marcus amat\n"
        "1\tMarcus\tmarcus\tPROPN\t_\tCase=Nom\t_\t_\t_\t_\n"
        "2\tamat\tamo\tVERB\t_\tMood=Ind\t_\t_\t_\t_\n"
        "\n"
        "# sent_id = s2\n"
        "# text = puella ridet\n"
        "1\tpuella\tpuella\tNOUN\t_\tCase=Nom\t_\t_\t_\t_\n"
        "2\tridet\trideo\tVERB\t_\tMood=Ind\t_\t_\t_\t_\n"
    )
    out_file = tmp_path / "pred.conllu"

    responses = iter([
        {"tokens": [
            {"id": 1, "head": 2, "deprel": "nsubj"},
            {"id": 2, "head": 0, "deprel": "root"},
        ]},
        {"tokens": [
            {"id": 1, "head": 2, "deprel": "nsubj"},
            {"id": 2, "head": 0, "deprel": "root"},
        ]},
    ])
    model = _FakeModel({})
    model._fake = None
    model.num_workers = 1  # iterator-backed fake isn't thread-safe

    def fake_call(single):
        return next(responses)

    model._call_llm = fake_call
    model.predict(test_file, out_file)

    parsed = conllu.parse(out_file.read_text())
    assert len(parsed) == 2
    s1, s2 = parsed
    assert [(t["head"], t["deprel"]) for t in s1] == [(2, "nsubj"), (0, "root")]
    assert [(t["head"], t["deprel"]) for t in s2] == [(2, "nsubj"), (0, "root")]

    out = capsys.readouterr().out
    # default model id slug
    assert "qwen3-0.6b-mlx" in out
    assert "2 sentences" in out
    assert "4 tokens" in out
    assert "0 fallback tokens" in out


def test_predict_resumes_from_partial_file(tmp_path, capsys):
    """If a .partial.json exists from a prior crash, replay it and only
    LLM-call the remaining sentences."""
    test_file = tmp_path / "test.conllu"
    test_file.write_text(
        "# sent_id = s1\n"
        "# text = a b\n"
        "1\ta\ta\tX\t_\t_\t_\t_\t_\t_\n"
        "2\tb\tb\tX\t_\t_\t_\t_\t_\t_\n"
        "\n"
        "# sent_id = s2\n"
        "# text = c d\n"
        "1\tc\tc\tX\t_\t_\t_\t_\t_\t_\n"
        "2\td\td\tX\t_\t_\t_\t_\t_\t_\n"
    )
    out_file = tmp_path / "pred.conllu"
    partial_file = out_file.with_suffix(".partial.json")

    partial_file.write_text(json.dumps({
        "0": {
            "tokens": {
                "1": {"head": 2, "deprel": "nsubj"},
                "2": {"head": 0, "deprel": "root"},
            },
            "n_toks": 2,
            "n_fb": 0,
        }
    }))

    n_calls = {"count": 0}

    class _CountingFake(_FakeModel):
        def _call_llm(self, single):
            n_calls["count"] += 1
            return {"tokens": [
                {"id": 1, "head": 2, "deprel": "nsubj"},
                {"id": 2, "head": 0, "deprel": "root"},
            ]}

    model = _CountingFake({})
    model.num_workers = 1
    model.predict(test_file, out_file)

    assert n_calls["count"] == 1

    parsed = conllu.parse(out_file.read_text())
    assert len(parsed) == 2
    assert [(t["head"], t["deprel"]) for t in parsed[0]] == [(2, "nsubj"), (0, "root")]
    assert [(t["head"], t["deprel"]) for t in parsed[1]] == [(2, "nsubj"), (0, "root")]
    assert not partial_file.exists()
    out = capsys.readouterr().out
    assert "resuming" in out


# ---------- _format_assistant_response ----------

def test_format_assistant_response_emits_compact_json():
    from latinbench.models.lmstudio_llm import _format_assistant_response

    sent = conllu.parse(
        "# sent_id = t\n"
        "1\tMarcus\tmarcus\tPROPN\t_\t_\t2\tnsubj\t_\t_\n"
        "2\tpoeta\tpoeta\tNOUN\t_\t_\t0\troot\t_\t_\n"
        "3\test\tsum\tAUX\t_\t_\t2\tcop\t_\t_\n"
    )[0]
    single = [t for t in sent if isinstance(t["id"], int)]
    out = _format_assistant_response(single)
    parsed = json.loads(out)
    assert parsed == {"tokens": [
        {"id": 1, "head": 2, "deprel": "nsubj"},
        {"id": 2, "head": 0, "deprel": "root"},
        {"id": 3, "head": 2, "deprel": "cop"},
    ]}


# ---------- _build_messages (few-shot chat history) ----------

def _target_single():
    return [
        {"id": 1, "form": "a", "lemma": "_", "upos": "X", "feats": None},
        {"id": 2, "form": "b", "lemma": "_", "upos": "X", "feats": None},
    ]


def test_build_messages_zero_shot_is_system_then_user():
    m = LMStudioModel(model_id="qwen3-0.6b-mlx")  # k_shot=0
    messages = m._build_messages(_target_single())
    assert [msg["role"] for msg in messages] == ["system", "user"]
    assert messages[0]["content"] == SYSTEM_PROMPT


def test_build_messages_k_shot_2_interleaves_user_assistant_demos():
    m = LMStudioModel(model_id="qwen3-0.6b-mlx", k_shot=2)
    messages = m._build_messages(_target_single())
    # system + (user, assistant) × 2 demos + final user = 6 messages
    assert [msg["role"] for msg in messages] == [
        "system", "user", "assistant", "user", "assistant", "user",
    ]
    # System prompt unchanged
    assert messages[0]["content"] == SYSTEM_PROMPT
    # Demonstration assistant turns parse as JSON with a "tokens" key
    for demo_idx in (2, 4):
        parsed = json.loads(messages[demo_idx]["content"])
        assert "tokens" in parsed and isinstance(parsed["tokens"], list)
        for entry in parsed["tokens"]:
            assert set(entry.keys()) == {"id", "head", "deprel"}
    # Target sentence is the LAST user message and uses the same formatter
    # as demonstration user messages (no special framing).
    final_user = messages[-1]["content"]
    assert "2 tokens" in final_user
    assert "[1, 2]" in final_user


def test_build_messages_k_shot_4_has_4_demo_pairs():
    m = LMStudioModel(model_id="qwen3-0.6b-mlx", k_shot=4)
    messages = m._build_messages(_target_single())
    roles = [msg["role"] for msg in messages]
    # system + 4 × (user, assistant) + user = 10 messages
    assert len(roles) == 10
    assert roles[0] == "system"
    assert roles[-1] == "user"
    # Interleaving: positions 1,3,5,7 are user demos; 2,4,6,8 are assistant demos
    for i in (1, 3, 5, 7):
        assert roles[i] == "user"
    for i in (2, 4, 6, 8):
        assert roles[i] == "assistant"


def test_call_llm_posts_few_shot_body_with_chat_history():
    """End-to-end: the HTTP body for a k=2 call has the expected interleaved
    chat history, and the final assistant generation still gets the JSON
    schema constraint applied."""
    m = LMStudioModel(model_id="qwen3-0.6b-mlx", k_shot=2)
    payload = {"tokens": [
        {"id": 1, "head": 2, "deprel": "nsubj"},
        {"id": 2, "head": 0, "deprel": "root"},
    ]}
    with patch("requests.post", return_value=_lmstudio_response(payload)) as mock_post:
        m._call_llm(_target_single())

    body = mock_post.call_args.kwargs["json"]
    roles = [msg["role"] for msg in body["messages"]]
    assert roles == ["system", "user", "assistant", "user", "assistant", "user"]
    # Schema still applies to the final completion only
    assert body["response_format"]["type"] == "json_schema"
    assert body["response_format"]["json_schema"]["strict"] is True


def test_predict_with_k_shot_2_uses_few_shot_messages(tmp_path):
    """Full path: LMStudioModel(k_shot=2).predict() sends 6-message chat turns
    (system + 2 user/assistant demo pairs + target user) to the HTTP layer."""
    test_file = tmp_path / "test.conllu"
    test_file.write_text(
        "# sent_id = s1\n"
        "# text = Marcus amat\n"
        "1\tMarcus\tmarcus\tPROPN\t_\tCase=Nom\t_\t_\t_\t_\n"
        "2\tamat\tamo\tVERB\t_\tMood=Ind\t_\t_\t_\t_\n"
    )
    out_file = tmp_path / "pred.conllu"
    payload = {"tokens": [
        {"id": 1, "head": 2, "deprel": "nsubj"},
        {"id": 2, "head": 0, "deprel": "root"},
    ]}
    m = LMStudioModel(model_id="qwen3-0.6b-mlx", k_shot=2)
    m.num_workers = 1
    with patch("requests.post", return_value=_lmstudio_response(payload)) as mock_post:
        m.predict(test_file, out_file)

    # The single HTTP call carries the few-shot chat history
    assert mock_post.call_count == 1
    body = mock_post.call_args.kwargs["json"]
    roles = [msg["role"] for msg in body["messages"]]
    assert roles == ["system", "user", "assistant", "user", "assistant", "user"]
    # Final assistant generation is still schema-constrained
    assert body["response_format"]["json_schema"]["strict"] is True


# ---------- few-shot pool-tag slug isolation ----------

def _perseus_pool():
    from latinbench.few_shot import ExamplePool, DEFAULT_EXAMPLES_PATH
    return ExamplePool(DEFAULT_EXAMPLES_PATH.parent / "few_shot_examples_perseus.conllu")


def test_name_slug_perseus_pool_appends_pool_tag():
    m = LMStudioModel(model_id="qwen3-0.6b-mlx", k_shot=2, example_pool=_perseus_pool())
    assert m.name == "qwen3-0.6b-mlx-2shot-perseus"


def test_name_slug_default_pool_has_no_tag_suffix():
    """Regression: the default (hand-curated) pool keeps the bare -2shot slug so
    existing committed predictions are not orphaned."""
    assert LMStudioModel(model_id="qwen3-0.6b-mlx", k_shot=2).name == "qwen3-0.6b-mlx-2shot"


def test_name_slug_seed_and_pool_tag_order():
    """Suffix order is -{k}shot -s{seed} -{pool}, matching the notebook-03 slug
    parser."""
    m = LMStudioModel(
        model_id="qwen3-0.6b-mlx", k_shot=2, shot_seed=7, example_pool=_perseus_pool()
    )
    assert m.name == "qwen3-0.6b-mlx-2shot-s7-perseus"
