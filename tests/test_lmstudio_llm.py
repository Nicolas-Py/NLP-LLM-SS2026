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


def test_parse_one_cycle_falls_back_whole_sentence():
    sent = conllu.parse(THREE_TOKEN_SENT)[0]
    # 1 -> 2, 2 -> 1, 3 root → cycle between 1 and 2
    fake = {"tokens": [
        {"id": 1, "head": 2, "deprel": "obj"},
        {"id": 2, "head": 1, "deprel": "nsubj"},
        {"id": 3, "head": 0, "deprel": "root"},
    ]}
    n_toks, n_fb = _FakeModel(fake)._parse_one(sent)
    assert n_toks == 3
    assert n_fb == 3
    assert [(t["head"], t["deprel"]) for t in sent] == [
        (2, "dep"), (3, "dep"), (0, "root"),
    ]


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
