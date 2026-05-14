from __future__ import annotations

import json
from unittest.mock import patch, MagicMock

import conllu
import requests

from latinbench.models.ollama_llm import (
    OllamaLLMModel,
    SCHEMA,
    SYSTEM_PROMPT,
    _collect_deprels_from_gold,
    _format_sentence,
    _is_valid_tree,
    _right_branching_default,
)


class _FakeModel(OllamaLLMModel):
    """OllamaLLMModel with `_call_ollama` swapped for a fixture."""
    def __init__(self, fake):
        super().__init__()
        self._fake = fake

    def _call_ollama(self, single):
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
    head, deprel = _right_branching_default(single, 2)
    assert head == 0
    assert deprel == "root"


def test_right_branching_non_last_attaches_to_next():
    single = _toks([1, 2, 3])
    head, deprel = _right_branching_default(single, 0)
    assert head == 2
    assert deprel == "dep"


def test_right_branching_handles_non_contiguous_ids():
    # CoNLL-U IDs can have gaps from multi-word tokens; right-branching uses
    # the *next single-word token's id*, not i+1 arithmetic.
    single = _toks([1, 2, 4])
    head, deprel = _right_branching_default(single, 1)
    assert head == 4
    assert deprel == "dep"


# ---------- _format_sentence ----------

def test_format_sentence_basic_columns():
    single = [
        {"id": 1, "form": "Marcus", "lemma": "marcus", "upos": "PROPN",
         "feats": {"Case": "Nom", "Number": "Sing"}},
        {"id": 2, "form": "amat", "lemma": "amo", "upos": "VERB",
         "feats": {"Mood": "Ind", "VerbForm": "Fin"}},
    ]
    out = _format_sentence(single)
    lines = out.strip().split("\n")
    assert len(lines) == 2
    # tab-separated id form lemma upos feats
    assert lines[0].split("\t") == [
        "1", "Marcus", "marcus", "PROPN", "Case=Nom|Number=Sing"
    ]
    assert lines[1].split("\t") == [
        "2", "amat", "amo", "VERB", "Mood=Ind|VerbForm=Fin"
    ]


def test_format_sentence_empty_feats_renders_underscore():
    single = [
        {"id": 1, "form": "et", "lemma": "et", "upos": "CCONJ", "feats": None},
    ]
    out = _format_sentence(single).strip()
    assert out.split("\t") == ["1", "et", "et", "CCONJ", "_"]


# ---------- _collect_deprels_from_gold ----------

def test_collect_deprels_from_tmp_gold(tmp_path):
    fake = tmp_path / "fake.conllu"
    fake.write_text(
        "# sent_id = 1\n"
        "# text = a b\n"
        "1\ta\ta\tX\t_\t_\t2\tnsubj\t_\t_\n"
        "2\tb\tb\tX\t_\t_\t0\troot\t_\t_\n"
        "\n"
        "# sent_id = 2\n"
        "# text = c d\n"
        "1\tc\tc\tX\t_\t_\t2\tobj\t_\t_\n"
        "2\td\td\tX\t_\t_\t0\troot\t_\t_\n"
    )
    labels = _collect_deprels_from_gold([fake])
    assert labels == sorted(["nsubj", "root", "obj"])


def test_collect_deprels_from_real_gold_files_contains_common_relations():
    # Sanity check against the canonical EvaLatin 2024 gold files.
    labels = _collect_deprels_from_gold()
    for required in ("root", "nsubj", "obj", "obl", "advmod", "cc", "conj"):
        assert required in labels, f"expected {required!r} in deprel set"
    # All values are str and sorted
    assert all(isinstance(x, str) for x in labels)
    assert labels == sorted(labels)
    # Some subtype labels expected too (Latin treebanks use them)
    assert any(":" in x for x in labels), "expected subtype labels like obl:arg"


# ---------- OllamaLLMModel.name (slug derivation) ----------

def test_name_slug_replaces_colon():
    assert OllamaLLMModel(model_id="qwen3:0.6b").name == "qwen3-0.6b"


def test_name_slug_default_model_is_qwen3():
    assert OllamaLLMModel().name == "qwen3-0.6b"


def test_name_slug_replaces_slash_too():
    assert OllamaLLMModel(model_id="hf.co/foo:Q4_K_M").name == "hf.co-foo-Q4_K_M"


def test_host_is_trimmed_of_trailing_slash():
    m = OllamaLLMModel(host="http://localhost:11434/")
    assert m.host == "http://localhost:11434"


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
    # CoNLL-U range row "5-6 Raetisque" + the two split tokens "5 Raetis", "6 que".
    # Only single-word tokens get head/deprel; the MWT row stays as-is.
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
    assert n_toks == 4   # 4 single-word tokens; MWT row excluded from count
    assert n_fb == 0
    # MWT row (id is a tuple) untouched
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
    # LLM-style response that produces a cycle: 1 -> 2, 2 -> 1, 3 root
    fake = {"tokens": [
        {"id": 1, "head": 2, "deprel": "obj"},
        {"id": 2, "head": 1, "deprel": "nsubj"},
        {"id": 3, "head": 0, "deprel": "root"},
    ]}
    n_toks, n_fb = _FakeModel(fake)._parse_one(sent)
    assert n_toks == 3
    assert n_fb == 3  # whole sentence right-branched
    # Right-branching chain: 1 -> 2 ("dep"), 2 -> 3 ("dep"), 3 -> 0 ("root")
    assert [(t["head"], t["deprel"]) for t in sent] == [
        (2, "dep"), (3, "dep"), (0, "root"),
    ]


# ---------- _call_ollama (HTTP wiring) ----------

def _ollama_response(payload: dict) -> MagicMock:
    """Build a MagicMock that looks like a successful Ollama HTTP response."""
    r = MagicMock()
    r.raise_for_status = MagicMock(return_value=None)
    r.json = MagicMock(return_value={"message": {"content": json.dumps(payload)}})
    return r


def test_call_ollama_posts_correct_body():
    m = OllamaLLMModel(model_id="qwen3:0.6b", num_ctx=4096)
    single = [
        {"id": 1, "form": "a", "lemma": "_", "upos": "X", "feats": None},
        {"id": 2, "form": "b", "lemma": "_", "upos": "X", "feats": None},
    ]
    payload = {"tokens": [
        {"id": 1, "head": 2, "deprel": "nsubj"},
        {"id": 2, "head": 0, "deprel": "root"},
    ]}
    with patch("requests.post", return_value=_ollama_response(payload)) as mock_post:
        out = m._call_ollama(single)

    mock_post.assert_called_once()
    args, kwargs = mock_post.call_args
    assert args[0] == "http://localhost:11434/api/chat"
    body = kwargs["json"]
    assert body["model"] == "qwen3:0.6b"
    assert body["stream"] is False
    assert body["think"] is False               # critical: disable Qwen3 reasoning
    assert body["format"] == SCHEMA
    assert body["options"]["num_ctx"] == 4096
    assert body["options"]["temperature"] == 0
    # system + user messages
    roles = [msg["role"] for msg in body["messages"]]
    assert roles == ["system", "user"]
    assert body["messages"][0]["content"] == SYSTEM_PROMPT
    # User content includes the formatted token table
    assert "1\ta\t_\tX\t_" in body["messages"][1]["content"]
    # And the parsed response was returned
    assert out == payload


def test_call_ollama_parses_json_string_from_message_content():
    m = OllamaLLMModel()
    payload = {"tokens": [{"id": 1, "head": 0, "deprel": "root"}]}
    with patch("requests.post", return_value=_ollama_response(payload)):
        out = m._call_ollama([
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

    # Two canned responses, one per sentence
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
    model._fake = None  # will be set per-call below
    model.num_workers = 1  # iterator-backed fake isn't thread-safe

    def fake_call(single):
        return next(responses)

    # Patch _call_ollama on the instance with our iterator-backed function.
    model._call_ollama = fake_call

    model.predict(test_file, out_file)

    # File exists and is valid CoNLL-U
    parsed = conllu.parse(out_file.read_text())
    assert len(parsed) == 2
    s1, s2 = parsed
    assert [(t["head"], t["deprel"]) for t in s1] == [(2, "nsubj"), (0, "root")]
    assert [(t["head"], t["deprel"]) for t in s2] == [(2, "nsubj"), (0, "root")]

    # Summary line printed
    out = capsys.readouterr().out
    assert "qwen3-0.6b" in out
    assert "2 sentences" in out
    assert "4 tokens" in out
    assert "0 fallback tokens" in out
