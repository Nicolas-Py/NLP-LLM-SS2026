"""Local-LLM dependency parser via Ollama with constrained JSON-schema output.

See docs/superpowers/specs/2026-05-14-qwen3-ollama-llm-model-design.md.
"""
from __future__ import annotations
import json
from pathlib import Path

import conllu
import requests

from ..core import Model
from ..data import gold_path


SYSTEM_PROMPT = """\
You are a Universal Dependencies parser for Latin. Given a list of tokens \
with their lemma, UPOS, and morphological features, output the syntactic \
head ID and dependency relation for each token, in input order.

Rules:
- `head` is the parent token's ID, or 0 if this token is the root.
- Exactly one token in the sentence has head=0 with deprel="root".
- `deprel` must be from the provided vocabulary.

Example input:
1\tMarcus\tmarcus\tPROPN\tCase=Nom|Number=Sing
2\tpuellam\tpuella\tNOUN\tCase=Acc|Number=Sing
3\tamat\tamo\tVERB\tMood=Ind|Person=3|VerbForm=Fin

Example output:
{"tokens": [
  {"id": 1, "head": 3, "deprel": "nsubj"},
  {"id": 2, "head": 3, "deprel": "obj"},
  {"id": 3, "head": 0, "deprel": "root"}
]}
"""


DEFAULT_HOST = "http://localhost:11434"
DEFAULT_MODEL_ID = "qwen3:0.6b"


class OllamaLLMModel(Model):
    def __init__(
        self,
        model_id: str = DEFAULT_MODEL_ID,
        host: str = DEFAULT_HOST,
        num_ctx: int = 8192,
    ) -> None:
        self.model_id = model_id
        self.host = host.rstrip("/")
        self.num_ctx = num_ctx
        self.name = model_id.replace(":", "-").replace("/", "-")

    def predict(self, test_path: Path, out_path: Path) -> None:
        sentences = conllu.parse(Path(test_path).read_text())
        total_toks = 0
        total_fallback = 0
        fallback_sents = 0
        for i, sent in enumerate(sentences):
            n_toks, n_fb = self._parse_one(sent)
            total_toks += n_toks
            total_fallback += n_fb
            if n_fb > 0:
                fallback_sents += 1
            if i % 25 == 0:
                print(f"[{self.name}] {i}/{len(sentences)} sentences")
        Path(out_path).write_text("".join(s.serialize() for s in sentences))
        pct = (100.0 * total_fallback / total_toks) if total_toks else 0.0
        print(
            f"[{self.name}] {len(sentences)} sentences, {total_toks} tokens; "
            f"{total_fallback} fallback tokens ({pct:.1f}%) across "
            f"{fallback_sents} sentences"
        )

    def _call_ollama(self, single: list) -> dict:
        body = {
            "model": self.model_id,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": _format_sentence(single)},
            ],
            "format": SCHEMA,
            "stream": False,
            "think": False,
            "options": {"num_ctx": self.num_ctx, "temperature": 0},
        }
        r = requests.post(f"{self.host}/api/chat", json=body, timeout=120)
        r.raise_for_status()
        content = r.json()["message"]["content"]
        return json.loads(content)

    def _parse_one(self, sent: conllu.TokenList) -> tuple[int, int]:
        single = [t for t in sent if isinstance(t["id"], int)]
        valid_ids = {t["id"] for t in single}

        try:
            response = self._call_ollama(single)
            pred_by_id = {
                p["id"]: p for p in response.get("tokens", [])
                if isinstance(p, dict) and isinstance(p.get("id"), int)
            }
        except (requests.RequestException, ValueError):
            pred_by_id = {}

        n_fallback = 0
        for i, tok in enumerate(single):
            pred = pred_by_id.get(tok["id"])
            head = pred.get("head") if pred else None
            deprel = pred.get("deprel") if pred else None

            valid = (
                pred is not None
                and isinstance(head, int)
                and (head == 0 or head in valid_ids)
                and isinstance(deprel, str)
            )
            if not valid:
                head, deprel = _right_branching_default(single, i)
                n_fallback += 1
            tok["head"] = head
            tok["deprel"] = deprel
        return len(single), n_fallback


def _collect_deprels_from_gold(paths: list[Path] | None = None) -> list[str]:
    """Union of `deprel` values seen in gold CoNLL-U files, sorted."""
    if paths is None:
        paths = [gold_path("poetry"), gold_path("prose")]
    labels: set[str] = set()
    for p in paths:
        for sent in conllu.parse(Path(p).read_text()):
            for tok in sent:
                if isinstance(tok["id"], int) and tok.get("deprel"):
                    labels.add(tok["deprel"])
    return sorted(labels)


DEPREL_LABELS = _collect_deprels_from_gold()

SCHEMA = {
    "type": "object",
    "properties": {
        "tokens": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "integer", "minimum": 1},
                    "head": {"type": "integer", "minimum": 0},
                    "deprel": {"type": "string", "enum": DEPREL_LABELS},
                },
                "required": ["id", "head", "deprel"],
            },
        },
    },
    "required": ["tokens"],
}


def _right_branching_default(single: list, i: int) -> tuple[int, str]:
    n = len(single)
    if i == n - 1:
        return 0, "root"
    return single[i + 1]["id"], "dep"


def _format_sentence(single: list) -> str:
    """Render single-word tokens as a tab-separated table for the prompt."""
    rows = []
    for tok in single:
        feats = tok.get("feats")
        feats_str = "|".join(f"{k}={v}" for k, v in feats.items()) if feats else "_"
        rows.append("\t".join([
            str(tok["id"]),
            str(tok["form"]),
            str(tok["lemma"]),
            str(tok["upos"]),
            feats_str,
        ]))
    return "\n".join(rows)
