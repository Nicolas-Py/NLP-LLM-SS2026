"""Local-LLM dependency parser via LM Studio (OpenAI-compatible API).

LM Studio exposes any loaded model behind `POST /v1/chat/completions` and
supports structured output via `response_format: {type: "json_schema", ...}`.
"""
from __future__ import annotations
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock

import conllu
import requests

from ..core import Model
from ..data import gold_path


SYSTEM_PROMPT = """\
You are a Universal Dependencies parser for Latin. The user will give you a \
table of tokens with their lemma, UPOS, and morphological features. You must \
predict the syntactic head and dependency relation for every token.

Rules:
- Output exactly one entry per input token, in the order given.
- Preserve the input ids exactly (do not renumber, do not invent new ids).
- `head` is the parent token's id (within the sentence's id range), or 0 if this token is the root.
- Exactly one token has head=0 with deprel="root".
- `deprel` must be a valid Universal Dependencies relation.
"""


DEFAULT_HOST = "http://localhost:1234"
DEFAULT_MODEL_ID = "qwen3-0.6b-mlx"


class LMStudioModel(Model):
    def __init__(
        self,
        model_id: str = DEFAULT_MODEL_ID,
        host: str = DEFAULT_HOST,
        num_workers: int = 8,
        max_tokens: int = 4096,
        temperature: float = 0.3,
    ) -> None:
        self.model_id = model_id
        self.host = host.rstrip("/")
        self.num_workers = num_workers
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.name = model_id.replace(":", "-").replace("/", "-")

    def predict(self, test_path: Path, out_path: Path) -> None:
        sentences = conllu.parse(Path(test_path).read_text())
        n_total = len(sentences)
        partial_path = Path(out_path).with_suffix(".partial.json")

        # Resume: if a partial file exists from a prior crashed run, replay it
        # back onto the sentence list and only process the remaining indices.
        partial: dict[str, dict] = {}
        done_idx: set[int] = set()
        if partial_path.exists():
            try:
                partial = json.loads(partial_path.read_text())
            except (ValueError, OSError):
                partial = {}
            for idx_str, sent_state in partial.items():
                idx = int(idx_str)
                if idx >= n_total:
                    continue
                tokens_state = sent_state.get("tokens", {})
                for tok in sentences[idx]:
                    if isinstance(tok["id"], int):
                        s = tokens_state.get(str(tok["id"]))
                        if s:
                            tok["head"] = s["head"]
                            tok["deprel"] = s["deprel"]
                done_idx.add(idx)
            if done_idx:
                print(f"[{self.name}] resuming: {len(done_idx)}/{n_total} sentences cached")

        pending_idx = [i for i in range(n_total) if i not in done_idx]
        total_toks = sum(p.get("n_toks", 0) for p in partial.values())
        total_fallback = sum(p.get("n_fb", 0) for p in partial.values())
        fallback_sents = sum(1 for p in partial.values() if p.get("n_fb", 0) > 0)
        done = len(done_idx)
        lock = Lock()

        if pending_idx:
            with ThreadPoolExecutor(max_workers=self.num_workers) as pool:
                future_to_idx = {
                    pool.submit(self._parse_one, sentences[i]): i for i in pending_idx
                }
                for f in as_completed(future_to_idx):
                    idx = future_to_idx[f]
                    n_toks, n_fb = f.result()
                    total_toks += n_toks
                    total_fallback += n_fb
                    if n_fb > 0:
                        fallback_sents += 1
                    with lock:
                        partial[str(idx)] = {
                            "tokens": {
                                str(t["id"]): {"head": t["head"], "deprel": t["deprel"]}
                                for t in sentences[idx] if isinstance(t["id"], int)
                            },
                            "n_toks": n_toks,
                            "n_fb": n_fb,
                        }
                        partial_path.write_text(json.dumps(partial))
                    done += 1
                    if done % 25 == 0 or done == n_total:
                        print(f"[{self.name}] {done}/{n_total} sentences")

        Path(out_path).write_text("".join(s.serialize() for s in sentences))
        partial_path.unlink(missing_ok=True)

        pct = (100.0 * total_fallback / total_toks) if total_toks else 0.0
        print(
            f"[{self.name}] {n_total} sentences, {total_toks} tokens; "
            f"{total_fallback} fallback tokens ({pct:.1f}%) across "
            f"{fallback_sents} sentences"
        )

    def _call_llm(self, single: list) -> dict:
        body = {
            "model": self.model_id,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": _format_user_message(single)},
            ],
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "ud_parse",
                    "strict": True,
                    "schema": SCHEMA,
                },
            },
            "stream": False,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        r = requests.post(f"{self.host}/v1/chat/completions", json=body, timeout=180)
        r.raise_for_status()
        content = r.json()["choices"][0]["message"]["content"]
        return json.loads(content)

    def _parse_one(self, sent: conllu.TokenList) -> tuple[int, int]:
        single = [t for t in sent if isinstance(t["id"], int)]
        valid_ids = {t["id"] for t in single}

        try:
            response = self._call_llm(single)
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

        # Tree-level repair: cycles + multi-root would crash the UD scorer.
        # If the result isn't a valid rooted tree, fall the whole sentence
        # back to right-branching (already used as the per-token default).
        if not _is_valid_tree(single):
            for i, tok in enumerate(single):
                tok["head"], tok["deprel"] = _right_branching_default(single, i)
            n_fallback = len(single)
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


def _is_valid_tree(single: list) -> bool:
    """True if the assigned heads form a single rooted tree with no cycles."""
    if not single:
        return True
    head_map = {t["id"]: t["head"] for t in single}
    if sum(1 for h in head_map.values() if h == 0) != 1:
        return False
    n = len(head_map)
    for tid in head_map:
        cur = head_map[tid]
        for _ in range(n + 1):
            if cur == 0:
                break
            if cur not in head_map:
                return False
            cur = head_map[cur]
        else:
            return False  # cycle: walked n+1 steps without reaching root
    return True


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


def _format_user_message(single: list) -> str:
    """User-side prompt: token table preceded by an explicit count + id list.

    Small models tend to collapse onto whatever example sits in the system
    prompt; spelling out the input shape per-sentence makes it harder to
    ignore. Each token row is `id\tform\tlemma\tupos\tfeats`.
    """
    ids = [t["id"] for t in single]
    return (
        f"Parse this Latin sentence ({len(single)} tokens, "
        f"ids {ids}).\n\n"
        f"{_format_sentence(single)}\n\n"
        f"Output a JSON object with a \"tokens\" array of {len(single)} entries, "
        f"one per row above, preserving the ids."
    )
