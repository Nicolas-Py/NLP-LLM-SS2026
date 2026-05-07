"""UDPipe 2 + UD Latin-Perseus via the LINDAT REST API.

The official EvaLatin 2024 baseline. No local install — just an HTTP POST per
file. Pinned to `DEFAULT_MODEL_ID` for reproducibility; pass any other LINDAT
model id (or `"latest"`) to try alternatives. Each model id gets its own
`predictions/<id>/` cache dir so swapping versions doesn't clobber old runs.
"""
from __future__ import annotations
from pathlib import Path

import requests

from ..core import Model

API = "https://lindat.mff.cuni.cz/services/udpipe/api"

# Pinned so the cached reference scores in the README stay reproducible.
# Bump this when you've re-run and updated the scores.
DEFAULT_MODEL_ID = "latin-perseus-ud-2.17-251125"


def list_perseus_models() -> list[str]:
    """All `latin-perseus-ud-*` model ids LINDAT currently serves, oldest first.

    Sort key is the trailing YYMMDD date, so `2.17-251125` correctly comes
    after `2.6-200830` (alphabetical sort would not).
    """
    models = requests.get(f"{API}/models", timeout=30).json()["models"]
    perseus = [m for m in models if "latin-perseus" in m]
    if not perseus:
        raise RuntimeError("LINDAT returned no latin-perseus models")
    return sorted(perseus, key=lambda m: m.rsplit("-", 1)[-1])


def _latest_perseus_model() -> str:
    return list_perseus_models()[-1]


class UdpipeModel(Model):
    def __init__(self, model_id: str | None = None) -> None:
        if model_id is None:
            self.model_id = DEFAULT_MODEL_ID
        elif model_id == "latest":
            self.model_id = _latest_perseus_model()
        else:
            self.model_id = model_id
        # Cache dir is per-model so trying alternatives doesn't overwrite results.
        self.name = self.model_id

    def predict(self, test_path: Path, out_path: Path) -> None:
        print(f"[{self.name}] using LINDAT model {self.model_id}")
        # Empty tagger= and parser= flags = "don't retag, only parse".
        # The pre-tagged columns from the test file pass through unchanged.
        r = requests.post(
            f"{API}/process",
            data={
                "model": self.model_id,
                "input": "conllu",
                "tagger": "",
                "parser": "",
                "data": test_path.read_text(),
            },
            timeout=600,
        )
        r.raise_for_status()
        out_path.write_text(r.json()["result"])
