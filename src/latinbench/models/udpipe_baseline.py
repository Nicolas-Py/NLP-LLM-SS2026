"""UDPipe 2 + UD Latin-Perseus via the LINDAT REST API.

This is the official baseline from EvaLatin 2024. No local install — just an
HTTP POST per file. Auto-picks the latest `latin-perseus-ud-X.Y-YYMMDD` model
served by LINDAT.
"""
from __future__ import annotations
from pathlib import Path

import requests

from ..core import Model

API = "https://lindat.mff.cuni.cz/services/udpipe/api"


def _latest_perseus_model() -> str:
    """Query the LINDAT model list and return the most recent latin-perseus-ud-*.

    Names look like `latin-perseus-ud-2.17-251125`. We sort by the trailing
    YYMMDD date — alphabetical sort would put `2.6-200830` after `2.17-251125`.
    """
    models = requests.get(f"{API}/models", timeout=30).json()["models"]
    perseus = [m for m in models if "latin-perseus" in m]
    if not perseus:
        raise RuntimeError("LINDAT returned no latin-perseus models")
    # date suffix is the last hyphen-separated component
    return max(perseus, key=lambda m: m.rsplit("-", 1)[-1])


class UdpipeBaselineModel(Model):
    name = "udpipe_baseline"

    def __init__(self, model_id: str | None = None) -> None:
        self.model_id = model_id or _latest_perseus_model()

    def predict(self, test_path: Path, out_path: Path) -> None:
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
