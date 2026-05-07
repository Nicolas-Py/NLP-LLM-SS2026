"""Registry of reference models.

The two reference models are eagerly importable. The template lives in
`template.py` and is meant to be imported by the user's notebook directly.
"""
from __future__ import annotations

from ..core import Model
from .latinpipe import LatinpipeModel
from .udpipe_baseline import UdpipeBaselineModel


def _make_registry() -> dict[str, Model]:
    # Lazy: instantiating LatinpipeModel checks for the checkpoint, which may
    # not exist yet. We keep the registry as factories so imports stay cheap
    # and errors only fire when the user actually runs that model.
    return {
        "udpipe_baseline": UdpipeBaselineModel(),
        "latinpipe": LatinpipeModel(),
    }


class _LazyRegistry(dict):
    def __missing__(self, key: str) -> Model:
        registry = _make_registry()
        for k, v in registry.items():
            self[k] = v
        if key not in self:
            raise KeyError(key)
        return self[key]


MODELS: dict[str, Model] = _LazyRegistry()
