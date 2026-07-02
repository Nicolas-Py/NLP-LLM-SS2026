"""Registry of reference models.

The two reference models are eagerly importable. The template lives in
`template.py` and is meant to be imported by the user's notebook directly.
"""
from __future__ import annotations

from ..core import Model
from .latinpipe import LatinpipeModel
from .lmstudio_llm import LMStudioModel
from .openrouter_llm import OpenRouterModel
from .udpipe import UdpipeModel


def _make_registry() -> dict[str, Model]:
    # Lazy: instantiating LatinpipeModel checks for the checkpoint, which may
    # not exist yet. We keep the registry as factories so imports stay cheap
    # and errors only fire when the user actually runs that model.
    #
    # The LM Studio entries share one running LM Studio server; only one model
    # is hot in memory at a time but LM Studio auto-swaps on request. Each
    # entry gets its own predictions/<slug>/ cache dir, so results from
    # earlier runs are preserved across model swaps.
    return {
        "udpipe": UdpipeModel(),
        "latinpipe": LatinpipeModel(),
        "qwen3-lmstudio": LMStudioModel(),  # 0.6B baseline
        "qwen3-vl-8b-lmstudio": LMStudioModel(model_id="qwen3-vl-8b-instruct-mlx"),
        "gemma-3-12b-lmstudio": LMStudioModel(model_id="google/gemma-3-12b"),
        # Hosted via OpenRouter (needs OPENROUTER_API_KEY at run time; the key
        # is resolved lazily so building this registry never requires it).
        "gemini-3-flash-openrouter": OpenRouterModel(),
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
