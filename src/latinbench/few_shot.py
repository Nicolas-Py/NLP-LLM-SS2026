"""Example pool for few-shot prompting of LLM-backed parsers.

Loads a CoNLL-U file of demonstration sentences and exposes deterministic
sampling. Used by `LMStudioModel(k_shot=k, ...)` to inject in-context
demonstrations into the chat history.

The default pool ships in `few_shot_examples.conllu` alongside this module.
"""
from __future__ import annotations

import random
from pathlib import Path

import conllu


DEFAULT_EXAMPLES_PATH = Path(__file__).parent / "few_shot_examples.conllu"


def _derive_tag(path: Path) -> str:
    """Short pool identifier used to keep prediction-cache slugs distinct per
    example source.

    The default pool gets the empty tag (so slugs stay `…-{k}shot`, leaving
    existing caches untouched); a `few_shot_examples_<x>.conllu` file → `<x>`
    (e.g. `…_perseus.conllu` → `"perseus"`); any other file → its stem.
    """
    stem = path.stem
    default_stem = DEFAULT_EXAMPLES_PATH.stem  # "few_shot_examples"
    if stem == default_stem:
        return ""
    prefix = default_stem + "_"
    if stem.startswith(prefix):
        return stem[len(prefix):]
    return stem


class ExamplePool:
    """A bag of demonstration sentences, sampled deterministically.

    Sentences are parsed once at construction. `sample(k, seed)` is pure
    in `(k, seed)` — same args, same returned sentences, no shared state.

    `tag` is a short identifier for the example source (derived from the
    filename unless passed explicitly); `LMStudioModel` appends it to its
    prediction-cache slug so runs off different pools never clobber each other.
    """

    def __init__(self, path: Path | None = None, tag: str | None = None) -> None:
        self.path = Path(path) if path else DEFAULT_EXAMPLES_PATH
        self.tag = tag if tag is not None else _derive_tag(self.path)
        self._sentences = conllu.parse(self.path.read_text())
        if not self._sentences:
            raise ValueError(f"Empty example pool: {self.path}")

    def __len__(self) -> int:
        return len(self._sentences)

    def sample(self, k: int, seed: int = 0) -> list[conllu.TokenList]:
        if k < 0:
            raise ValueError(f"k must be >= 0, got {k}")
        if k == 0:
            return []
        if k > len(self._sentences):
            raise ValueError(
                f"requested k={k} but pool only has {len(self._sentences)} "
                f"sentences at {self.path}"
            )
        return random.Random(seed).sample(self._sentences, k)
