"""Hosted-LLM dependency parser via OpenRouter (OpenAI-compatible API).

OpenRouter exposes hundreds of hosted models behind one OpenAI-compatible
`POST /v1/chat/completions` endpoint, so this is a thin subclass of
`LMStudioModel`: same prompt construction, structured-output schema, fallback
defaults and tree repair — only the endpoint, the bearer-token auth and a bit
of OpenRouter-specific response handling differ.

The API key is read from the `OPENROUTER_API_KEY` environment variable (or
passed explicitly). It is never written to disk, logged, or baked into the
model `name`/cache slug. Keep it out of git — see the repo `.env` (gitignored).

Usage:

    OPENROUTER_API_KEY=sk-or-... .venv/bin/python -c "\
        from latinbench import Bench; \
        from latinbench.models.openrouter_llm import OpenRouterModel; \
        Bench().run(OpenRouterModel())"

Default model is `google/gemini-3-flash-preview` ($0.50/M input — within the
shared-key budget; structured output supported). Override with `model_id=`.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import requests

from .lmstudio_llm import LMStudioModel, DEPREL_LABELS


# OpenAI's strict structured-output validator (used by gpt-5-mini etc. via
# OpenRouter) is stricter than Gemini/LM Studio: every object must declare
# `additionalProperties: false`, and unsupported keywords like `minimum` are
# rejected outright. This schema satisfies that strict subset and is still
# accepted by the looser providers — so we send it for every OpenRouter model.
# (Head/id range validity is enforced downstream in _parse_one regardless.)
STRICT_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "tokens": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "id": {"type": "integer"},
                    "head": {"type": "integer"},
                    "deprel": {"type": "string", "enum": DEPREL_LABELS},
                },
                "required": ["id", "head", "deprel"],
            },
        },
    },
    "required": ["tokens"],
}


DEFAULT_HOST = "https://openrouter.ai/api"
DEFAULT_MODEL_ID = "google/gemini-3-flash-preview"
DEFAULT_REFERER = "https://github.com/tum/NLP-LLM-SS2026"
DEFAULT_TITLE = "latinbench"

# Transient HTTP statuses worth retrying (rate limit + upstream/gateway errors).
_RETRYABLE_STATUS = {429, 500, 502, 503, 504}


class OpenRouterModel(LMStudioModel):
    """Dependency parser backed by any OpenRouter-hosted chat model.

    Inherits the whole pipeline from `LMStudioModel` (prompt building, few-shot
    demonstrations, structured-output schema, per-token validity fallback and
    tree repair). Only `_call_llm` is overridden to add bearer auth, the
    OpenRouter provider routing flag, and lenient response parsing.
    """

    def __init__(
        self,
        model_id: str = DEFAULT_MODEL_ID,
        host: str = DEFAULT_HOST,
        api_key: str | None = None,
        num_workers: int = 4,   # gentler default than LM Studio: shared remote key
        referer: str = DEFAULT_REFERER,
        title: str = DEFAULT_TITLE,
        timeout: float = 180.0,
        max_retries: int = 4,
        backoff_base: float = 1.0,
        send_temperature: bool = True,
        reasoning_effort: str | None = None,
        **kwargs,
    ) -> None:
        super().__init__(model_id=model_id, host=host, num_workers=num_workers, **kwargs)
        # Resolve lazily so constructing the model (e.g. building the MODELS
        # registry) never fails when the key is absent — we only need it at
        # call time. The check fires loudly in `predict`, before any network.
        self.api_key = api_key or os.environ.get("OPENROUTER_API_KEY")
        self.referer = referer
        self.title = title
        self.timeout = timeout
        self.max_retries = max_retries
        self.backoff_base = backoff_base
        # Reasoning models (e.g. openai/gpt-5-mini) reject a non-default
        # `temperature`, so with provider.require_parameters they route to no
        # provider (HTTP 404). Set send_temperature=False to omit it. They also
        # spend output tokens on hidden reasoning — raise max_tokens and pass
        # reasoning_effort="low" to keep that bounded.
        self.send_temperature = send_temperature
        self.reasoning_effort = reasoning_effort

    def predict(self, test_path: Path, out_path: Path) -> None:
        if not self.api_key:
            raise RuntimeError(
                "OpenRouter API key missing. Set OPENROUTER_API_KEY (e.g. in the "
                "repo's gitignored .env) or pass api_key= to OpenRouterModel."
            )
        super().predict(test_path, out_path)

    def _headers(self) -> dict[str, str]:
        headers = {"Authorization": f"Bearer {self.api_key}"}
        # OpenRouter attribution headers (optional, but recommended).
        if self.referer:
            headers["HTTP-Referer"] = self.referer
        if self.title:
            headers["X-Title"] = self.title
        return headers

    def _call_llm(self, single: list) -> dict:
        body = {
            "model": self.model_id,
            "messages": self._build_messages(single),
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "ud_parse",
                    "strict": True,
                    "schema": STRICT_SCHEMA,
                },
            },
            # Route only to providers that actually honour the params we send
            # (notably response_format) — otherwise OpenRouter may silently
            # drop structured output and we'd get free-text back.
            "provider": {"require_parameters": True},
            "stream": False,
            "max_tokens": self.max_tokens,
        }
        if self.send_temperature:
            body["temperature"] = self.temperature
        if self.reasoning_effort:
            body["reasoning"] = {"effort": self.reasoning_effort}
        url = f"{self.host}/v1/chat/completions"
        headers = self._headers()

        # Retry transient failures (rate limits, gateway/upstream errors,
        # timeouts) with exponential backoff, honouring Retry-After. Without
        # this a single 429 during a several-hundred-sentence run would silently
        # collapse that whole sentence to right-branching fallback and quietly
        # bias the score downward. After retries are exhausted the error
        # propagates and `_parse_one` applies its per-token fallback.
        for attempt in range(self.max_retries + 1):
            try:
                r = requests.post(url, json=body, headers=headers, timeout=self.timeout)
            except (requests.Timeout, requests.ConnectionError):
                if attempt >= self.max_retries:
                    raise
                time.sleep(self._retry_delay(attempt, None))
                continue
            if r.status_code in _RETRYABLE_STATUS and attempt < self.max_retries:
                time.sleep(self._retry_delay(attempt, r.headers.get("Retry-After")))
                continue
            r.raise_for_status()
            return self._parse_response(r)
        raise RuntimeError("unreachable: retry loop exited without return/raise")

    @staticmethod
    def _parse_response(r: requests.Response) -> dict:
        data = r.json()
        # OpenRouter can return HTTP 200 with an error envelope.
        if isinstance(data, dict) and data.get("error"):
            raise ValueError(f"OpenRouter error: {data['error']}")
        choices = data.get("choices") if isinstance(data, dict) else None
        if not choices:
            raise ValueError(f"OpenRouter returned no choices: {data}")
        content = choices[0].get("message", {}).get("content")
        if not content:
            raise ValueError("OpenRouter returned empty message content")
        return _loads_lenient(content)

    def _retry_delay(self, attempt: int, retry_after: str | None) -> float:
        """Seconds to wait before the next attempt. Prefer the server's
        Retry-After header (capped) when present, else exponential backoff."""
        if retry_after:
            try:
                return min(float(retry_after), 60.0)
            except ValueError:
                pass
        return self.backoff_base * (2 ** attempt)


def _loads_lenient(content: str) -> dict:
    """Parse a JSON object from a model message, tolerating markdown fences.

    Structured output should give us a raw JSON object, but some providers still
    wrap it in ```json … ``` fences, add stray prose, or emit the bare token
    array instead of the `{"tokens": [...]}` wrapper. Strategy:

    1. strict parse;
    2. on failure, parse the first balanced `{ … }` block.

    The result is always coerced to a dict (a top-level list is wrapped as
    `{"tokens": list}`); anything else (scalars, garbage) raises ValueError so
    `_parse_one` applies its right-branching fallback rather than crashing the
    run on a later `.get("tokens")` — see lmstudio_llm._parse_one.
    """
    try:
        return _coerce_to_dict(json.loads(content))
    except ValueError:
        pass

    start = content.find("{")
    end = content.rfind("}")
    if start != -1 and end > start:
        try:
            return _coerce_to_dict(json.loads(content[start : end + 1]))
        except ValueError:
            pass
    raise ValueError(f"Could not parse JSON object from model output: {content[:200]!r}")


def _coerce_to_dict(obj) -> dict:
    """Normalise parsed JSON to the expected `{"tokens": [...]}` object shape.

    Dicts pass through; a bare list is assumed to be the token array; everything
    else (None, numbers, strings, bools) is rejected with ValueError.
    """
    if isinstance(obj, dict):
        return obj
    if isinstance(obj, list):
        return {"tokens": obj}
    raise ValueError(f"expected a JSON object, got {type(obj).__name__}")
