from __future__ import annotations

import json
from unittest.mock import patch, MagicMock

import conllu
import pytest
import requests

from latinbench.models.lmstudio_llm import SYSTEM_PROMPT
from latinbench.models.openrouter_llm import (
    OpenRouterModel,
    DEFAULT_MODEL_ID,
    STRICT_SCHEMA,
    _coerce_to_dict,
    _loads_lenient,
)


def _openrouter_response(payload: dict) -> MagicMock:
    """MagicMock mimicking an OpenRouter (OpenAI-compatible) HTTP response."""
    r = MagicMock()
    r.raise_for_status = MagicMock(return_value=None)
    r.json = MagicMock(return_value={
        "choices": [{"message": {"content": json.dumps(payload)}}]
    })
    return r


def _raw_response(content: str) -> MagicMock:
    r = MagicMock()
    r.raise_for_status = MagicMock(return_value=None)
    r.json = MagicMock(return_value={"choices": [{"message": {"content": content}}]})
    return r


def _single():
    return [
        {"id": 1, "form": "a", "lemma": "_", "upos": "X", "feats": None},
        {"id": 2, "form": "b", "lemma": "_", "upos": "X", "feats": None},
    ]


# ---------- construction / key resolution ----------

def test_default_model_id_and_slug():
    m = OpenRouterModel(api_key="sk-test")
    assert m.model_id == DEFAULT_MODEL_ID
    # slug replaces "/" → "-"
    assert m.name == "google-gemini-3-flash-preview"


def test_host_default_points_at_openrouter():
    assert OpenRouterModel(api_key="sk-test").host == "https://openrouter.ai/api"


def test_k_shot_slug_suffix_inherited():
    m = OpenRouterModel(api_key="sk-test", k_shot=2)
    assert m.name == "google-gemini-3-flash-preview-2shot"


def test_explicit_api_key_wins_over_env(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-env")
    assert OpenRouterModel(api_key="sk-explicit").api_key == "sk-explicit"


def test_api_key_falls_back_to_env(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-env")
    assert OpenRouterModel().api_key == "sk-env"


def test_construction_without_key_does_not_raise(monkeypatch):
    """Building the model (e.g. the MODELS registry) must not require a key."""
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    m = OpenRouterModel()
    assert m.api_key is None


def test_predict_without_key_raises_clear_error(monkeypatch, tmp_path):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    m = OpenRouterModel()
    test_file = tmp_path / "t.conllu"
    test_file.write_text("1\ta\ta\tX\t_\t_\t_\t_\t_\t_\n")
    with pytest.raises(RuntimeError, match="OPENROUTER_API_KEY"):
        m.predict(test_file, tmp_path / "out.conllu")


# ---------- _call_llm HTTP wiring ----------

def test_call_llm_posts_to_openrouter_with_auth_and_schema():
    m = OpenRouterModel(api_key="sk-secret", model_id="google/gemini-3-flash-preview",
                        max_tokens=2048)
    payload = {"tokens": [
        {"id": 1, "head": 2, "deprel": "nsubj"},
        {"id": 2, "head": 0, "deprel": "root"},
    ]}
    with patch("requests.post", return_value=_openrouter_response(payload)) as post:
        out = m._call_llm(_single())

    args, kwargs = post.call_args
    assert args[0] == "https://openrouter.ai/api/v1/chat/completions"
    # bearer auth header carries the key
    assert kwargs["headers"]["Authorization"] == "Bearer sk-secret"
    body = kwargs["json"]
    assert body["model"] == "google/gemini-3-flash-preview"
    assert body["max_tokens"] == 2048
    assert body["stream"] is False
    # structured output preserved from LMStudioModel
    assert body["response_format"]["type"] == "json_schema"
    assert body["response_format"]["json_schema"]["strict"] is True
    assert body["response_format"]["json_schema"]["schema"] == STRICT_SCHEMA
    # OpenRouter-only: force providers that honour our params
    assert body["provider"] == {"require_parameters": True}
    # same system+user message shape as the parent
    assert [msg["role"] for msg in body["messages"]] == ["system", "user"]
    assert body["messages"][0]["content"] == SYSTEM_PROMPT
    assert out == payload


def test_call_llm_includes_temperature_and_no_reasoning_by_default():
    m = OpenRouterModel(api_key="sk", temperature=0.3)
    with patch("requests.post", return_value=_openrouter_response({"tokens": []})) as post:
        m._call_llm(_single())
    body = post.call_args.kwargs["json"]
    assert body["temperature"] == 0.3
    assert "reasoning" not in body


def test_call_llm_omits_temperature_for_reasoning_models():
    """gpt-5-mini & friends 404 if temperature is sent under require_parameters."""
    m = OpenRouterModel(api_key="sk", send_temperature=False, reasoning_effort="low")
    with patch("requests.post", return_value=_openrouter_response({"tokens": []})) as post:
        m._call_llm(_single())
    body = post.call_args.kwargs["json"]
    assert "temperature" not in body
    assert body["reasoning"] == {"effort": "low"}


def test_strict_schema_is_openai_compatible():
    """OpenAI strict mode requires additionalProperties:false on every object
    and rejects keywords like `minimum`. Guard both."""
    obj = STRICT_SCHEMA
    item = obj["properties"]["tokens"]["items"]
    assert obj["additionalProperties"] is False
    assert item["additionalProperties"] is False
    assert set(item["required"]) == {"id", "head", "deprel"}
    # no `minimum`/`maximum` anywhere (unsupported by OpenAI strict mode)
    import json as _json
    assert "minimum" not in _json.dumps(obj)
    assert "maximum" not in _json.dumps(obj)
    # deprel still constrained to the real label set
    assert "root" in item["properties"]["deprel"]["enum"]


def test_max_tokens_override_flows_to_body():
    m = OpenRouterModel(api_key="sk", max_tokens=6000)
    with patch("requests.post", return_value=_openrouter_response({"tokens": []})) as post:
        m._call_llm(_single())
    assert post.call_args.kwargs["json"]["max_tokens"] == 6000


def test_call_llm_omits_attribution_headers_when_blank():
    m = OpenRouterModel(api_key="sk", referer="", title="")
    with patch("requests.post", return_value=_openrouter_response({"tokens": []})) as post:
        m._call_llm(_single())
    headers = post.call_args.kwargs["headers"]
    assert "HTTP-Referer" not in headers and "X-Title" not in headers
    assert headers["Authorization"] == "Bearer sk"


def test_call_llm_parses_fenced_json():
    m = OpenRouterModel(api_key="sk")
    payload = {"tokens": [{"id": 1, "head": 0, "deprel": "root"}]}
    fenced = f"```json\n{json.dumps(payload)}\n```"
    with patch("requests.post", return_value=_raw_response(fenced)):
        out = m._call_llm(_single())
    assert out == payload


def test_call_llm_raises_on_error_envelope():
    m = OpenRouterModel(api_key="sk")
    r = MagicMock()
    r.raise_for_status = MagicMock(return_value=None)
    r.json = MagicMock(return_value={"error": {"message": "rate limited", "code": 429}})
    with patch("requests.post", return_value=r):
        with pytest.raises(ValueError, match="OpenRouter error"):
            m._call_llm(_single())


def test_call_llm_raises_on_missing_choices():
    m = OpenRouterModel(api_key="sk")
    r = MagicMock()
    r.raise_for_status = MagicMock(return_value=None)
    r.json = MagicMock(return_value={"choices": []})
    with patch("requests.post", return_value=r):
        with pytest.raises(ValueError, match="no choices"):
            m._call_llm(_single())


# ---------- _loads_lenient ----------

def test_loads_lenient_plain_json():
    assert _loads_lenient('{"a": 1}') == {"a": 1}


def test_loads_lenient_strips_code_fence():
    assert _loads_lenient('```json\n{"a": 1}\n```') == {"a": 1}


def test_loads_lenient_extracts_from_surrounding_prose():
    assert _loads_lenient('Sure! {"a": 1} done.') == {"a": 1}


def test_loads_lenient_raises_on_garbage():
    with pytest.raises(ValueError):
        _loads_lenient("no json here")


def test_loads_lenient_wraps_bare_token_array():
    """A provider that emits the bare array instead of {"tokens": [...]} should
    be recovered, not crash the run."""
    arr = '[{"id": 1, "head": 0, "deprel": "root"}]'
    assert _loads_lenient(arr) == {"tokens": [{"id": 1, "head": 0, "deprel": "root"}]}


def test_loads_lenient_rejects_scalar_json():
    """Valid-but-scalar JSON (null / number / string) must raise ValueError so
    _parse_one falls back, rather than returning a non-dict that later crashes
    on .get('tokens')."""
    for scalar in ("null", "42", '"hi"', "true"):
        with pytest.raises(ValueError):
            _loads_lenient(scalar)


def test_coerce_to_dict_passthrough_wrap_and_reject():
    assert _coerce_to_dict({"tokens": []}) == {"tokens": []}
    assert _coerce_to_dict([{"id": 1}]) == {"tokens": [{"id": 1}]}
    for bad in (None, 1, "x", True):
        with pytest.raises(ValueError):
            _coerce_to_dict(bad)


def test_call_llm_never_returns_non_dict_for_array_response():
    """End-to-end guard: a 200 response whose content is a bare JSON array must
    come back as a dict (wrapped), never a list."""
    m = OpenRouterModel(api_key="sk")
    with patch("requests.post",
               return_value=_raw_response('[{"id": 1, "head": 0, "deprel": "root"}]')):
        out = m._call_llm(_single())
    assert isinstance(out, dict)
    assert out == {"tokens": [{"id": 1, "head": 0, "deprel": "root"}]}


# ---------- retry / backoff ----------

def _status_response(status: int, headers: dict | None = None) -> MagicMock:
    r = MagicMock()
    r.status_code = status
    r.headers = headers or {}
    r.raise_for_status = MagicMock(
        side_effect=requests.HTTPError(f"{status}") if status >= 400 else None
    )
    return r


def test_call_llm_retries_then_succeeds_on_429(monkeypatch):
    m = OpenRouterModel(api_key="sk", max_retries=3, backoff_base=0.0)
    sleeps = []
    monkeypatch.setattr("time.sleep", lambda s: sleeps.append(s))
    payload = {"tokens": [{"id": 1, "head": 0, "deprel": "root"}]}
    ok = _openrouter_response(payload)
    ok.status_code = 200
    seq = [_status_response(429, {"Retry-After": "0"}),
           _status_response(503),
           ok]
    with patch("requests.post", side_effect=seq) as post:
        out = m._call_llm(_single())
    assert out == payload
    assert post.call_count == 3
    assert len(sleeps) == 2  # two retries waited


def test_call_llm_raises_after_exhausting_retries(monkeypatch):
    m = OpenRouterModel(api_key="sk", max_retries=2, backoff_base=0.0)
    monkeypatch.setattr("time.sleep", lambda s: None)
    seq = [_status_response(429), _status_response(429), _status_response(429)]
    with patch("requests.post", side_effect=seq) as post:
        with pytest.raises(requests.HTTPError):
            m._call_llm(_single())
    assert post.call_count == 3  # initial + 2 retries


def test_call_llm_retries_on_timeout_then_succeeds(monkeypatch):
    m = OpenRouterModel(api_key="sk", max_retries=2, backoff_base=0.0)
    monkeypatch.setattr("time.sleep", lambda s: None)
    payload = {"tokens": [{"id": 1, "head": 0, "deprel": "root"}]}
    ok = _openrouter_response(payload)
    ok.status_code = 200
    with patch("requests.post", side_effect=[requests.Timeout("slow"), ok]) as post:
        out = m._call_llm(_single())
    assert out == payload
    assert post.call_count == 2


def test_call_llm_timeout_propagates_after_retries(monkeypatch):
    m = OpenRouterModel(api_key="sk", max_retries=1, backoff_base=0.0)
    monkeypatch.setattr("time.sleep", lambda s: None)
    with patch("requests.post", side_effect=requests.Timeout("slow")):
        with pytest.raises(requests.Timeout):
            m._call_llm(_single())


def test_retry_delay_prefers_retry_after_header():
    m = OpenRouterModel(api_key="sk", backoff_base=1.0)
    assert m._retry_delay(3, "5") == 5.0          # header wins over backoff
    assert m._retry_delay(0, None) == 1.0         # backoff_base * 2**0
    assert m._retry_delay(2, None) == 4.0         # backoff_base * 2**2
    assert m._retry_delay(0, "garbage") == 1.0    # bad header -> backoff
    assert m._retry_delay(0, "999") == 60.0       # capped


# ---------- inherited pipeline still works end-to-end ----------

def test_predict_writes_predictions_via_inherited_pipeline(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk")
    test_file = tmp_path / "test.conllu"
    test_file.write_text(
        "# sent_id = s1\n# text = Marcus amat\n"
        "1\tMarcus\tmarcus\tPROPN\t_\tCase=Nom\t_\t_\t_\t_\n"
        "2\tamat\tamo\tVERB\t_\tMood=Ind\t_\t_\t_\t_\n"
    )
    out_file = tmp_path / "pred.conllu"
    payload = {"tokens": [
        {"id": 1, "head": 2, "deprel": "nsubj"},
        {"id": 2, "head": 0, "deprel": "root"},
    ]}
    m = OpenRouterModel(api_key="sk")
    m.num_workers = 1
    with patch("requests.post", return_value=_openrouter_response(payload)):
        m.predict(test_file, out_file)

    parsed = conllu.parse(out_file.read_text())[0]
    assert [(t["head"], t["deprel"]) for t in parsed] == [(2, "nsubj"), (0, "root")]
