"""Unit tests for LLM prompt-caching logic (Bedrock cache points + Gemini
explicit caching). These exercise pure logic only — no network, no real
provider SDK calls — by driving the helpers with fakes/monkeypatch.
"""
import asyncio
from types import SimpleNamespace

import pytest

from app.ai.llm.clients import bedrock_client as bc
from app.ai.llm.clients import google_client as gc


# --------------------------------------------------------------------------- #
# Bedrock: cache-point markers + converse_stream fallback                      #
# --------------------------------------------------------------------------- #
def _is_cached(kwargs) -> bool:
    """A request carries cache points iff the _CACHE_POINT marker is appended
    to the system block or the tool list."""
    if bc._CACHE_POINT in (kwargs.get("system") or []):
        return True
    tool_cfg = kwargs.get("toolConfig") or {}
    return bc._CACHE_POINT in (tool_cfg.get("tools") or [])


class _FakeBedrock:
    """Records whether each converse_stream call was the cached or plain variant
    and can be told to reject cached requests like an unsupported model."""

    def __init__(self, reject_cache: bool = False):
        self.reject_cache = reject_cache
        self.calls: list[str] = []

    def converse_stream(self, **kwargs):
        cached = _is_cached(kwargs)
        self.calls.append("cached" if cached else "plain")
        if cached and self.reject_cache:
            raise Exception("ValidationException: cachePoint is not supported for this model")
        return f"stream-{'cached' if cached else 'plain'}"


def _client(reject_cache=False) -> bc.BedrockClient:
    # Bypass __init__ (which builds a real boto3 client); the fallback helper
    # only touches self.client.
    obj = bc.BedrockClient.__new__(bc.BedrockClient)
    obj.client = _FakeBedrock(reject_cache=reject_cache)
    return obj


def _kwargs(model_id="anthropic.claude-3-5-sonnet"):
    plain = {
        "modelId": model_id,
        "system": [{"text": "you are a planner"}],
        "toolConfig": {"tools": [{"toolSpec": {"name": "t"}}]},
        "messages": [{"role": "user", "content": [{"text": "hi"}]}],
    }
    cached = bc.BedrockClient._with_cache_points(plain)
    return cached, plain


@pytest.fixture(autouse=True)
def _clear_unsupported():
    bc._CACHE_POINT_UNSUPPORTED_MODELS.clear()
    yield
    bc._CACHE_POINT_UNSUPPORTED_MODELS.clear()


def test_with_cache_points_marks_prefix_without_mutating_original():
    cached, plain = _kwargs()
    # original untouched
    assert plain["system"] == [{"text": "you are a planner"}]
    assert plain["toolConfig"]["tools"] == [{"toolSpec": {"name": "t"}}]
    # cached has the marker appended to both system and tools
    assert cached["system"][-1] == bc._CACHE_POINT
    assert cached["toolConfig"]["tools"][-1] == bc._CACHE_POINT


def test_fallback_prefers_cached_request():
    client = _client()
    cached, plain = _kwargs()
    out = client._converse_stream_with_fallback(cached, plain)
    assert out == "stream-cached"
    assert client.client.calls == ["cached"]


def test_fallback_retries_plain_and_memoizes_on_cache_rejection():
    client = _client(reject_cache=True)
    cached, plain = _kwargs(model_id="meta.llama-no-cache")
    out = client._converse_stream_with_fallback(cached, plain)
    # fell back to the plain request after the cached one was rejected
    assert out == "stream-plain"
    assert client.client.calls == ["cached", "plain"]
    # the model is now memoized as cache-unsupported
    assert "meta.llama-no-cache" in bc._CACHE_POINT_UNSUPPORTED_MODELS


def test_memoized_model_skips_cached_attempt_entirely():
    bc._CACHE_POINT_UNSUPPORTED_MODELS.add("meta.llama-no-cache")
    client = _client(reject_cache=True)
    cached, plain = _kwargs(model_id="meta.llama-no-cache")
    out = client._converse_stream_with_fallback(cached, plain)
    # goes straight to plain — never even tries the cached variant
    assert out == "stream-plain"
    assert client.client.calls == ["plain"]


def test_fallback_propagates_non_cache_errors():
    class _Boom:
        def converse_stream(self, **kwargs):
            raise Exception("ThrottlingException: rate exceeded")

    client = bc.BedrockClient.__new__(bc.BedrockClient)
    client.client = _Boom()
    cached, plain = _kwargs()
    with pytest.raises(Exception, match="ThrottlingException"):
        client._converse_stream_with_fallback(cached, plain)


def test_no_cached_kwargs_uses_plain():
    client = _client()
    _, plain = _kwargs()
    out = client._converse_stream_with_fallback(None, plain)
    assert out == "stream-plain"
    assert client.client.calls == ["plain"]


# --------------------------------------------------------------------------- #
# Gemini: model-aware cache token floor                                         #
# --------------------------------------------------------------------------- #
def test_cache_min_tokens_is_model_aware(monkeypatch):
    monkeypatch.delenv("BOW_GEMINI_CACHE_MIN_TOKENS", raising=False)
    assert gc._cache_min_tokens("gemini-3-pro-preview") == 4096
    assert gc._cache_min_tokens("models/gemini-3-flash") == 4096
    assert gc._cache_min_tokens("gemini-2.5-flash") == 2048
    assert gc._cache_min_tokens("gemini-1.5-pro") == 2048
    assert gc._cache_min_tokens("") == 2048


def test_cache_min_tokens_env_override_wins(monkeypatch):
    monkeypatch.setenv("BOW_GEMINI_CACHE_MIN_TOKENS", "1000")
    # override applies regardless of model family
    assert gc._cache_min_tokens("gemini-3-pro-preview") == 1000
    assert gc._cache_min_tokens("gemini-2.5-flash") == 1000


def test_cache_min_tokens_bad_override_falls_back_to_model_floor(monkeypatch):
    monkeypatch.setenv("BOW_GEMINI_CACHE_MIN_TOKENS", "not-an-int")
    assert gc._cache_min_tokens("gemini-3-pro-preview") == 4096
    assert gc._cache_min_tokens("gemini-2.5-flash") == 2048


# --------------------------------------------------------------------------- #
# Gemini: cache signature is namespaced by credential/project                  #
# --------------------------------------------------------------------------- #
def test_signature_is_credential_scoped():
    sig = gc._GeminiCacheManager._signature
    base = sig("cred-A", "gemini-2.5-flash", "sys", "tools")
    # same inputs -> stable
    assert base == sig("cred-A", "gemini-2.5-flash", "sys", "tools")
    # different credential -> different cache namespace (no cross-project reuse)
    assert base != sig("cred-B", "gemini-2.5-flash", "sys", "tools")
    # each content component still participates in the hash
    assert base != sig("cred-A", "gemini-3-pro", "sys", "tools")
    assert base != sig("cred-A", "gemini-2.5-flash", "other-sys", "tools")
    assert base != sig("cred-A", "gemini-2.5-flash", "sys", "other-tools")


# --------------------------------------------------------------------------- #
# Gemini: get_or_create returns creation tokens on create, 0 on hit            #
# --------------------------------------------------------------------------- #
class _FakeCaches:
    def __init__(self, creation_tokens):
        self.created = 0
        self._creation_tokens = creation_tokens

    def create(self, **kwargs):
        self.created += 1
        return SimpleNamespace(
            name=f"cachedContents/{self.created}",
            usage_metadata=SimpleNamespace(total_token_count=self._creation_tokens),
        )


class _FakeGenaiClient:
    def __init__(self, creation_tokens=5000):
        self.caches = _FakeCaches(creation_tokens)


def test_get_or_create_reports_creation_tokens_then_zero_on_hit(monkeypatch):
    monkeypatch.setenv("BOW_GEMINI_EXPLICIT_CACHE", "1")
    monkeypatch.delenv("BOW_GEMINI_CACHE_MIN_TOKENS", raising=False)
    mgr = gc._GeminiCacheManager()
    client = _FakeGenaiClient(creation_tokens=5000)

    async def run():
        common = dict(
            client=client, cred_id="cred-A", model_id="gemini-2.5-flash",
            system="sys", tools=None, tools_payload="tools",
            estimated_tokens=8000,  # above the 2048 floor
        )
        first = await mgr.get_or_create(**common)
        second = await mgr.get_or_create(**common)
        return first, second

    (name1, created1), (name2, created2) = asyncio.run(run())
    assert name1 == "cachedContents/1"
    assert created1 == 5000              # tokens billed to populate the cache
    assert name2 == "cachedContents/1"   # served from the in-process registry
    assert created2 == 0                 # a hit creates nothing
    assert client.caches.created == 1    # create() called exactly once


def test_get_or_create_skips_below_token_floor(monkeypatch):
    monkeypatch.setenv("BOW_GEMINI_EXPLICIT_CACHE", "1")
    monkeypatch.delenv("BOW_GEMINI_CACHE_MIN_TOKENS", raising=False)
    mgr = gc._GeminiCacheManager()
    client = _FakeGenaiClient()

    async def run():
        return await mgr.get_or_create(
            client=client, cred_id="cred-A", model_id="gemini-3-pro-preview",
            system="sys", tools=None, tools_payload="tools",
            estimated_tokens=3000,  # below the gemini-3 floor of 4096
        )

    name, created = asyncio.run(run())
    assert name is None and created == 0
    assert client.caches.created == 0  # never paid a create round-trip


def test_get_or_create_disabled_returns_none(monkeypatch):
    monkeypatch.setenv("BOW_GEMINI_EXPLICIT_CACHE", "0")
    mgr = gc._GeminiCacheManager()
    client = _FakeGenaiClient()

    async def run():
        return await mgr.get_or_create(
            client=client, cred_id="cred-A", model_id="gemini-2.5-flash",
            system="sys", tools=None, tools_payload="tools", estimated_tokens=9000,
        )

    name, created = asyncio.run(run())
    assert name is None and created == 0
    assert client.caches.created == 0
