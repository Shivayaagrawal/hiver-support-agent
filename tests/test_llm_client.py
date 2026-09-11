"""Tests for LLM client cache, empty guards, key failover, and fail-fast retries."""
from __future__ import annotations

import json
import time

import pytest

from src import llm_client
from src.llm_client import (
    EmptyCompletionError,
    RequestIntervalLimiter,
    normalize_completion,
)


def test_normalize_completion_rejects_empty():
    with pytest.raises(EmptyCompletionError):
        normalize_completion("")
    with pytest.raises(EmptyCompletionError):
        normalize_completion(None)
    with pytest.raises(EmptyCompletionError):
        normalize_completion("   ")


def test_normalize_completion_rejects_too_short():
    with pytest.raises(EmptyCompletionError):
        normalize_completion("Refund")


def test_normalize_completion_accepts_normal_reply():
    text = "Thanks for reaching out. Which iOS version are you on?"
    assert normalize_completion(text) == text


def test_cache_key_is_stable_for_same_inputs():
    first = llm_client.cache_key("system", "user message")
    second = llm_client.cache_key("system", "user message")
    other = llm_client.cache_key("system", "other message")
    assert first == second
    assert first != other


def test_api_keys_reads_primary_and_secondary(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "key-one")
    monkeypatch.setenv("GROQ_API_KEY_2", "key-two")
    monkeypatch.setenv("GROQ_API_KEY_3", "key-three")
    assert llm_client.api_keys() == ["key-one", "key-two", "key-three"]


def test_complete_rotates_to_second_key_on_429(tmp_path, monkeypatch):
    monkeypatch.setattr(llm_client, "CACHE_DIR", tmp_path)
    monkeypatch.setattr(llm_client, "_use_cache", False)
    monkeypatch.setattr(llm_client, "_limiter", RequestIntervalLimiter(0.0))
    monkeypatch.setenv("GROQ_API_KEY", "key-one")
    monkeypatch.setenv("GROQ_API_KEY_2", "key-two")
    llm_client.reset_key_rotation()
    calls: list[str] = []

    def fake_request(system: str, user: str, *, api_key: str) -> str:
        calls.append(api_key)
        if api_key == "key-one":
            error = RuntimeError("rate_limit 429 tokens per day")
            error.status_code = 429  # type: ignore[attr-defined]
            raise error
        return "Second key reply that is long enough."

    monkeypatch.setattr(llm_client, "_request_completion", fake_request)
    monkeypatch.setattr(llm_client.time, "sleep", lambda _seconds: None)
    text = llm_client.complete("sys", "user")
    assert "Second key" in text
    assert calls[0] == "key-one"
    assert "key-two" in calls
    # Primary should be marked exhausted so the next pick prefers secondary.
    assert llm_client._active_api_key() == "key-two"


def test_complete_returns_disk_cache_without_api_call(tmp_path, monkeypatch):
    monkeypatch.setattr(llm_client, "CACHE_DIR", tmp_path)
    monkeypatch.setattr(llm_client, "is_available", lambda: True)
    monkeypatch.setattr(llm_client, "_use_cache", True)
    cached = "Cached reply that is definitely long enough."
    key = llm_client.cache_key("sys", "user")
    (tmp_path / f"{key}.json").write_text(
        json.dumps({"response": cached}),
        encoding="utf-8",
    )
    calls: list[int] = []

    def boom(_system: str, _user: str, *, api_key: str) -> str:
        calls.append(1)
        raise AssertionError("API should not be called on cache hit")

    monkeypatch.setattr(llm_client, "_request_completion", boom)
    assert llm_client.complete("sys", "user") == cached
    assert calls == []


def test_complete_writes_cache_after_api_call(tmp_path, monkeypatch):
    monkeypatch.setattr(llm_client, "CACHE_DIR", tmp_path)
    monkeypatch.setattr(llm_client, "is_available", lambda: True)
    monkeypatch.setattr(llm_client, "_use_cache", True)
    monkeypatch.setenv("GROQ_API_KEY", "only-key")
    monkeypatch.delenv("GROQ_API_KEY_2", raising=False)
    monkeypatch.delenv("GROQ_API_KEY_3", raising=False)
    llm_client.reset_key_rotation()
    monkeypatch.setattr(
        llm_client,
        "_request_completion",
        lambda _system, _user, *, api_key: "Fresh reply that is long enough here.",
    )
    monkeypatch.setattr(llm_client, "_limiter", RequestIntervalLimiter(0.0))
    text = llm_client.complete("sys", "user")
    assert "Fresh reply" in text
    key = llm_client.cache_key("sys", "user")
    payload = json.loads((tmp_path / f"{key}.json").read_text(encoding="utf-8"))
    assert payload["response"] == text


def test_complete_skips_cache_when_disabled(tmp_path, monkeypatch):
    monkeypatch.setattr(llm_client, "CACHE_DIR", tmp_path)
    monkeypatch.setattr(llm_client, "is_available", lambda: True)
    monkeypatch.setattr(llm_client, "_use_cache", False)
    monkeypatch.setenv("GROQ_API_KEY", "only-key")
    monkeypatch.delenv("GROQ_API_KEY_2", raising=False)
    monkeypatch.delenv("GROQ_API_KEY_3", raising=False)
    llm_client.reset_key_rotation()
    key = llm_client.cache_key("sys", "user")
    (tmp_path / f"{key}.json").write_text(
        json.dumps({"response": "Stale cached reply that is long enough."}),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        llm_client,
        "_request_completion",
        lambda _system, _user, *, api_key: "Live reply that is long enough here.",
    )
    monkeypatch.setattr(llm_client, "_limiter", RequestIntervalLimiter(0.0))
    assert llm_client.complete("sys", "user") == "Live reply that is long enough here."


def test_request_interval_limiter_spaces_calls():
    limiter = RequestIntervalLimiter(0.05)
    start = time.monotonic()
    limiter.wait()
    limiter.wait()
    assert time.monotonic() - start >= 0.05


def test_complete_fails_fast_after_capped_retries(monkeypatch):
    monkeypatch.setattr(llm_client, "is_available", lambda: True)
    monkeypatch.setattr(llm_client, "_use_cache", False)
    monkeypatch.setattr(llm_client, "_limiter", RequestIntervalLimiter(0.0))
    monkeypatch.setenv("GROQ_API_KEY", "only-key")
    monkeypatch.delenv("GROQ_API_KEY_2", raising=False)
    monkeypatch.delenv("GROQ_API_KEY_3", raising=False)
    llm_client.reset_key_rotation()
    sleeps: list[float] = []

    def always_429(_system: str, _user: str, *, api_key: str) -> str:
        error = RuntimeError("rate_limit 429")
        error.status_code = 429  # type: ignore[attr-defined]
        raise error

    monkeypatch.setattr(llm_client, "_request_completion", always_429)
    monkeypatch.setattr(llm_client.time, "sleep", sleeps.append)
    with pytest.raises(RuntimeError, match="after retries"):
        llm_client.complete("sys", "user")
    assert len(sleeps) == llm_client.LLM_MAX_RETRIES - 1
    assert max(sleeps) <= llm_client.LLM_MAX_BACKOFF_SECONDS + 1e-9
