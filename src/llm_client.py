"""Thin LLM client wrapper — prompts, retries, cache, and rate limits live here."""
from __future__ import annotations

import hashlib
import json
import os
import threading
import time
from pathlib import Path

from dotenv import load_dotenv

from src.config import (
    LLM_BASE_BACKOFF_SECONDS,
    LLM_MAX_BACKOFF_SECONDS,
    LLM_MAX_RETRIES,
    LLM_MAX_TOKENS,
    LLM_MIN_COMPLETION_CHARS,
    LLM_MIN_REQUEST_INTERVAL,
    LLM_MODEL,
)

load_dotenv()

CACHE_DIR = Path(__file__).resolve().parents[1] / "data" / "processed" / "llm_cache"
_use_cache = True
_key_index = 0
_key_lock = threading.Lock()
_clients: dict[str, object] = {}
_clients_lock = threading.Lock()
_exhausted_until: dict[str, float] = {}


class EmptyCompletionError(RuntimeError):
    """Raised when the model returns blank or truncated stub text."""


class RequestIntervalLimiter:
    """Serialize API starts so concurrent workers do not burst past free-tier TPM."""

    def __init__(self, min_interval_seconds: float = LLM_MIN_REQUEST_INTERVAL) -> None:
        self._lock = threading.Lock()
        self._last_call = 0.0
        self._min_interval = min_interval_seconds

    def wait(self) -> None:
        """Block until the minimum interval since the last wait() has elapsed."""
        with self._lock:
            elapsed = time.monotonic() - self._last_call
            if elapsed < self._min_interval:
                time.sleep(self._min_interval - elapsed)
            self._last_call = time.monotonic()


_limiter = RequestIntervalLimiter()


def set_use_cache(enabled: bool) -> None:
    """Enable or disable disk cache for subsequent complete() calls."""
    global _use_cache
    _use_cache = enabled


def reset_key_rotation() -> None:
    """Reset failover index and exhaustion timers (tests / fresh eval runs)."""
    global _key_index
    with _key_lock:
        _key_index = 0
        _exhausted_until.clear()


def api_keys() -> list[str]:
    """Return configured Groq keys (primary then optional backups), de-duped."""
    keys: list[str] = []
    for name in ("GROQ_API_KEY", "GROQ_API_KEY_2", "GROQ_API_KEY_3"):
        value = os.getenv(name, "").strip()
        if value and value not in keys:
            keys.append(value)
    return keys


def is_available() -> bool:
    """Return True when at least one Groq API key is configured."""
    return bool(api_keys())


def cache_key(system: str, user: str) -> str:
    """Stable hash for (model, system, user) used as the cache filename stem."""
    raw = f"{LLM_MODEL}\n{system}\n{user}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def normalize_completion(content: str | None) -> str:
    """Reject blank/None/too-short completions so callers can retry."""
    if content is None:
        raise EmptyCompletionError("completion content was None")
    text = content.strip()
    if len(text) < LLM_MIN_COMPLETION_CHARS:
        raise EmptyCompletionError(
            f"completion too short ({len(text)} < {LLM_MIN_COMPLETION_CHARS})"
        )
    return text


def _read_cache(system: str, user: str) -> str | None:
    """Return a cached response string, or None on miss/corrupt entry."""
    path = CACHE_DIR / f"{cache_key(system, user)}.json"
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    response = payload.get("response")
    return response if isinstance(response, str) and response.strip() else None


def _write_cache(system: str, user: str, response: str) -> None:
    """Persist a successful completion for reuse across eval runs."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = CACHE_DIR / f"{cache_key(system, user)}.json"
    path.write_text(
        json.dumps({"model": LLM_MODEL, "response": response}, ensure_ascii=False),
        encoding="utf-8",
    )


def _cooldown_seconds(error: Exception) -> float:
    """Parse a short cooldown from Groq errors; default a few minutes for TPD."""
    text = str(error).lower()
    if "tokens per day" in text or "tpd" in text:
        return 180.0
    return 30.0


def _mark_key_exhausted(api_key: str, seconds: float) -> None:
    """Temporarily skip a key after quota / hard rate limits."""
    with _key_lock:
        _exhausted_until[api_key] = time.monotonic() + seconds


def _active_api_key() -> str:
    """Return a non-exhausted key when possible."""
    global _key_index
    keys = api_keys()
    if not keys:
        raise RuntimeError("GROQ_API_KEY is not set")
    now = time.monotonic()
    with _key_lock:
        for offset in range(len(keys)):
            index = (_key_index + offset) % len(keys)
            candidate = keys[index]
            if _exhausted_until.get(candidate, 0.0) <= now:
                _key_index = index
                return candidate
        return keys[_key_index % len(keys)]


def _rotate_api_key() -> bool:
    """Advance to the next configured key; False when there is only one."""
    keys = api_keys()
    if len(keys) < 2:
        return False
    with _key_lock:
        global _key_index
        _key_index = (_key_index + 1) % len(keys)
    return True


def _client(api_key: str):
    """Lazy-create a Groq client for one API key."""
    with _clients_lock:
        cached = _clients.get(api_key)
        if cached is not None:
            return cached
        from groq import Groq

        client = Groq(api_key=api_key)
        _clients[api_key] = client
        return client


def _is_retryable(error: Exception) -> bool:
    """True for rate limits, empty stubs, and transient connection failures."""
    if isinstance(error, EmptyCompletionError):
        return True
    name = type(error).__name__
    text = str(error).lower()
    if "RateLimit" in name or "APIConnection" in name:
        return True
    if "rate_limit" in text or "429" in text:
        return True
    status = getattr(error, "status_code", None)
    return status in {408, 429, 500, 502, 503, 504}


def _is_rate_limit(error: Exception) -> bool:
    """True when the failure is a quota / 429 style rate limit."""
    text = str(error).lower()
    if "rate_limit" in text or "429" in text or "tokens per day" in text:
        return True
    return getattr(error, "status_code", None) == 429


def _request_completion(system: str, user: str, *, api_key: str) -> str:
    """One Groq chat completion on a specific key (no retry / cache)."""
    response = _client(api_key).chat.completions.create(
        model=LLM_MODEL,
        max_tokens=LLM_MAX_TOKENS,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    )
    return normalize_completion(response.choices[0].message.content)


def complete(system: str, user: str) -> str:
    """Call the configured chat model; use disk cache and fail-fast retries."""
    if not is_available():
        raise RuntimeError("GROQ_API_KEY is not set")
    if _use_cache:
        cached = _read_cache(system, user)
        if cached is not None:
            return cached

    keys = api_keys()
    max_attempts = LLM_MAX_RETRIES * max(1, len(keys))
    delay = LLM_BASE_BACKOFF_SECONDS
    last_error: Exception | None = None
    for attempt in range(max_attempts):
        api_key = _active_api_key()
        try:
            _limiter.wait()
            text = _request_completion(system, user, api_key=api_key)
            if _use_cache:
                _write_cache(system, user, text)
            return text
        except Exception as error:  # noqa: BLE001 — classify then retry/raise
            last_error = error
            if not _is_retryable(error):
                raise
            if _is_rate_limit(error):
                _mark_key_exhausted(api_key, _cooldown_seconds(error))
                rotated = _rotate_api_key()
                # Immediate retry on the other key — do not burn backoff on failover.
                if rotated and attempt < max_attempts - 1:
                    continue
            if attempt == max_attempts - 1:
                break
            time.sleep(delay)
            delay = min(delay * 2, LLM_MAX_BACKOFF_SECONDS)
    raise RuntimeError(f"Groq complete failed after retries: {last_error}")
