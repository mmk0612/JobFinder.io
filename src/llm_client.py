"""
llm_client.py
-------------
Thin wrapper for the NVIDIA-hosted OpenAI-compatible chat API.
Sends prompts and returns plain text or parsed JSON.
"""

from __future__ import annotations

import json
import os
import random
import re
import threading
import time

import requests


# ── configuration ────────────────────────────────────────────────────────────

_DEFAULT_BASE_URL = os.environ.get("LLM_BASE_URL", "https://integrate.api.nvidia.com/v1").strip() or "https://integrate.api.nvidia.com/v1"
_DEFAULT_MODEL = os.environ.get("LLM_MODEL", "minimaxai/minimax-m2.5").strip() or "minimaxai/minimax-m2.5"
_MAX_RETRIES = int(os.environ.get("LLM_MAX_RETRIES", "5") or "5")
_RETRY_DELAY = float(os.environ.get("LLM_RETRY_DELAY_SECONDS", "1.5") or "1.5")
_RATE_LIMIT_RETRY_DELAY = float(os.environ.get("LLM_RATE_LIMIT_RETRY_DELAY_SECONDS", "3.0") or "3.0")
_MIN_REQUEST_INTERVAL = float(os.environ.get("LLM_MIN_REQUEST_INTERVAL_SECONDS", "1.5") or "1.5")
_REQUEST_TIMEOUT_SECONDS = float(os.environ.get("LLM_REQUEST_TIMEOUT_SECONDS", "180") or "180")
_TOTAL_TIMEOUT_SECONDS = float(os.environ.get("LLM_TOTAL_TIMEOUT_SECONDS", "180") or "180")
_DEFAULT_TOP_P = float(os.environ.get("LLM_TOP_P", "0.95") or "0.95")
_DEFAULT_MAX_TOKENS = int(os.environ.get("LLM_MAX_TOKENS", "8192") or "8192")
_THINKING_ENABLED = os.environ.get("LLM_THINKING", "true").strip().lower() in {"1", "true", "yes", "on"}
_SYSTEM_PROMPT = os.environ.get(
    "LLM_SYSTEM_PROMPT",
    "You are a structured API assistant. Return only the final answer with no reasoning trace, no <think> tags, and no hidden chain-of-thought.",
).strip()

_rate_limit_lock = threading.Lock()
_last_request_ts = 0.0


def _get_api_key() -> str:
    key = os.environ.get("NVIDIA_API_KEY", "").strip()
    if not key:
        raise EnvironmentError(
            "NVIDIA_API_KEY is not set. "
            "Add it to your .env file or export it in your shell."
        )
    return key


def _invoke_nvidia_chat(
    *,
    prompt: str,
    model: str,
    temperature: float,
    top_p: float,
    max_tokens: int,
    stream: bool,
    timeout_seconds: float,
) -> str:
    url = f"{_DEFAULT_BASE_URL.rstrip('/')}/chat/completions"
    headers = {
        "Authorization": f"Bearer {_get_api_key()}",
        "Accept": "text/event-stream" if stream else "application/json",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "stream": stream,
        "chat_template_kwargs": {"thinking": _THINKING_ENABLED},
    }

    response = requests.post(
        url,
        headers=headers,
        json=payload,
        stream=stream,
        timeout=timeout_seconds,
    )
    response.raise_for_status()

    if not stream:
        body = response.json()
        choices = body.get("choices") or []
        if not choices:
            return ""
        message = choices[0].get("message") or {}
        return _sanitize_model_text(str(message.get("content") or "").strip())

    parts: list[str] = []
    for raw_line in response.iter_lines(decode_unicode=True):
        if not raw_line:
            continue
        line = str(raw_line).strip()
        if not line.startswith("data:"):
            continue
        data = line[5:].strip()
        if data == "[DONE]":
            break
        try:
            chunk = json.loads(data)
        except json.JSONDecodeError:
            continue
        choices = chunk.get("choices") or []
        if not choices:
            continue
        delta = choices[0].get("delta") or {}
        content = delta.get("content")
        if content:
            parts.append(str(content))
    return _sanitize_model_text("".join(parts).strip())


def _throttle_request() -> None:
    global _last_request_ts
    if _MIN_REQUEST_INTERVAL <= 0:
        return
    with _rate_limit_lock:
        now = time.time()
        wait = _MIN_REQUEST_INTERVAL - (now - _last_request_ts)
        if wait > 0:
            time.sleep(wait)
        _last_request_ts = time.time()


# ── public API ───────────────────────────────────────────────────────────────

def call_llm(
    prompt: str,
    *,
    model: str = _DEFAULT_MODEL,
    temperature: float = 0.2,
    top_p: float = _DEFAULT_TOP_P,
    max_tokens: int = _DEFAULT_MAX_TOKENS,
    stream: bool = False,
) -> str:
    """Send a text prompt to the configured chat model and return raw response text."""
    started_at = time.monotonic()

    last_err: Exception | None = None
    for attempt in range(1, _MAX_RETRIES + 1):
        try:
            elapsed = time.monotonic() - started_at
            remaining = _TOTAL_TIMEOUT_SECONDS - elapsed
            if remaining <= 0:
                break
            request_timeout = max(1.0, min(_REQUEST_TIMEOUT_SECONDS, remaining))

            _throttle_request()
            return _invoke_nvidia_chat(
                prompt=prompt,
                model=model,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
                stream=stream,
                timeout_seconds=request_timeout,
            )
        except Exception as exc:
            last_err = exc
            if attempt < _MAX_RETRIES:
                err_text = str(exc)
                is_rate_limited = "429" in err_text or "Too Many Requests" in err_text or "rate limit" in err_text.lower()
                base_delay = _RATE_LIMIT_RETRY_DELAY if is_rate_limited else _RETRY_DELAY
                delay = (base_delay * (2 ** (attempt - 1))) + random.uniform(0.0, 0.35)
                elapsed = time.monotonic() - started_at
                remaining = _TOTAL_TIMEOUT_SECONDS - elapsed
                if remaining <= 0:
                    break
                delay = min(delay, max(0.0, remaining))
                time.sleep(delay)

    raise RuntimeError(
        f"LLM call failed after {_MAX_RETRIES} attempts or {_TOTAL_TIMEOUT_SECONDS:.0f}s: {last_err}"
    )


def call_llm_for_json(
    prompt: str,
    *,
    model: str = _DEFAULT_MODEL,
    temperature: float = 0.1,
    top_p: float = _DEFAULT_TOP_P,
    max_tokens: int = _DEFAULT_MAX_TOKENS,
) -> dict:
    """Call LLM and parse response as JSON (markdown fences are stripped)."""
    raw = call_llm(
        prompt,
        model=model,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
    )
    return _parse_json_response(raw)


# Backward-compat aliases (safe for stale imports)
def call_gemini(
    prompt: str,
    *,
    model: str = _DEFAULT_MODEL,
    temperature: float = 0.2,
    top_p: float = _DEFAULT_TOP_P,
    max_tokens: int = _DEFAULT_MAX_TOKENS,
    stream: bool = False,
) -> str:
    return call_llm(
        prompt,
        model=model,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        stream=stream,
    )


def call_gemini_for_json(
    prompt: str,
    *,
    model: str = _DEFAULT_MODEL,
    temperature: float = 0.1,
    top_p: float = _DEFAULT_TOP_P,
    max_tokens: int = _DEFAULT_MAX_TOKENS,
) -> dict:
    return call_llm_for_json(
        prompt,
        model=model,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
    )


# ── helpers ───────────────────────────────────────────────────────────────────

def _parse_json_response(text: str) -> dict:
    text = _sanitize_model_text(text)
    text = re.sub(r"^```(?:json)?\s*", "", text.strip(), flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text.strip())
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        candidate = _extract_json_object(text)
        if candidate:
            try:
                return json.loads(candidate)
            except json.JSONDecodeError:
                pass
        raise ValueError(
            f"LLM did not return valid JSON.\n"
            f"Response was:\n{text}\n\nError: {exc}"
        ) from exc


def _sanitize_model_text(text: str) -> str:
    cleaned = str(text or "")
    cleaned = re.sub(r"<think>.*?</think>", "", cleaned, flags=re.IGNORECASE | re.DOTALL)
    if cleaned.strip().lower().startswith("<think>") and "</think>" in cleaned.lower():
        close_idx = cleaned.lower().find("</think>")
        cleaned = cleaned[close_idx + len("</think>"):]
    return cleaned.strip()


def _extract_json_object(text: str) -> str | None:
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    return text[start : end + 1]
