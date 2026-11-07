"""Thin LLM clients with disk caching. Keys come only from environment variables."""
from __future__ import annotations

import os
import time
from dataclasses import dataclass

import httpx

from .cache import JsonlCache, cache_key


@dataclass
class LLMResult:
    text: str
    input_tokens: int
    output_tokens: int
    latency_s: float
    cached: bool


class OfflineError(RuntimeError):
    """Raised when a call is needed but not in cache and live calls are disabled."""


def _call_anthropic(model, prompt, max_tokens, temperature):
    import anthropic

    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise OfflineError("ANTHROPIC_API_KEY is not set")
    client = anthropic.Anthropic()
    t0 = time.perf_counter()
    msg = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
        messages=[{"role": "user", "content": prompt}],
    )
    dt = time.perf_counter() - t0
    text = "".join(b.text for b in msg.content if getattr(b, "type", "") == "text")
    return text, msg.usage.input_tokens, msg.usage.output_tokens, dt


def _call_ollama(model, prompt, max_tokens, temperature):
    host = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
    t0 = time.perf_counter()
    try:
        r = httpx.post(
            f"{host}/api/generate",
            json={"model": model, "prompt": prompt, "stream": False,
                  "options": {"temperature": temperature, "num_predict": max_tokens, "seed": 0}},
            timeout=300,
        )
    except httpx.ConnectError as e:
        raise OfflineError(f"Ollama not reachable at {host}") from e
    r.raise_for_status()
    j = r.json()
    dt = time.perf_counter() - t0
    return j["response"], j.get("prompt_eval_count", 0), j.get("eval_count", 0), dt


BHARATCODE_URL = "https://bharatcode.ai/api/model/v1/chat/completions"


def _verify():
    try:  # certifi bundle avoids stale Windows CA stores; verification is never disabled
        import certifi
        return certifi.where()
    except ImportError:
        return True


def _call_bharatcode(model, prompt, max_tokens, temperature):
    key = os.environ.get("BHARATCODE_API_KEY")
    if not key:
        raise OfflineError("BHARATCODE_API_KEY is not set")
    t0 = time.perf_counter()
    for attempt in range(4):
        try:
            r = httpx.post(BHARATCODE_URL, headers={"Authorization": f"Bearer {key}"},
                           json={"model": model, "max_tokens": max_tokens,
                                 "temperature": temperature,
                                 "messages": [{"role": "user", "content": prompt}]},
                           timeout=180, verify=_verify())
            if r.status_code in (429, 500, 502, 503, 504) and attempt < 3:
                time.sleep(2 ** attempt * 3)
                continue
            r.raise_for_status()
            break
        except httpx.TransportError:
            if attempt == 3:
                raise
            time.sleep(2 ** attempt * 3)
    j = r.json()
    dt = time.perf_counter() - t0
    u = j.get("usage") or {}
    text = j["choices"][0]["message"]["content"] or ""
    return text, u.get("prompt_tokens", 0), u.get("completion_tokens", 0), dt


BACKENDS = {"anthropic": _call_anthropic, "ollama": _call_ollama,
            "bharatcode": _call_bharatcode}


def complete(cache: JsonlCache, provider: str, model: str, prompt: str, max_tokens: int,
             temperature: float = 0.0, offline: bool = False) -> LLMResult:
    key = cache_key(provider=provider, model=model, prompt=prompt,
                    max_tokens=max_tokens, temperature=temperature)
    hit = cache.get(key)
    if hit is not None:
        return LLMResult(hit["text"], hit["input_tokens"], hit["output_tokens"],
                         hit["latency_s"], cached=True)
    if offline:
        raise OfflineError(f"cache miss for {provider}:{model} in offline mode")
    text, tin, tout, dt = BACKENDS[provider](model, prompt, max_tokens, temperature)
    cache.put(key, {"provider": provider, "model": model, "text": text,
                    "input_tokens": tin, "output_tokens": tout, "latency_s": round(dt, 3)})
    return LLMResult(text, tin, tout, dt, cached=False)


def is_cached(cache: JsonlCache, provider, model, prompt, max_tokens, temperature=0.0) -> bool:
    return cache_key(provider=provider, model=model, prompt=prompt,
                     max_tokens=max_tokens, temperature=temperature) in cache


def pmap(fn, xs, workers: int = 6) -> list:
    """Order-preserving parallel map for I/O-bound API calls."""
    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(max_workers=workers) as ex:
        return list(ex.map(fn, xs))
