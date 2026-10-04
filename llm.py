"""Anthropic client for the routing experiments.

Every response is cached on disk under cache/ keyed by a hash of
(model, seed, temperature, prompt). A run that stalls on a rate limit therefore
resumes for free, and re-running an unchanged experiment costs nothing — the
same discipline papers/dare/bench/run_all.py uses.

The key is read from scholarforge/.env (gitignored) and never logged.
Endpoint is api.anthropic.com directly; verified 2026-10-03.
"""
from __future__ import annotations

import hashlib
import json
import os
import random
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

HERE = Path(__file__).resolve().parent
CACHE = HERE / "cache"
# Searched in order, so the module works both inside the paper repository and
# as a standalone artefact checkout.
ENV_CANDIDATES = (HERE / ".env", HERE.parents[1] / ".env")

API_URL = "https://api.anthropic.com/v1/messages"
API_VERSION = "2023-06-01"

MAIN_MODEL = "claude-haiku-4-5-20251001"
TRANSFER_MODEL = "claude-sonnet-4-5-20250929"

# Anthropic list prices, USD per million tokens, recorded 2026-10-03. Used only
# to report experiment cost in the paper; not billed against.
PRICES = {
    MAIN_MODEL: {"in": 1.00, "out": 5.00},
    TRANSFER_MODEL: {"in": 3.00, "out": 15.00},
}


def _load_key() -> str:
    key = os.environ.get("ANTHROPIC_API_KEY")
    if key:
        return key
    for env_file in ENV_CANDIDATES:
        if env_file.exists():
            for line in env_file.read_text().splitlines():
                if line.startswith("ANTHROPIC_API_KEY="):
                    return line.split("=", 1)[1].strip().strip("\"'")
    raise RuntimeError(
        "ANTHROPIC_API_KEY not in the environment or in "
        + " / ".join(str(c) for c in ENV_CANDIDATES)
    )


@dataclass
class Usage:
    calls: int = 0
    cached: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    retries: int = 0
    seconds: float = 0.0
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def cost_usd(self, model: str) -> float:
        p = PRICES.get(model)
        if not p:
            return 0.0
        return self.input_tokens / 1e6 * p["in"] + self.output_tokens / 1e6 * p["out"]

    def summary(self, model: str) -> dict:
        return {
            "calls": self.calls,
            "cache_hits": self.cached,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "retries": self.retries,
            "wall_seconds": round(self.seconds, 1),
            "est_cost_usd": round(self.cost_usd(model), 4),
        }


class LLM:
    """Minimal Anthropic Messages client with disk cache and backoff."""

    def __init__(
        self,
        model: str = MAIN_MODEL,
        temperature: float = 0.0,
        max_tokens: int = 512,
        max_retries: int = 6,
    ):
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_retries = max_retries
        self._key = _load_key()
        self.usage = Usage()
        CACHE.mkdir(exist_ok=True)

    def _cache_path(self, system: str, prompt: str, seed: int) -> Path:
        h = hashlib.sha256(
            json.dumps(
                [self.model, self.temperature, self.max_tokens, seed, system, prompt]
            ).encode()
        ).hexdigest()
        return CACHE / f"{h}.json"

    def complete(self, prompt: str, system: str = "", seed: int = 0) -> str:
        """Return the model's text reply, from cache when available.

        `seed` does not reach the API — Anthropic exposes no seed parameter. It
        varies the cache key and, for temperature > 0, yields independent
        samples. At temperature 0 the seeds are near-replicates, which is what
        we want for measuring decode variance rather than prompt variance.
        """
        path = self._cache_path(system, prompt, seed)
        if path.exists():
            self.usage.cached += 1
            return json.loads(path.read_text())["text"]

        body = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "messages": [{"role": "user", "content": prompt}],
        }
        if system:
            body["system"] = system

        delay = 2.0
        last_err = None
        for attempt in range(self.max_retries):
            started = time.time()
            try:
                req = urllib.request.Request(
                    API_URL,
                    data=json.dumps(body).encode(),
                    headers={
                        "x-api-key": self._key,
                        "anthropic-version": API_VERSION,
                        "content-type": "application/json",
                    },
                )
                with urllib.request.urlopen(req, timeout=120) as resp:
                    data = json.loads(resp.read())
                text = "".join(b.get("text", "") for b in data.get("content", []))
                u = data.get("usage", {})
                with self.usage._lock:
                    self.usage.calls += 1
                    self.usage.input_tokens += u.get("input_tokens", 0)
                    self.usage.output_tokens += u.get("output_tokens", 0)
                    self.usage.seconds += time.time() - started
                path.write_text(json.dumps({"text": text, "usage": u}))
                return text
            except urllib.error.HTTPError as e:
                last_err = e
                # 429 rate limit and 5xx are transient; 4xx otherwise is not.
                if e.code != 429 and e.code < 500:
                    raise RuntimeError(f"Anthropic HTTP {e.code}: {e.read()[:300]!r}") from e
            except Exception as e:  # network blips
                last_err = e
            with self.usage._lock:
                self.usage.retries += 1
            time.sleep(delay + random.uniform(0, 1.0))
            delay = min(delay * 2, 60.0)
        raise RuntimeError(f"Anthropic call failed after {self.max_retries} attempts: {last_err}")


def extract_json(text: str) -> dict | None:
    """Pull the first JSON object out of a model reply.

    Models sometimes wrap JSON in prose or a fenced block. A parse failure is
    recorded by the caller as a router error, never silently treated as an
    abstention — abstaining is a decision the router must actually make.
    """
    text = text.strip()
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    for i, ch in enumerate(text[start:], start):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(text[start : i + 1])
                except json.JSONDecodeError:
                    return None
    return None
