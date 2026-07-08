from __future__ import annotations

import json
import os
import time
import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import requests

try:
    from dotenv import load_dotenv

    load_dotenv(".env.local")
except Exception:
    pass


def _env_int(name: str) -> Optional[int]:
    v = os.environ.get(name, "").strip()
    if not v:
        return None
    try:
        return int(float(v))
    except Exception:
        return None


def _env_float(name: str) -> Optional[float]:
    v = os.environ.get(name, "").strip()
    if not v:
        return None
    try:
        return float(v)
    except Exception:
        return None


@dataclass
class VLMClient:
    api_base: str
    api_key: str
    model: str
    timeout_s: int = 120
    max_retries: int = 3
    temperature: float = 0.7
    top_p: float = 0.8

    _session: requests.Session = field(default_factory=requests.Session, init=False, repr=False)

    def __post_init__(self):
        self.api_base = (self.api_base or "").rstrip("/")

        if not self.api_key:
            self.api_key = os.environ.get("DASHSCOPE_API_KEY", "") or os.environ.get("OPENAI_API_KEY", "")
        if not self.api_key:
            raise ValueError("api_key is empty (set --api-key or env DASHSCOPE_API_KEY/OPENAI_API_KEY).")

        env_timeout = _env_float("VLM_HTTP_TIMEOUT")
        env_retries = _env_int("VLM_MAX_RETRIES")

        if env_timeout is not None and self.timeout_s == 120:
            self.timeout_s = int(max(1.0, env_timeout))
        if env_retries is not None and self.max_retries == 3:
            self.max_retries = int(max(1, env_retries))

        self._connect_timeout_s = _env_float("VLM_CONNECT_TIMEOUT") or 10.0

        self._backoff_cap_s = _env_float("VLM_BACKOFF_CAP") or 8.0

    @property
    def endpoint(self) -> str:
        return f"{self.api_base}/chat/completions"

    @property
    def cache_tag(self) -> str:
        return f"{self.model}::t{self.temperature:.2f}::p{self.top_p:.2f}"

    def _timeout_tuple(self) -> Tuple[float, float]:
        return (float(self._connect_timeout_s), float(self.timeout_s))

    def invoke(
        self,
        content_parts: List[Dict[str, Any]],
        *,
        temperature: float | None = None,
        top_p: float | None = None,
        max_tokens: int = 512,
    ) -> str:
        temp = self.temperature if temperature is None else float(temperature)
        tp = self.top_p if top_p is None else float(top_p)
        debug = os.environ.get("VLM_DEBUG", "").strip().lower() in ("1", "true", "yes", "on")

        payload: Dict[str, Any] = {
            "model": self.model,
            "temperature": temp,
            "top_p": tp,
            "max_tokens": max_tokens,
            "messages": [{"role": "user", "content": content_parts}],
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        last_err: Optional[Exception] = None

        for attempt in range(1, self.max_retries + 1):
            t0 = time.perf_counter()
            try:
                if debug:
                    kinds = [str(p.get("type", "?")) for p in content_parts if isinstance(p, dict)]
                    print(
                        f"[VLM] start model={self.model} attempt={attempt}/{self.max_retries} "
                        f"parts={kinds} max_tokens={max_tokens}",
                        flush=True,
                    )
                r = self._session.post(
                    self.endpoint,
                    headers=headers,
                    json=payload,                 
                    timeout=self._timeout_tuple(), 
                )

                if r.status_code == 429 or 500 <= r.status_code < 600:
                    raise RuntimeError(f"HTTP {r.status_code}: {r.text[:300]}")

                if r.status_code != 200:
                    raise RuntimeError(f"HTTP {r.status_code}: {r.text[:500]}")

                obj = r.json()
                content = obj["choices"][0]["message"]["content"]
                if debug:
                    dt = time.perf_counter() - t0
                    print(f"[VLM] ok model={self.model} attempt={attempt} elapsed={dt:.1f}s", flush=True)
                return str(content) if content is not None else ""

            except Exception as e:
                last_err = e
                if debug:
                    dt = time.perf_counter() - t0
                    print(f"[VLM] fail model={self.model} attempt={attempt} elapsed={dt:.1f}s err={e}", flush=True)

                backoff = min(self._backoff_cap_s, 0.6 * (2 ** (attempt - 1)))
                backoff += random.random() * 0.2
                time.sleep(backoff)
                continue

        raise RuntimeError(
            f"VLM invoke failed after {self.max_retries} retries "
            f"(timeout_s={self.timeout_s}, connect_timeout_s={self._connect_timeout_s}): {last_err}"
        )

