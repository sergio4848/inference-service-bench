"""A thin gateway to an OpenAI-compatible chat-completions endpoint that measures what matters for
serving LLMs: time to first token, total latency and generation speed. Streams from the upstream
so TTFT is real, returns one JSON document to the caller.
"""

from __future__ import annotations

import json
import os
import time

import httpx

from infer_service.config import Settings
from infer_service.observability import LLM_TOKENS_PER_S, LLM_TTFT


class LLMGateway:
    def __init__(self, settings: Settings):
        self._base = settings.llm_base_url.rstrip("/")
        self._key = settings.llm_api_key or os.environ.get("OPENAI_API_KEY")
        self._model = settings.llm_model
        self._timeout = settings.llm_timeout_s

    @property
    def configured(self) -> bool:
        return bool(self._key)

    async def complete(self, prompt: str, *, max_tokens: int = 128, model: str | None = None) -> dict:
        payload = {
            "model": model or self._model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        headers = {"authorization": f"Bearer {self._key}", "content-type": "application/json"}
        started = time.perf_counter()
        first_token_at: float | None = None
        chunks: list[str] = []
        usage: dict = {}
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            async with client.stream("POST", f"{self._base}/chat/completions", json=payload,
                                     headers=headers) as resp:
                resp.raise_for_status()
                async for line in resp.aiter_lines():
                    if not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if data == "[DONE]":
                        break
                    event = json.loads(data)
                    if event.get("usage"):
                        usage = event["usage"]
                    for choice in event.get("choices", []):
                        delta = choice.get("delta", {}).get("content")
                        if delta:
                            if first_token_at is None:
                                first_token_at = time.perf_counter()
                            chunks.append(delta)
        finished = time.perf_counter()
        text = "".join(chunks)
        completion_tokens = usage.get("completion_tokens") or max(1, len(text.split()))
        ttft = (first_token_at or finished) - started
        gen_seconds = max(finished - (first_token_at or started), 1e-6)
        tokens_per_s = completion_tokens / gen_seconds
        LLM_TTFT.observe(ttft)
        LLM_TOKENS_PER_S.observe(tokens_per_s)
        return {
            "model": model or self._model,
            "text": text,
            "completion_tokens": completion_tokens,
            "prompt_tokens": usage.get("prompt_tokens"),
            "ttft_ms": round(ttft * 1000, 1),
            "total_ms": round((finished - started) * 1000, 1),
            "tokens_per_second": round(tokens_per_s, 1),
        }
