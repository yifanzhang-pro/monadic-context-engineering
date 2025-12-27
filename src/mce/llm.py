from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, TypedDict
from urllib import error, request


class ChatMessage(TypedDict):
    role: str
    content: str


@dataclass(frozen=True)
class OpenRouterConfig:
    api_key: str
    model: str = "x-ai/grok-4.1-fast"
    base_url: str = "https://openrouter.ai/api/v1"
    referer: str | None = None
    title: str | None = None
    timeout_s: float = 30.0


class OpenRouterClient:
    def __init__(self, config: OpenRouterConfig) -> None:
        self._config = config

    @classmethod
    def from_env(cls) -> OpenRouterClient:
        api_key = os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError("OPENROUTER_API_KEY is required to call OpenRouter.")

        model = os.environ.get("OPENROUTER_MODEL", "x-ai/grok-4.1-fast")
        base_url = os.environ.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
        referer = os.environ.get("OPENROUTER_REFERER")
        title = os.environ.get("OPENROUTER_TITLE")
        timeout_raw = os.environ.get("OPENROUTER_TIMEOUT_S")
        timeout_s = float(timeout_raw) if timeout_raw else 30.0

        return cls(
            OpenRouterConfig(
                api_key=api_key,
                model=model,
                base_url=base_url,
                referer=referer,
                title=title,
                timeout_s=timeout_s,
            )
        )

    def chat(
        self,
        messages: list[ChatMessage],
        temperature: float = 0.2,
        max_tokens: int | None = None,
    ) -> str:
        payload: dict[str, Any] = {
            "model": self._config.model,
            "messages": messages,
            "temperature": temperature,
        }
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens

        data = json.dumps(payload).encode("utf-8")
        url = f"{self._config.base_url}/chat/completions"
        req = request.Request(url, data=data, method="POST")
        req.add_header("Authorization", f"Bearer {self._config.api_key}")
        req.add_header("Content-Type", "application/json")
        if self._config.referer:
            req.add_header("HTTP-Referer", self._config.referer)
        if self._config.title:
            req.add_header("X-Title", self._config.title)

        try:
            with request.urlopen(req, timeout=self._config.timeout_s) as response:
                raw = response.read()
        except error.HTTPError as exc:
            body = exc.read().decode("utf-8", "replace")
            raise RuntimeError(f"OpenRouter error {exc.code}: {body}") from exc
        except error.URLError as exc:
            raise RuntimeError(f"OpenRouter request failed: {exc.reason}") from exc

        try:
            decoded = json.loads(raw.decode("utf-8"))
        except json.JSONDecodeError as exc:
            raise RuntimeError("OpenRouter returned invalid JSON.") from exc

        try:
            return decoded["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError(f"OpenRouter response missing content: {decoded}") from exc
