"""Dependency-light client for OpenAI-compatible chat completion APIs."""

from collections.abc import Callable
import json
import os
import socket
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from cot_enem.providers.base import LLMProvider, LLMResponse, Message
from cot_enem.providers.errors import ProviderConfigurationError, ProviderRequestError
from cot_enem.providers.structured import parse_json_object, require_schema_keys
from cot_enem.utils.retry import retry_call

OpenFunction = Callable[..., Any]


class OpenAICompatibleProvider(LLMProvider):
    """Call an OpenAI-compatible `/chat/completions` endpoint."""

    def __init__(
        self,
        *,
        base_url: str | None = None,
        api_key: str | None = None,
        model: str | None = None,
        timeout: float = 60.0,
        max_attempts: int = 3,
        opener: OpenFunction = urlopen,
    ) -> None:
        self.base_url = (base_url or os.getenv("LLM_BASE_URL", "")).rstrip("/")
        self._api_key = api_key or os.getenv("LLM_API_KEY", "")
        self.model = model or os.getenv("LLM_MODEL", "")
        self.timeout = timeout
        self.max_attempts = max_attempts
        self._opener = opener
        if not self.base_url:
            raise ProviderConfigurationError("LLM_BASE_URL is required")
        if not self.model:
            raise ProviderConfigurationError("LLM_MODEL is required")

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(base_url={self.base_url!r}, "
            f"model={self.model!r}, timeout={self.timeout!r})"
        )

    def generate(
        self,
        messages: list[Message],
        temperature: float = 0.0,
        response_schema: dict[str, Any] | None = None,
    ) -> LLMResponse:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
        }
        if response_schema is not None:
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "cot_enem_response", "strict": True, "schema": response_schema},
            }
        raw = retry_call(
            lambda: self._request(payload),
            max_attempts=self.max_attempts,
            retry_on=(ProviderRequestError,),
        )
        try:
            choice = raw["choices"][0]
            content = choice["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise ProviderRequestError("provider returned an unexpected response shape") from exc
        parsed = parse_json_object(content) if response_schema is not None else None
        if parsed is not None:
            require_schema_keys(parsed, response_schema)
        usage = {
            key: int(value)
            for key, value in raw.get("usage", {}).items()
            if isinstance(value, int)
        }
        return LLMResponse(
            content=content,
            parsed=parsed,
            model=raw.get("model", self.model),
            finish_reason=choice.get("finish_reason"),
            usage=usage,
            raw=raw,
        )

    def _request(self, payload: dict[str, Any]) -> dict[str, Any]:
        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        request = Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        try:
            with self._opener(request, timeout=self.timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:
            raise ProviderRequestError(f"provider request failed with HTTP {exc.code}") from exc
        except (URLError, socket.timeout, TimeoutError) as exc:
            raise ProviderRequestError("provider request failed due to a network error") from exc
        except (UnicodeError, json.JSONDecodeError) as exc:
            raise ProviderRequestError("provider returned an invalid JSON response") from exc
