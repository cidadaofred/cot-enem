"""Deterministic provider for tests and offline pipeline development."""

from collections.abc import Callable, Iterable
import json
from typing import Any

from cot_enem.providers.base import LLMProvider, LLMResponse, Message
from cot_enem.providers.structured import parse_json_object, require_schema_keys

MockValue = str | dict[str, Any] | LLMResponse


class MockLLMProvider(LLMProvider):
    """Return queued values or invoke a deterministic responder."""

    def __init__(
        self,
        responses: Iterable[MockValue] | None = None,
        *,
        responder: Callable[[list[Message]], MockValue] | None = None,
        model: str = "mock-llm",
    ) -> None:
        self._responses = iter(responses or [])
        self._responder = responder
        self.model = model
        self.calls: list[dict[str, Any]] = []

    def generate(
        self,
        messages: list[Message],
        temperature: float = 0.0,
        response_schema: dict[str, Any] | None = None,
    ) -> LLMResponse:
        self.calls.append(
            {
                "messages": messages,
                "temperature": temperature,
                "response_schema": response_schema,
            }
        )
        if self._responder:
            value = self._responder(messages)
        else:
            try:
                value = next(self._responses)
            except StopIteration as exc:
                raise RuntimeError("MockLLMProvider has no response configured") from exc

        if isinstance(value, LLMResponse):
            return value
        content = (
            json.dumps(value, ensure_ascii=False) if isinstance(value, dict) else str(value)
        )
        parsed = parse_json_object(content) if response_schema is not None else None
        if parsed is not None:
            require_schema_keys(parsed, response_schema)
        return LLMResponse(content=content, parsed=parsed, model=self.model, finish_reason="stop")
