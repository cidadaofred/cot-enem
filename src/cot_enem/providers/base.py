"""Stable contracts shared by all LLM providers."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Literal, TypedDict

Role = Literal["system", "user", "assistant"]


class Message(TypedDict):
    role: Role
    content: str


@dataclass(frozen=True, slots=True)
class LLMResponse:
    """Normalized provider response with optional usage and raw metadata."""

    content: str
    parsed: dict[str, Any] | None = None
    model: str | None = None
    finish_reason: str | None = None
    usage: dict[str, int] = field(default_factory=dict)
    raw: dict[str, Any] = field(default_factory=dict, repr=False)


class LLMProvider(ABC):
    """Provider-neutral synchronous generation interface."""

    @abstractmethod
    def generate(
        self,
        messages: list[Message],
        temperature: float = 0.0,
        response_schema: dict[str, Any] | None = None,
    ) -> LLMResponse:
        raise NotImplementedError
