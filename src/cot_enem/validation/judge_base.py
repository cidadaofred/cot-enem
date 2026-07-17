"""Shared implementation for structured boolean LLM judges."""

from typing import Any

from cot_enem.config import PromptDefinition
from cot_enem.generation.schema import JudgeDecision
from cot_enem.providers.base import LLMProvider, Message

JUDGE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "approved": {"type": "boolean"},
        "reasons": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["approved", "reasons"],
    "additionalProperties": False,
}


class StructuredJudge:
    def __init__(
        self,
        provider: LLMProvider,
        prompt: PromptDefinition,
        *,
        temperature: float = 0.0,
    ) -> None:
        self.provider = provider
        self.prompt = prompt
        self.temperature = temperature

    def _decide(self, user_content: str) -> JudgeDecision:
        messages: list[Message] = [
            {"role": "system", "content": self.prompt.system},
            {"role": "user", "content": user_content},
        ]
        response = self.provider.generate(
            messages, temperature=self.temperature, response_schema=JUDGE_SCHEMA
        )
        return JudgeDecision.model_validate(response.parsed)
