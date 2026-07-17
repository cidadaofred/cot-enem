"""Initial chain-of-thought generation without exposing the official answer."""

from typing import Any

from cot_enem.config import PromptCatalog
from cot_enem.dataset.schema import NormalizedQuestion
from cot_enem.generation.prompting import render_question
from cot_enem.generation.schema import InitialCoT
from cot_enem.providers.base import LLMProvider

INITIAL_COT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "reasoning_steps": {
            "type": "array",
            "items": {"type": "string"},
            "minItems": 1,
        },
        "final_answer": {"type": "string", "enum": list("ABCDE")},
    },
    "required": ["reasoning_steps", "final_answer"],
    "additionalProperties": False,
}


class InitialCoTGenerator:
    def __init__(
        self,
        provider: LLMProvider,
        prompts: PromptCatalog,
        *,
        temperature: float = 0.2,
    ) -> None:
        self.provider = provider
        self.prompt = prompts.get("initial_cot")
        self.temperature = temperature

    def generate(self, question: NormalizedQuestion) -> InitialCoT:
        question_text = render_question(question, include_gold=False)
        response = self.provider.generate(
            [
                {"role": "system", "content": self.prompt.system},
                {"role": "user", "content": self.prompt.user.format(question=question_text)},
            ],
            temperature=self.temperature,
            response_schema=INITIAL_COT_SCHEMA,
        )
        return InitialCoT.model_validate(response.parsed)
