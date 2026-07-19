"""Shared implementation for strategies that rewrite the question itself."""

from cot_enem.agents.base import EvolutionAgent
from cot_enem.config import PromptCatalog
from cot_enem.dataset.schema import NormalizedQuestion, Strategy
from cot_enem.generation.prompting import render_question_plain, render_steps
from cot_enem.generation.schema import EvolvedQuestionCoT, InitialCoT
from cot_enem.providers.base import LLMProvider

EVOLVED_QUESTION_SCHEMA = {
    "type": "object",
    "properties": {
        "header": {"type": "string"},
        "statement": {"type": "string", "minLength": 1},
        "alternatives": {
            "type": "object",
            "properties": {
                key: {"type": "string"} for key in "ABCDE"
            },
            "required": list("ABCDE"),
            "additionalProperties": False,
        },
        "reasoning_steps": {
            "type": "array",
            "items": {"type": "string"},
            "minItems": 1,
            "maxItems": 8,
        },
        "final_answer": {"type": "string", "enum": list("ABCDE")},
    },
    "required": [
        "header",
        "statement",
        "alternatives",
        "reasoning_steps",
        "final_answer",
    ],
    "additionalProperties": False,
}


class QuestionEvolutionAgent(EvolutionAgent[InitialCoT, EvolvedQuestionCoT]):
    """Base for Complicate and Diversify with a shared structured contract."""

    strategy: Strategy

    def __init__(
        self,
        provider: LLMProvider,
        prompts: PromptCatalog,
        *,
        temperature: float = 0.2,
    ) -> None:
        self.provider = provider
        self.prompt = prompts.get(self.strategy.value)
        self.temperature = temperature

    def evolve(
        self, question: NormalizedQuestion, value: InitialCoT
    ) -> EvolvedQuestionCoT:
        response = self.provider.generate(
            [
                {"role": "system", "content": self.prompt.system},
                {
                    "role": "user",
                    "content": self.prompt.user.format(
                        question=render_question_plain(question),
                        initial_cot=render_steps(value.reasoning_steps),
                    ),
                },
            ],
            temperature=self.temperature,
            response_schema=EVOLVED_QUESTION_SCHEMA,
        )
        return EvolvedQuestionCoT.model_validate(response.parsed)
