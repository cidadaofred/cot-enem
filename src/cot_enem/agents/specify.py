"""Specify agent: improve reasoning while preserving the original item."""

from cot_enem.agents.base import EvolutionAgent
from cot_enem.config import PromptCatalog
from cot_enem.dataset.schema import NormalizedQuestion
from cot_enem.generation.initial_cot import INITIAL_COT_SCHEMA
from cot_enem.generation.prompting import render_question, render_steps
from cot_enem.generation.schema import InitialCoT, SpecifiedCoT
from cot_enem.providers.base import LLMProvider


class SpecifyAgent(EvolutionAgent[InitialCoT, SpecifiedCoT]):
    def __init__(
        self,
        provider: LLMProvider,
        prompts: PromptCatalog,
        *,
        temperature: float = 0.2,
    ) -> None:
        self.provider = provider
        self.prompt = prompts.get("specify")
        self.temperature = temperature

    def evolve(self, question: NormalizedQuestion, value: InitialCoT) -> SpecifiedCoT:
        response = self.provider.generate(
            [
                {"role": "system", "content": self.prompt.system},
                {
                    "role": "user",
                    "content": self.prompt.user.format(
                        question=render_question(question),
                        initial_cot=render_steps(value.reasoning_steps),
                    ),
                },
            ],
            temperature=self.temperature,
            response_schema=INITIAL_COT_SCHEMA,
        )
        return SpecifiedCoT.model_validate(response.parsed)
