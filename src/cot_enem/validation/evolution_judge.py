"""Judge whether Specify materially improved the initial reasoning."""

from cot_enem.config import PromptCatalog
from cot_enem.dataset.schema import NormalizedQuestion
from cot_enem.generation.prompting import render_question, render_steps
from cot_enem.generation.schema import InitialCoT, JudgeDecision, SpecifiedCoT
from cot_enem.providers.base import LLMProvider
from cot_enem.validation.judge_base import StructuredJudge


class EvolutionSuccessJudge(StructuredJudge):
    def __init__(self, provider: LLMProvider, prompts: PromptCatalog) -> None:
        super().__init__(provider, prompts.get("evolution_success_judge"))

    def judge(
        self,
        question: NormalizedQuestion,
        initial: InitialCoT,
        improved: SpecifiedCoT,
    ) -> JudgeDecision:
        content = self.prompt.user.format(
            question=render_question(question),
            initial_cot=render_steps(initial.reasoning_steps),
            improved_cot=render_steps(improved.reasoning_steps),
        )
        return self._decide(content)
