"""Judge logical and answer correctness of an improved chain."""

from cot_enem.config import PromptCatalog
from cot_enem.dataset.schema import NormalizedQuestion
from cot_enem.generation.prompting import render_question, render_steps
from cot_enem.generation.schema import JudgeDecision, SpecifiedCoT
from cot_enem.providers.base import LLMProvider
from cot_enem.validation.judge_base import StructuredJudge


class CorrectnessJudge(StructuredJudge):
    def __init__(self, provider: LLMProvider, prompts: PromptCatalog) -> None:
        super().__init__(provider, prompts.get("correctness_judge"))

    def judge(self, question: NormalizedQuestion, reasoning: SpecifiedCoT) -> JudgeDecision:
        content = self.prompt.user.format(
            question=render_question(question),
            gold_answer=question.gold_answer,
            reasoning=render_steps(reasoning.reasoning_steps),
        )
        return self._decide(content)
