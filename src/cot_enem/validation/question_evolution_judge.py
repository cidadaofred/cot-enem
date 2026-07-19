"""Evolution-success judge for Complicate and Diversify candidates."""

from cot_enem.config import PromptCatalog
from cot_enem.dataset.schema import NormalizedQuestion, Strategy
from cot_enem.generation.prompting import render_question, render_question_plain, render_steps
from cot_enem.generation.schema import EvolvedQuestionCoT, InitialCoT, JudgeDecision
from cot_enem.providers.base import LLMProvider
from cot_enem.validation.judge_base import StructuredJudge

OBJECTIVES = {
    Strategy.COMPLICATE: (
        "a questão evoluída deve exigir raciocínio mais profundo, mais condições "
        "ou mais etapas do que a entrada"
    ),
    Strategy.DIVERSIFY: (
        "a questão evoluída deve mudar substancialmente cenário ou núcleo temático "
        "sem ser mera paráfrase"
    ),
}


class QuestionEvolutionSuccessJudge(StructuredJudge):
    def __init__(
        self,
        provider: LLMProvider,
        prompts: PromptCatalog,
        *,
        temperature: float = 0.0,
    ) -> None:
        super().__init__(
            provider,
            prompts.get("question_evolution_success_judge"),
            temperature=temperature,
        )

    def judge(
        self,
        strategy: Strategy,
        source: NormalizedQuestion,
        initial: InitialCoT,
        evolved: EvolvedQuestionCoT,
        evolved_question: NormalizedQuestion,
    ) -> JudgeDecision:
        if strategy not in OBJECTIVES:
            raise ValueError(f"unsupported question evolution strategy: {strategy}")
        content = self.prompt.user.format(
            strategy=strategy.value,
            objective=OBJECTIVES[strategy],
            input_question=render_question_plain(source),
            initial_cot=render_steps(initial.reasoning_steps),
            evolved_question=render_question(evolved_question),
            evolved_cot=render_steps(evolved.reasoning_steps),
        )
        return self._decide(content)
