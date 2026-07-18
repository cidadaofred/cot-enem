"""Complicate: increase question difficulty and regenerate its rationale."""

from cot_enem.agents.question_evolution import QuestionEvolutionAgent
from cot_enem.dataset.schema import Strategy


class ComplicateAgent(QuestionEvolutionAgent):
    strategy = Strategy.COMPLICATE
