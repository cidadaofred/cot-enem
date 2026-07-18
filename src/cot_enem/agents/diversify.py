"""Diversify: create a new question inspired by the input and its rationale."""

from cot_enem.agents.question_evolution import QuestionEvolutionAgent
from cot_enem.dataset.schema import Strategy


class DiversifyAgent(QuestionEvolutionAgent):
    strategy = Strategy.DIVERSIFY
