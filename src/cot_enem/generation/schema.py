"""Structured outputs used during reasoning generation."""

from pydantic import Field

from cot_enem.dataset.schema import AnswerKey, StrictModel


class InitialCoT(StrictModel):
    reasoning_steps: list[str] = Field(min_length=1)
    final_answer: AnswerKey


class SpecifiedCoT(StrictModel):
    reasoning_steps: list[str] = Field(min_length=1)
    final_answer: AnswerKey


class EvolvedQuestionCoT(StrictModel):
    """A rewritten multiple-choice question and its corresponding rationale."""

    header: str = ""
    statement: str = Field(min_length=1)
    alternatives: dict[AnswerKey, str]
    reasoning_steps: list[str] = Field(min_length=1)
    final_answer: AnswerKey


class JudgeDecision(StrictModel):
    approved: bool
    reasons: list[str] = Field(default_factory=list)
