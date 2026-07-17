"""Validated contracts for source and evolved records."""
from datetime import datetime
from enum import StrEnum
from typing import Literal
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

AnswerKey = Literal["A", "B", "C", "D", "E"]

class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

class Annotations(StrictModel):
    image: bool = False
    EK: bool = False
    TC: bool = False
    IC: bool = False
    DS: bool = False
    MR: bool = False
    CE: bool = False

class NormalizedQuestion(StrictModel):
    id: str = Field(min_length=1)
    year: int | None = Field(default=None, ge=1998, le=2100)
    question_number: int | None = Field(default=None, ge=1)
    area: str | None = None
    header: str = ""
    statement: str = Field(min_length=1)
    alternatives: dict[AnswerKey, str]
    gold_answer: AnswerKey
    annotations: Annotations = Field(default_factory=Annotations)
    eligible: bool = True
    ineligibility_reasons: list[str] = Field(default_factory=list)

    @field_validator("alternatives")
    @classmethod
    def validate_alternatives(cls, value: dict[str, str]) -> dict[str, str]:
        if set(value) != set("ABCDE"):
            raise ValueError("alternatives must contain exactly A, B, C, D and E")
        texts = [text.strip() for text in value.values()]
        if any(not text for text in texts):
            raise ValueError("alternatives cannot be empty")
        if len({text.casefold() for text in texts}) != 5:
            raise ValueError("alternatives must be unique")
        return value

class Strategy(StrEnum):
    SPECIFY = "specify"
    COMPLICATE = "complicate"
    DIVERSIFY = "diversify"

class RecordStatus(StrEnum):
    ACCEPTED = "accepted"
    REJECTED = "rejected"
    PENDING = "pending"

class QuestionContent(StrictModel):
    header: str = ""
    statement: str = Field(min_length=1)
    alternatives: dict[AnswerKey, str]
    gold_answer: AnswerKey

class Reasoning(StrictModel):
    initial_cot: str
    improved_cot: str | None = None
    predicted_answer: AnswerKey

class GenerationMetadata(StrictModel):
    generator_model: str
    judge_model: str | None = None
    prompt_version: str
    temperature: float = Field(ge=0, le=2)
    timestamp: datetime
    execution_id: str
    attempts: int = Field(default=1, ge=1)

class ValidationResult(StrictModel):
    format_valid: bool = False
    answer_matches_gold: bool = False
    correctness_verified: bool = False
    evolution_success: bool = False
    enem_style_valid: bool = False
    accepted: bool = False
    rejection_reasons: list[str] = Field(default_factory=list)

class EvolvedRecord(StrictModel):
    id: str
    root_id: str
    parent_id: str
    generation: int = Field(ge=1)
    source_type: Literal["synthetic_evolved"] = "synthetic_evolved"
    strategy: Strategy
    question: QuestionContent
    reasoning: Reasoning
    generation_metadata: GenerationMetadata
    validation: ValidationResult
    status: RecordStatus = RecordStatus.PENDING

    @model_validator(mode="after")
    def status_matches_validation(self) -> "EvolvedRecord":
        if self.validation.accepted and self.status == RecordStatus.REJECTED:
            raise ValueError("accepted validation cannot have rejected status")
        return self
