"""Validation for a generated reasoning object."""

from dataclasses import dataclass, field

from cot_enem.generation.schema import InitialCoT, SpecifiedCoT


@dataclass(frozen=True, slots=True)
class FormatValidation:
    valid: bool
    reasons: list[str] = field(default_factory=list)


class FormatValidator:
    def validate(self, value: InitialCoT | SpecifiedCoT) -> FormatValidation:
        reasons = []
        if not value.reasoning_steps or any(not step.strip() for step in value.reasoning_steps):
            reasons.append("reasoning_steps_invalid")
        if value.final_answer not in set("ABCDE"):
            reasons.append("predicted_answer_invalid")
        return FormatValidation(not reasons, reasons)
