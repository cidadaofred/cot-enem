"""Programmatic comparison between predicted and official alternatives."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class AnswerValidation:
    matches: bool
    reason: str | None = None


class AnswerValidator:
    def validate(self, predicted: str, gold: str) -> AnswerValidation:
        predicted_key, gold_key = predicted.strip().upper(), gold.strip().upper()
        if predicted_key not in set("ABCDE"):
            return AnswerValidation(False, "predicted_answer_invalid")
        if predicted_key != gold_key:
            return AnswerValidation(False, "initial_answer_mismatch")
        return AnswerValidation(True)
