"""Dataset-level eligibility policies."""
from dataclasses import dataclass, field
from cot_enem.dataset.schema import NormalizedQuestion

@dataclass(slots=True)
class FilterReport:
    eligible: list[NormalizedQuestion] = field(default_factory=list)
    ineligible: list[NormalizedQuestion] = field(default_factory=list)
    duplicate_ids: set[str] = field(default_factory=set)

def partition_questions(questions: list[NormalizedQuestion]) -> FilterReport:
    report, seen = FilterReport(), set()
    for question in questions:
        if question.id in seen:
            report.duplicate_ids.add(question.id)
            question = question.model_copy(update={
                "eligible": False,
                "ineligibility_reasons": [*question.ineligibility_reasons, "duplicate_id"],
            })
        seen.add(question.id)
        (report.eligible if question.eligible else report.ineligible).append(question)
    return report
