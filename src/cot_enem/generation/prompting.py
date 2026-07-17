"""Prompt rendering helpers shared by generation, agents, and judges."""

import json

from cot_enem.dataset.schema import NormalizedQuestion


def render_question(question: NormalizedQuestion, *, include_gold: bool = False) -> str:
    payload = {
        "header": question.header,
        "statement": question.statement,
        "alternatives": question.alternatives,
    }
    if include_gold:
        payload["gold_answer"] = question.gold_answer
    return json.dumps(payload, ensure_ascii=False, indent=2)


def render_steps(steps: list[str]) -> str:
    return "\n".join(f"{index}. {step}" for index, step in enumerate(steps, 1))
