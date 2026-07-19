"""Prompt rendering helpers shared by generation, agents, and judges."""

import html
import json
import re
import xml.etree.ElementTree as ET

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


def markup_to_plain_text(value: str) -> str:
    """Convert common HTML/MathML fragments to copy-safe plain mathematical text."""

    if "<" not in value:
        return html.unescape(value)

    def local_name(element: ET.Element) -> str:
        return element.tag.rsplit("}", 1)[-1].casefold()

    def render(element: ET.Element) -> str:
        tag = local_name(element)
        children = list(element)
        rendered = [render(child) for child in children]
        text = element.text or ""
        if tag == "semantics":
            body = next(
                (
                    render(child)
                    for child in children
                    if local_name(child) not in {"annotation", "annotation-xml"}
                ),
                "",
            )
        elif tag == "mfrac" and len(rendered) >= 2:
            body = f"({rendered[0]})/({rendered[1]})"
        elif tag == "msup" and len(rendered) >= 2:
            body = f"{rendered[0]}^({rendered[1]})"
        elif tag == "msub" and len(rendered) >= 2:
            body = f"{rendered[0]}_({rendered[1]})"
        elif tag == "msubsup" and len(rendered) >= 3:
            body = f"{rendered[0]}_({rendered[1]})^({rendered[2]})"
        elif tag == "msqrt":
            body = f"sqrt({''.join(rendered)})"
        elif tag == "mroot" and len(rendered) >= 2:
            body = f"root({rendered[0]}, {rendered[1]})"
        elif tag == "mfenced":
            body = f"({''.join(rendered)})"
        elif tag in {"annotation", "annotation-xml"}:
            body = ""
        else:
            body = text + "".join(rendered)
        return body + (element.tail or "")

    decoded = html.unescape(value)
    try:
        root = ET.fromstring(f"<root>{decoded}</root>")
        plain = render(root)
    except ET.ParseError:
        plain = re.sub(r"<[^>]+>", " ", decoded)
    return re.sub(r"\s+", " ", plain).strip()


def render_question_plain(
    question: NormalizedQuestion, *, include_gold: bool = False
) -> str:
    """Render a question after removing markup models tend to copy into invalid JSON."""

    payload = {
        "header": markup_to_plain_text(question.header),
        "statement": markup_to_plain_text(question.statement),
        "alternatives": {
            key: markup_to_plain_text(text)
            for key, text in question.alternatives.items()
        },
    }
    if include_gold:
        payload["gold_answer"] = question.gold_answer
    return json.dumps(payload, ensure_ascii=False, indent=2)


def render_steps(steps: list[str]) -> str:
    return "\n".join(f"{index}. {step}" for index, step in enumerate(steps, 1))
