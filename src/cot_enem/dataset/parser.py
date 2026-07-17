"""Defensive XML parser for historical ENEM exports."""
from dataclasses import dataclass
from pathlib import Path
import re
import xml.etree.ElementTree as ET

@dataclass(slots=True)
class RawQuestion:
    attributes: dict[str, str]
    header: str
    statement: str
    alternatives: dict[str, str]
    gold_answer: str | None
    source_index: int

def _name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1].lower()

def _text(element: ET.Element | None) -> str:
    return " ".join("".join(element.itertext()).split()) if element is not None else ""

def _first(element: ET.Element, names: set[str]) -> ET.Element | None:
    return next((child for child in element.iter() if _name(child.tag) in names), None)

def _label(element: ET.Element) -> str | None:
    for key in ("id", "label", "letter", "key", "option"):
        value = element.attrib.get(key, "").strip().upper()
        if value in "ABCDE" and len(value) == 1:
            return value
    match = re.match(r"(?:answer|alternative|option)?[_-]?([a-e])$", _name(element.tag))
    return match.group(1).upper() if match else None

class EnemXMLParser:
    def parse(self, path: str | Path) -> list[RawQuestion]:
        try:
            root = ET.parse(Path(path)).getroot()
        except (ET.ParseError, UnicodeError) as exc:
            raise ValueError(f"invalid XML or encoding: {exc}") from exc
        nodes = [node for node in root.iter() if _name(node.tag) in {"question", "questao", "item"}]
        return [self._parse(node, index) for index, node in enumerate(nodes, 1)]

    def _parse(self, node: ET.Element, index: int) -> RawQuestion:
        alternatives: dict[str, str] = {}
        gold = node.attrib.get("answer") or node.attrib.get("correct") or node.attrib.get("gold")
        for child in node.iter():
            label = _label(child)
            if label:
                alternatives[label] = _text(child)
                marker = (child.attrib.get("correct") or child.attrib.get("isCorrect") or "").lower()
                if marker in {"true", "1", "yes"}:
                    gold = label
        answer_node = _first(node, {"correctanswer", "correct_answer", "goldanswer", "gold_answer", "key"})
        if answer_node is not None and _text(answer_node).strip().upper() in set("ABCDE"):
            gold = _text(answer_node).strip().upper()
        return RawQuestion(
            attributes={_name(key): value for key, value in node.attrib.items()},
            header=_text(_first(node, {"header", "context", "cabecalho"})),
            statement=_text(_first(node, {"statement", "body", "prompt", "enunciado"})),
            alternatives=alternatives,
            gold_answer=gold.strip().upper() if gold else None,
            source_index=index,
        )
