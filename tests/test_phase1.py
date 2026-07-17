from pathlib import Path
import pytest
from pydantic import ValidationError
from cot_enem.dataset.filters import partition_questions
from cot_enem.dataset.normalizer import QuestionNormalizer
from cot_enem.dataset.parser import EnemXMLParser
from cot_enem.dataset.repository import QuestionRepository
from cot_enem.dataset.schema import NormalizedQuestion
from cot_enem.utils.jsonl import read_jsonl

FIXTURE = Path(__file__).parent / "fixtures" / "enem_sample.xml"

def question(identifier="x"):
    return NormalizedQuestion(id=identifier, statement="Q", gold_answer="A",
        alternatives={"A": "1", "B": "2", "C": "3", "D": "4", "E": "5"})

def test_parser_and_normalizer():
    raw = EnemXMLParser().parse(FIXTURE)
    assert len(raw) == 2 and raw[0].gold_answer == "C"
    first, second = [QuestionNormalizer().normalize(item) for item in raw]
    assert first.id == "enem_2015_042" and first.eligible
    assert not second.eligible and "depends_on_image" in second.ineligibility_reasons

def test_schema_rejects_duplicate_alternatives():
    with pytest.raises(ValidationError, match="unique"):
        NormalizedQuestion(id="x", statement="Q", gold_answer="A",
            alternatives={"A": "x", "B": "x", "C": "z", "D": "w", "E": "v"})

def test_duplicate_ids_are_reported():
    report = partition_questions([question(), question()])
    assert report.duplicate_ids == {"x"} and len(report.ineligible) == 1

def test_repository_is_resumable(tmp_path):
    output = tmp_path / "normalized.jsonl"
    repository = QuestionRepository()
    assert repository.prepare(FIXTURE, output) == 2
    assert repository.prepare(FIXTURE, output) == 0
    assert len(list(read_jsonl(output))) == 2


def test_repository_records_invalid_item_and_continues(tmp_path):
    source = tmp_path / "enem-2017.xml"
    source.write_text(
        """<exam>
        <question id="1"><statement>Incompleta</statement>
          <option id="A"></option><option id="B"></option><option id="C"></option>
          <option id="D"></option><option id="E" correct="Yes"></option>
        </question>
        <question id="2"><statement>Completa</statement>
          <option id="A" correct="Yes">A</option><option id="B">B</option>
          <option id="C">C</option><option id="D">D</option><option id="E">E</option>
        </question></exam>""",
        encoding="utf-8",
    )
    output = tmp_path / "normalized.jsonl"
    repository = QuestionRepository()
    assert repository.prepare(source, output) == 1
    assert repository.last_rejected_count == 1
    assert list(read_jsonl(output))[0]["id"] == "enem_2017_002"
    rejected = list(read_jsonl(tmp_path / "normalized.rejected.jsonl"))
    assert rejected[0]["source_id"] == "1"
    assert rejected[0]["status"] == "rejected"
