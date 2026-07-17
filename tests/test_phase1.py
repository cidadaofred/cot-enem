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
