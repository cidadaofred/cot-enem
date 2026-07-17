"""Read-only source processing and incremental JSONL persistence."""
from pathlib import Path
from cot_enem.dataset.normalizer import QuestionNormalizer
from cot_enem.dataset.parser import EnemXMLParser
from cot_enem.utils.jsonl import append_jsonl, read_jsonl

class QuestionRepository:
    def __init__(self, parser=None, normalizer=None):
        self.parser = parser or EnemXMLParser()
        self.normalizer = normalizer or QuestionNormalizer()

    def prepare(self, source: str | Path, destination: str | Path, *, year: int | None = None) -> int:
        output = Path(destination)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.touch(exist_ok=True)
        existing = {item["id"] for item in read_jsonl(output)}
        count = 0
        for raw in self.parser.parse(source):
            question = self.normalizer.normalize(raw, default_year=year)
            if question.id not in existing:
                append_jsonl(output, question)
                existing.add(question.id)
                count += 1
        return count
