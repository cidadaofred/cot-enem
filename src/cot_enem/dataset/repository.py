"""Read-only source processing and incremental JSONL persistence."""
from dataclasses import asdict
from pathlib import Path
import re

from pydantic import ValidationError
from cot_enem.dataset.normalizer import QuestionNormalizer
from cot_enem.dataset.parser import EnemXMLParser
from cot_enem.utils.jsonl import append_jsonl, read_jsonl

class QuestionRepository:
    def __init__(self, parser=None, normalizer=None):
        self.parser = parser or EnemXMLParser()
        self.normalizer = normalizer or QuestionNormalizer()
        self.last_rejected_count = 0

    def prepare(self, source: str | Path, destination: str | Path, *, year: int | None = None) -> int:
        source_path = Path(source)
        output = Path(destination)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.touch(exist_ok=True)
        existing = {item["id"] for item in read_jsonl(output)}
        rejected_output = output.with_name(f"{output.stem}.rejected.jsonl")
        rejected_ids = {
            item["source_id"] for item in read_jsonl(rejected_output) if item.get("source_id")
        }
        effective_year = year or self._infer_year(source_path)
        count = 0
        self.last_rejected_count = 0
        for raw in self.parser.parse(source_path):
            try:
                question = self.normalizer.normalize(raw, default_year=effective_year)
            except ValidationError as exc:
                source_id = raw.attributes.get("id") or f"index_{raw.source_index}"
                if source_id not in rejected_ids:
                    append_jsonl(
                        rejected_output,
                        {
                            "source_id": source_id,
                            "source_index": raw.source_index,
                            "year": effective_year,
                            "status": "rejected",
                            "rejection_reasons": [
                                {
                                    "field": ".".join(str(part) for part in error["loc"]),
                                    "message": error["msg"],
                                    "type": error["type"],
                                }
                                for error in exc.errors()
                            ],
                            "raw_question": asdict(raw),
                        },
                    )
                    rejected_ids.add(source_id)
                    self.last_rejected_count += 1
                continue
            if question.id not in existing:
                append_jsonl(output, question)
                existing.add(question.id)
                count += 1
        return count

    @staticmethod
    def _infer_year(source: Path) -> int | None:
        match = re.search(r"(?:19|20)\d{2}", source.stem)
        return int(match.group()) if match else None
