"""Conversion from permissive parsed values into the canonical schema."""
import re
import unicodedata
from cot_enem.dataset.parser import RawQuestion
from cot_enem.dataset.schema import Annotations, NormalizedQuestion

def _boolean(value: str | None) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "sim", "s"}

def clean_text(value: str) -> str:
    return re.sub(r"\s+", " ", unicodedata.normalize("NFC", value).replace("\x00", "")).strip()

class QuestionNormalizer:
    def normalize(self, raw: RawQuestion, *, default_year: int | None = None) -> NormalizedQuestion:
        attrs = {key.lower(): value for key, value in raw.attributes.items()}
        year = self._integer(attrs.get("year") or attrs.get("ano")) or default_year
        number = self._integer(attrs.get("id") or attrs.get("number") or attrs.get("numero"))
        source_id = attrs.get("id") or str(raw.source_index)
        identifier = f"enem_{year}_{number:03d}" if year and number else f"enem_{year or 'unknown'}_{source_id}"
        annotations = Annotations(**{
            key: _boolean(attrs.get(key.lower()))
            for key in ("image", "EK", "TC", "IC", "DS", "MR", "CE")
        })
        reasons = ["depends_on_image"] if annotations.image or annotations.IC else []
        return NormalizedQuestion(
            id=identifier, year=year, question_number=number,
            area=clean_text(attrs.get("area", "")) or None,
            header=clean_text(raw.header), statement=clean_text(raw.statement),
            alternatives={key: clean_text(value) for key, value in raw.alternatives.items()},
            gold_answer=raw.gold_answer, annotations=annotations,
            eligible=not reasons, ineligibility_reasons=reasons,
        )

    @staticmethod
    def _integer(value: str | None) -> int | None:
        match = re.search(r"\d+", value or "")
        return int(match.group()) if match else None
