from datetime import datetime, timezone
import json

import pytest

from cot_enem.cli import main
from cot_enem.dataset.schema import (
    EvolvedRecord,
    GenerationMetadata,
    QuestionContent,
    Reasoning,
    RecordStatus,
    Strategy,
    ValidationResult,
)
from cot_enem.phase6 import finalize_phase6
from cot_enem.utils.jsonl import append_jsonl, read_jsonl


def _record(identifier: str, strategy: Strategy, accepted: bool) -> EvolvedRecord:
    return EvolvedRecord(
        id=identifier,
        root_id="enem_2017_001",
        parent_id="enem_2017_001",
        generation=1,
        strategy=strategy,
        question=QuestionContent(
            statement=f"Questão {strategy.value}",
            alternatives={"A": "1", "B": "2", "C": "3", "D": "4", "E": "5"},
            gold_answer="D",
        ),
        reasoning=Reasoning(
            initial_cot="1. Primeira etapa\n2. Segunda etapa",
            improved_cot="1. Etapa detalhada\n2. Conclusão",
            predicted_answer="D",
        ),
        generation_metadata=GenerationMetadata(
            generator_model="generator",
            judge_model="majority:j1,j2,j3",
            prompt_version="v1",
            temperature=0.2,
            timestamp=datetime.now(timezone.utc),
            execution_id="run",
        ),
        validation=ValidationResult(
            format_valid=True,
            answer_matches_gold=True,
            correctness_verified=accepted,
            evolution_success=accepted,
            enem_style_valid=True,
            accepted=accepted,
            rejection_reasons=[] if accepted else ["correctness: majority_rejected"],
        ),
        status=RecordStatus.ACCEPTED if accepted else RecordStatus.REJECTED,
    )


def _inputs(tmp_path):
    normalized = tmp_path / "normalized.jsonl"
    append_jsonl(
        normalized,
        {
            "id": "enem_2017_001",
            "year": 2017,
            "statement": "Quanto é 2 + 2?",
            "alternatives": {"A": "1", "B": "2", "C": "3", "D": "4", "E": "5"},
            "gold_answer": "D",
        },
    )
    paths = {}
    for index, strategy in enumerate(Strategy):
        path = tmp_path / f"{strategy.value}.jsonl"
        append_jsonl(path, _record(f"record-{index}", strategy, strategy != Strategy.SPECIFY))
        paths[strategy] = path
    return normalized, paths


def test_finalize_phase6_writes_deterministic_dataset_and_reports(tmp_path):
    normalized, paths = _inputs(tmp_path)
    output = tmp_path / "final"

    summary = finalize_phase6(
        normalized,
        paths[Strategy.SPECIFY],
        paths[Strategy.COMPLICATE],
        paths[Strategy.DIVERSIFY],
        output,
    )

    assert summary["total_records"] == 3
    assert summary["accepted"] == 2
    assert summary["rejected"] == 1
    assert summary["complete"] is True
    assert len(list(read_jsonl(output / "cot_enem_2017_accepted.jsonl"))) == 2
    assert len(list(read_jsonl(output / "cot_enem_2017_rejected.jsonl"))) == 1
    metrics = json.loads((output / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["by_strategy"]["specify"]["rejection_reasons"] == {
        "correctness: majority_rejected": 1
    }
    assert (output / "strategy_metrics.csv").is_file()
    assert "Auditoria consolidada completa: `true`" in (
        output / "REPORT.md"
    ).read_text(encoding="utf-8")

    first = (output / "cot_enem_2017_accepted.jsonl").read_bytes()
    finalize_phase6(
        normalized,
        paths[Strategy.SPECIFY],
        paths[Strategy.COMPLICATE],
        paths[Strategy.DIVERSIFY],
        output,
    )
    assert (output / "cot_enem_2017_accepted.jsonl").read_bytes() == first


def test_finalize_phase6_rejects_missing_root_reference(tmp_path):
    normalized, paths = _inputs(tmp_path)
    record = _record("orphan", Strategy.DIVERSIFY, True)
    record.root_id = "absent"
    orphan_path = tmp_path / "orphan.jsonl"
    append_jsonl(orphan_path, record)

    with pytest.raises(ValueError, match="missing_roots"):
        finalize_phase6(
            normalized,
            paths[Strategy.SPECIFY],
            paths[Strategy.COMPLICATE],
            orphan_path,
            tmp_path / "final",
        )


def test_finalize_cli_is_cpu_only(tmp_path, capsys, monkeypatch):
    normalized, paths = _inputs(tmp_path)
    output = tmp_path / "final"
    monkeypatch.setattr("cot_enem.cli.load_env_file", lambda _path: None)

    code = main(
        [
            "finalize",
            "--normalized", str(normalized),
            "--specify", str(paths[Strategy.SPECIFY]),
            "--complicate", str(paths[Strategy.COMPLICATE]),
            "--diversify", str(paths[Strategy.DIVERSIFY]),
            "--output-dir", str(output),
        ]
    )

    assert code == 0
    assert '"complete": true' in capsys.readouterr().out
