"""CPU-only consolidation, metrics, audit, and reporting for the final dataset."""

from __future__ import annotations

from collections import Counter
import csv
from dataclasses import dataclass
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Iterable

from cot_enem.dataset.schema import EvolvedRecord, NormalizedQuestion, Strategy
from cot_enem.utils.jsonl import read_jsonl


@dataclass(frozen=True)
class Phase6Outputs:
    accepted: Path
    rejected: Path
    metrics_json: Path
    metrics_csv: Path
    report: Path


def _load_records(path: str | Path, strategy: Strategy) -> list[EvolvedRecord]:
    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(f"expected {strategy.value} results not found: {source}")
    records = [EvolvedRecord.model_validate(raw) for raw in read_jsonl(source)]
    unexpected = [record.id for record in records if record.strategy != strategy]
    if unexpected:
        raise ValueError(
            f"{source} contains records outside strategy={strategy.value}: "
            f"{unexpected[:10]}"
        )
    return records


def _atomic_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _write_jsonl(path: Path, records: Iterable[EvolvedRecord]) -> None:
    content = "".join(
        json.dumps(record.model_dump(mode="json"), ensure_ascii=False) + "\n"
        for record in records
    )
    _atomic_text(path, content)


def _step_count(value: str | None) -> int:
    if not value:
        return 0
    lines = [line for line in value.splitlines() if line.strip()]
    return len(lines) if len(lines) > 1 else 1


def _mean(values: list[int]) -> float:
    return round(sum(values) / len(values), 3) if values else 0.0


def _strategy_metrics(records: list[EvolvedRecord]) -> dict[str, Any]:
    accepted = [record for record in records if record.validation.accepted]
    rejected = [record for record in records if not record.validation.accepted]
    reasons = Counter(
        reason
        for record in rejected
        for reason in record.validation.rejection_reasons
    )
    judge_votes = [
        record.validation.judge_votes
        for record in records
        if record.validation.judge_votes
    ]
    evolution_unanimous = sum(
        len({vote.evolution_success for vote in votes}) == 1 for votes in judge_votes
    )
    correctness_unanimous = sum(
        len({vote.correctness_verified for vote in votes}) == 1 for votes in judge_votes
    )
    return {
        "records": len(records),
        "accepted": len(accepted),
        "rejected": len(rejected),
        "acceptance_rate_percent": round(
            len(accepted) / len(records) * 100, 3
        ) if records else 0.0,
        "average_initial_cot_steps": _mean(
            [_step_count(record.reasoning.initial_cot) for record in records]
        ),
        "average_improved_cot_steps": _mean(
            [_step_count(record.reasoning.improved_cot) for record in records]
        ),
        "records_with_judge_votes": len(judge_votes),
        "evolution_unanimity_rate_percent": round(
            evolution_unanimous / len(judge_votes) * 100, 3
        ) if judge_votes else 0.0,
        "correctness_unanimity_rate_percent": round(
            correctness_unanimous / len(judge_votes) * 100, 3
        ) if judge_votes else 0.0,
        "rejection_reasons": dict(sorted(reasons.items())),
    }


def _report_markdown(metrics: dict[str, Any], outputs: Phase6Outputs) -> str:
    rows = []
    for strategy in Strategy:
        item = metrics["by_strategy"][strategy.value]
        rows.append(
            f"| {strategy.value} | {item['records']} | {item['accepted']} | "
            f"{item['rejected']} | {item['acceptance_rate_percent']:.1f}% |"
        )
    return f"""# Relatório final — CoT-ENEM ENEM 2017

## Escopo

Consolidação CPU-only dos resultados validados de `Specify`, `Complicate` e
`Diversify`. Os artefatos intermediários permanecem imutáveis e rastreáveis.

## Resultado

| Estratégia | Registros | Aceitos | Rejeitados | Aceitação |
|---|---:|---:|---:|---:|
{chr(10).join(rows)}
| **Total** | **{metrics['total_records']}** | **{metrics['accepted']}** | **{metrics['rejected']}** | **{metrics['acceptance_rate_percent']:.1f}%** |

- Questões normalizadas: {metrics['normalized_questions']}
- Questões elegíveis: {metrics['eligible_questions']}
- Questões-raiz representadas nos aceitos: {metrics['accepted_unique_roots']}
- IDs duplicados: {metrics['duplicate_ids']}
- Referências a raízes ausentes: {metrics['missing_root_references']}
- Auditoria consolidada completa: `{str(metrics['complete']).lower()}`

## Artefatos

- `{outputs.accepted.name}`: somente registros aceitos, pronto para consumo.
- `{outputs.rejected.name}`: rejeitados preservados para auditoria.
- `{outputs.metrics_json.name}`: métricas completas e legíveis por máquina.
- `{outputs.metrics_csv.name}`: resumo tabular por estratégia.
- `{outputs.report.name}`: este relatório reproduzível.

## Critério de uso

O dataset aprovado contém apenas registros com `validation.accepted=true`. Os
rejeitados não fazem parte do conjunto de treinamento ou avaliação, mas devem ser
mantidos para análise dos filtros e justificativas dos juízes.
"""


def finalize_phase6(
    normalized_path: str | Path,
    specify_path: str | Path,
    complicate_path: str | Path,
    diversify_path: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    """Consolidate persisted results without invoking an LLM."""

    normalized = [
        NormalizedQuestion.model_validate(raw) for raw in read_jsonl(normalized_path)
    ]
    if not normalized:
        raise ValueError("normalized dataset is empty")
    root_ids = {question.id for question in normalized}
    if len(root_ids) != len(normalized):
        raise ValueError("normalized dataset contains duplicate IDs")

    by_strategy = {
        Strategy.SPECIFY: _load_records(specify_path, Strategy.SPECIFY),
        Strategy.COMPLICATE: _load_records(complicate_path, Strategy.COMPLICATE),
        Strategy.DIVERSIFY: _load_records(diversify_path, Strategy.DIVERSIFY),
    }
    records = [record for strategy in Strategy for record in by_strategy[strategy]]
    counts = Counter(record.id for record in records)
    duplicate_ids = sorted(record_id for record_id, count in counts.items() if count > 1)
    missing_roots = sorted({record.root_id for record in records} - root_ids)
    if duplicate_ids or missing_roots:
        raise ValueError(
            "Phase 6 integrity errors: "
            f"duplicate_ids={duplicate_ids[:10]}, missing_roots={missing_roots[:10]}"
        )
    for record in records:
        expected_status = "accepted" if record.validation.accepted else "rejected"
        if record.status.value != expected_status:
            raise ValueError(f"{record.id}: status does not match validation.accepted")

    accepted = sorted(
        (record for record in records if record.validation.accepted),
        key=lambda record: (record.root_id, record.strategy.value, record.id),
    )
    rejected = sorted(
        (record for record in records if not record.validation.accepted),
        key=lambda record: (record.root_id, record.strategy.value, record.id),
    )
    destination = Path(output_dir)
    outputs = Phase6Outputs(
        accepted=destination / "cot_enem_2017_accepted.jsonl",
        rejected=destination / "cot_enem_2017_rejected.jsonl",
        metrics_json=destination / "metrics.json",
        metrics_csv=destination / "strategy_metrics.csv",
        report=destination / "REPORT.md",
    )
    metrics: dict[str, Any] = {
        "dataset": "CoT-ENEM",
        "source_year": 2017,
        "normalized_questions": len(normalized),
        "eligible_questions": sum(question.eligible for question in normalized),
        "total_records": len(records),
        "accepted": len(accepted),
        "rejected": len(rejected),
        "acceptance_rate_percent": round(len(accepted) / len(records) * 100, 3),
        "accepted_unique_roots": len({record.root_id for record in accepted}),
        "duplicate_ids": len(duplicate_ids),
        "missing_root_references": len(missing_roots),
        "complete": True,
        "by_strategy": {
            strategy.value: _strategy_metrics(by_strategy[strategy])
            for strategy in Strategy
        },
    }
    _write_jsonl(outputs.accepted, accepted)
    _write_jsonl(outputs.rejected, rejected)
    _atomic_text(
        outputs.metrics_json,
        json.dumps(metrics, ensure_ascii=False, indent=2) + "\n",
    )
    csv_rows = [
        {
            "strategy": strategy.value,
            **{
                key: value
                for key, value in metrics["by_strategy"][strategy.value].items()
                if key != "rejection_reasons"
            },
        }
        for strategy in Strategy
    ]
    csv_header = list(csv_rows[0])
    with tempfile.NamedTemporaryFile(
        "w", delete=False, encoding="utf-8", newline="", dir=destination
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=csv_header)
        writer.writeheader()
        writer.writerows(csv_rows)
        csv_temporary = stream.name
    os.replace(csv_temporary, outputs.metrics_csv)
    _atomic_text(outputs.report, _report_markdown(metrics, outputs))
    return {**metrics, "outputs": {key: str(value) for key, value in outputs.__dict__.items()}}
