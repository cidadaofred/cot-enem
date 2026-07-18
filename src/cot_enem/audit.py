"""CPU-only integrity audit for persisted Phase 3 ensemble artifacts."""

from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from cot_enem.dataset.schema import EvolvedRecord, NormalizedQuestion
from cot_enem.ensemble_pipeline import EnsembleVote, SpecifyCandidate
from cot_enem.utils.jsonl import read_jsonl


def _validated(path: str | Path, model: Any) -> list[Any]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"expected Phase 3 artifact not found: {path}")
    return [model.model_validate(raw) for raw in read_jsonl(path)]


def _require_unique(values: list[Any], label: str) -> None:
    duplicates = [value for value, count in Counter(values).items() if count > 1]
    if duplicates:
        raise ValueError(f"{label} contains duplicates: {duplicates[:10]}")


def audit_phase3(
    input_path: str | Path,
    candidate_path: str | Path,
    vote_path: str | Path,
    output_path: str | Path,
    judge_models: list[str],
    *,
    require_complete: bool = True,
) -> dict[str, Any]:
    """Validate schemas, references, majority votes, and optional full completion."""

    if len(judge_models) != 3 or len(set(judge_models)) != 3:
        raise ValueError("Phase 3 audit requires exactly three distinct judge models")
    questions = _validated(input_path, NormalizedQuestion)
    candidates = _validated(candidate_path, SpecifyCandidate)
    votes = _validated(vote_path, EnsembleVote)
    results = _validated(output_path, EvolvedRecord)
    _require_unique([question.id for question in questions], "normalized questions")
    _require_unique([candidate.id for candidate in candidates], "candidates")
    _require_unique(
        [(vote.candidate_id, vote.judge_model) for vote in votes],
        "candidate/model votes",
    )
    _require_unique([result.id for result in results], "aggregated results")

    eligible_ids = {question.id for question in questions if question.eligible}
    candidate_by_id = {candidate.id: candidate for candidate in candidates}
    result_by_id = {result.id: result for result in results}
    votes_by_candidate: dict[str, dict[str, EnsembleVote]] = defaultdict(dict)
    errors: list[str] = []

    for candidate in candidates:
        if candidate.root.id not in eligible_ids:
            errors.append(f"{candidate.id}: root is absent or ineligible")
        if not candidate.initial.reasoning_steps or not candidate.improved.reasoning_steps:
            errors.append(f"{candidate.id}: empty reasoning")
    for vote in votes:
        if vote.candidate_id not in candidate_by_id:
            errors.append(f"{vote.candidate_id}: orphan vote")
        if vote.judge_model not in judge_models:
            errors.append(f"{vote.candidate_id}: unexpected judge {vote.judge_model}")
        votes_by_candidate[vote.candidate_id][vote.judge_model] = vote
    for result in results:
        if result.id not in candidate_by_id:
            errors.append(f"{result.id}: result has no candidate")
            continue
        model_votes = votes_by_candidate.get(result.id, {})
        if set(model_votes) != set(judge_models):
            errors.append(f"{result.id}: result does not have the three configured votes")
            continue
        evolution = sum(model_votes[name].evolution.approved for name in judge_models) >= 2
        correctness = (
            sum(model_votes[name].correctness.approved for name in judge_models) >= 2
        )
        if result.validation.evolution_success != evolution:
            errors.append(f"{result.id}: evolution majority mismatch")
        if result.validation.correctness_verified != correctness:
            errors.append(f"{result.id}: correctness majority mismatch")
        expected_status = "accepted" if result.validation.accepted else "rejected"
        if result.status.value != expected_status:
            errors.append(f"{result.id}: status does not match validation.accepted")
    if errors:
        raise ValueError("Phase 3 integrity errors:\n- " + "\n- ".join(errors[:50]))

    missing_candidates = eligible_ids - {
        candidate.root.id for candidate in candidates
    }
    incomplete_votes = {
        candidate_id: sorted(set(judge_models) - set(votes_by_candidate[candidate_id]))
        for candidate_id in candidate_by_id
        if set(votes_by_candidate[candidate_id]) != set(judge_models)
    }
    missing_results = set(candidate_by_id) - set(result_by_id)
    complete = not missing_candidates and not incomplete_votes and not missing_results
    if require_complete and not complete:
        raise ValueError(
            "Phase 3 artifacts are valid but incomplete: "
            f"missing_candidates={len(missing_candidates)}, "
            f"incomplete_votes={len(incomplete_votes)}, "
            f"missing_results={len(missing_results)}"
        )
    statuses = Counter(result.status.value for result in results)
    return {
        "normalized_questions": len(questions),
        "eligible_questions": len(eligible_ids),
        "candidates": len(candidates),
        "votes": len(votes),
        "expected_votes": len(candidates) * 3,
        "results": len(results),
        "accepted": statuses["accepted"],
        "rejected": statuses["rejected"],
        "acceptance_rate": (
            statuses["accepted"] / len(results) * 100 if results else 0.0
        ),
        "missing_candidates": len(missing_candidates),
        "incomplete_votes": len(incomplete_votes),
        "missing_results": len(missing_results),
        "complete": complete,
    }
