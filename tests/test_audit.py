from cot_enem.audit import audit_phase3
from cot_enem.config import PromptCatalog
from cot_enem.ensemble_pipeline import SpecifyEnsemblePipeline
from cot_enem.providers.mock_provider import MockLLMProvider
from cot_enem.utils.jsonl import append_jsonl


def test_phase3_audit_accepts_complete_majority_artifacts(tmp_path):
    source = tmp_path / "input.jsonl"
    candidates = tmp_path / "candidates.jsonl"
    votes = tmp_path / "votes.jsonl"
    output = tmp_path / "output.jsonl"
    append_jsonl(
        source,
        {
            "id": "enem_2017_001",
            "year": 2017,
            "question_number": 1,
            "statement": "Quanto é 2 + 2?",
            "alternatives": {"A": "1", "B": "2", "C": "3", "D": "4", "E": "5"},
            "gold_answer": "D",
            "eligible": True,
        },
    )
    pipeline = SpecifyEnsemblePipeline(PromptCatalog.from_yaml("config/prompts.yaml"))
    generator = MockLLMProvider(
        [
            {"reasoning_steps": ["2 + 2 = 4"], "final_answer": "D"},
            {"reasoning_steps": ["A soma resulta em 4."], "final_answer": "D"},
        ]
    )
    generator.model = "generator"
    pipeline.generate_candidates(source, candidates, generator)
    judges = ["judge-1", "judge-2", "judge-3"]
    for model in judges:
        judge = MockLLMProvider(
            [
                {"approved": True, "reasons": []},
                {"approved": True, "reasons": []},
            ]
        )
        judge.model = model
        pipeline.collect_votes(candidates, votes, judge)
    pipeline.aggregate(candidates, votes, output, judges)

    summary = audit_phase3(source, candidates, votes, output, judges)

    assert summary["complete"] is True
    assert summary["results"] == 1
    assert summary["votes"] == 3
