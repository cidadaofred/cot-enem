from cot_enem.config import PromptCatalog
from cot_enem.ensemble_pipeline import SpecifyEnsemblePipeline
from cot_enem.providers.mock_provider import MockLLMProvider
from cot_enem.utils.jsonl import append_jsonl, read_jsonl


def question():
    return {
        "id": "enem_2017_001",
        "year": 2017,
        "question_number": 1,
        "statement": "Quanto é 2 + 2?",
        "alternatives": {"A": "1", "B": "2", "C": "3", "D": "4", "E": "5"},
        "gold_answer": "D",
        "eligible": True,
    }


def test_three_pass_ensemble_persists_votes_and_uses_majority(tmp_path):
    source = tmp_path / "input.jsonl"
    candidates = tmp_path / "candidates.jsonl"
    votes = tmp_path / "votes.jsonl"
    output = tmp_path / "output.jsonl"
    append_jsonl(source, question())
    prompts = PromptCatalog.from_yaml("config/prompts.yaml")
    pipeline = SpecifyEnsemblePipeline(prompts)
    generator = MockLLMProvider(
        [
            {"reasoning_steps": ["2 + 2 = 4"], "final_answer": "D"},
            {"reasoning_steps": ["Somando 2 e 2, obtemos 4."], "final_answer": "D"},
        ]
    )
    generator.model = "generator"
    assert pipeline.generate_candidates(source, candidates, generator) == 1

    decisions = [(True, True), (True, False), (False, True)]
    for index, (evolution, correctness) in enumerate(decisions, 1):
        judge = MockLLMProvider(
            [
                {"approved": evolution, "reasons": []},
                {"approved": correctness, "reasons": []},
            ]
        )
        judge.model = f"judge-{index}"
        assert pipeline.collect_votes(candidates, votes, judge) == 1

    assert pipeline.aggregate(
        candidates,
        votes,
        output,
        ["judge-1", "judge-2", "judge-3"],
    ) == 1
    record = list(read_jsonl(output))[0]
    assert record["validation"]["accepted"] is True
    assert record["validation"]["evolution_success"] is True
    assert record["validation"]["correctness_verified"] is True
    assert len(record["validation"]["judge_votes"]) == 3

    assert pipeline.generate_candidates(source, candidates, generator) == 0
    assert pipeline.aggregate(
        candidates,
        votes,
        output,
        ["judge-1", "judge-2", "judge-3"],
    ) == 0
