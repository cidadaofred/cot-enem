from datetime import datetime, timezone

import pytest

from cot_enem.audit import audit_phase4_branch
from cot_enem.config import PromptCatalog
from cot_enem.dataset.schema import NormalizedQuestion, Strategy
from cot_enem.ensemble_pipeline import SpecifyCandidate
from cot_enem.generation.schema import InitialCoT, SpecifiedCoT
from cot_enem.phase4_pipeline import (
    EvolutionSeed,
    QuestionEvolutionEnsemblePipeline,
)
from cot_enem.providers.mock_provider import MockLLMProvider
from cot_enem.utils.jsonl import append_jsonl, read_jsonl


def root_question():
    return NormalizedQuestion(
        id="enem_2017_001",
        year=2017,
        question_number=1,
        statement="Quanto é 2 + 2?",
        alternatives={"A": "1", "B": "2", "C": "3", "D": "4", "E": "5"},
        gold_answer="D",
    )


def write_phase3_candidate(path):
    append_jsonl(
        path,
        SpecifyCandidate(
            id="specify-artifact",
            root=root_question(),
            initial=InitialCoT(reasoning_steps=["Somar 2 e 2."], final_answer="D"),
            improved=SpecifiedCoT(
                reasoning_steps=["Somando, obtemos 4."], final_answer="D"
            ),
            generator_model="generator",
            prompt_version="specify_v1",
            temperature=0.2,
            timestamp=datetime.now(timezone.utc),
            execution_id="phase3-run",
        ),
    )


@pytest.mark.parametrize("strategy", [Strategy.COMPLICATE, Strategy.DIVERSIFY])
def test_phase4_branches_reuse_initial_seed_and_persist_separate_lineage(
    tmp_path, strategy
):
    phase3 = tmp_path / "phase3.jsonl"
    seeds = tmp_path / "seeds.jsonl"
    candidates = tmp_path / "candidates.jsonl"
    votes = tmp_path / "votes.jsonl"
    output = tmp_path / "output.jsonl"
    write_phase3_candidate(phase3)
    prompts = PromptCatalog.from_yaml("config/prompts.yaml")
    pipeline = QuestionEvolutionEnsemblePipeline(prompts, strategy)

    assert pipeline.import_phase3_seeds(phase3, seeds) == 1
    seed = EvolutionSeed.model_validate(next(read_jsonl(seeds)))
    assert seed.parent_id == root_question().id
    assert seed.source_artifact_id == "specify-artifact"
    assert seed.cot.reasoning_steps == ["Somar 2 e 2."]

    generator = MockLLMProvider(
        [
            {
                "header": "Questão evoluída",
                "statement": "Quanto é 3 + 3?",
                "alternatives": {
                    "A": "2",
                    "B": "3",
                    "C": "4",
                    "D": "5",
                    "E": "6",
                },
                "reasoning_steps": ["Somar 3 e 3 resulta em 6."],
                "final_answer": "E",
            }
        ]
    )
    generator.model = "generator"
    assert pipeline.generate_candidates(seeds, candidates, generator) == 1

    judges = ["judge-1", "judge-2", "judge-3"]
    for model in judges:
        judge = MockLLMProvider(
            [
                {"approved": True, "reasons": []},
                {"approved": True, "reasons": []},
            ]
        )
        judge.model = model
        assert pipeline.collect_votes(candidates, votes, judge) == 1

    assert pipeline.aggregate(candidates, votes, output, judges) == 1
    result = next(read_jsonl(output))
    assert result["strategy"] == strategy.value
    assert result["root_id"] == root_question().id
    assert result["parent_id"] == seed.id
    assert result["generation"] == 1
    assert result["question"]["statement"] == "Quanto é 3 + 3?"
    assert result["validation"]["accepted"] is True
    assert len(result["validation"]["judge_votes"]) == 3
    audit = audit_phase4_branch(
        seeds,
        candidates,
        votes,
        output,
        judges,
        strategy,
    )
    assert audit["complete"] is True


def test_phase4_can_import_accepted_result_as_next_generation_seed(tmp_path):
    phase3 = tmp_path / "phase3.jsonl"
    seeds = tmp_path / "seeds.jsonl"
    candidates = tmp_path / "candidates.jsonl"
    votes = tmp_path / "votes.jsonl"
    output = tmp_path / "output.jsonl"
    next_seeds = tmp_path / "next-seeds.jsonl"
    write_phase3_candidate(phase3)
    pipeline = QuestionEvolutionEnsemblePipeline(
        PromptCatalog.from_yaml("config/prompts.yaml"),
        Strategy.COMPLICATE,
    )
    pipeline.import_phase3_seeds(phase3, seeds)
    generator = MockLLMProvider(
        [
            {
                "header": "",
                "statement": "Quanto é 3 + 3?",
                "alternatives": {
                    "A": "2",
                    "B": "3",
                    "C": "4",
                    "D": "5",
                    "E": "6",
                },
                "reasoning_steps": ["3 + 3 = 6."],
                "final_answer": "E",
            }
        ]
    )
    generator.model = "generator"
    pipeline.generate_candidates(seeds, candidates, generator)
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

    diversify = QuestionEvolutionEnsemblePipeline(
        PromptCatalog.from_yaml("config/prompts.yaml"),
        Strategy.DIVERSIFY,
    )
    assert diversify.import_accepted_results(output, next_seeds) == 1
    next_seed = EvolutionSeed.model_validate(next(read_jsonl(next_seeds)))
    parent = next(read_jsonl(output))
    assert next_seed.parent_id == parent["id"]
    assert next_seed.root_id == root_question().id
    assert next_seed.generation == 1
