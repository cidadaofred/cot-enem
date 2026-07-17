from pathlib import Path

from cot_enem.agents.specify import SpecifyAgent
from cot_enem.config import PromptCatalog
from cot_enem.dataset.schema import NormalizedQuestion, RecordStatus
from cot_enem.generation.initial_cot import InitialCoTGenerator
from cot_enem.pipeline import SpecifyPipeline
from cot_enem.providers.mock_provider import MockLLMProvider
from cot_enem.utils.jsonl import append_jsonl, read_jsonl
from cot_enem.validation.correctness_judge import CorrectnessJudge
from cot_enem.validation.evolution_judge import EvolutionSuccessJudge

PROMPTS = Path(__file__).parents[1] / "config" / "prompts.yaml"


def question() -> NormalizedQuestion:
    return NormalizedQuestion(
        id="enem_2015_042",
        year=2015,
        question_number=42,
        statement="Qual é a velocidade média ao percorrer 100 m em 5 s?",
        alternatives={"A": "5", "B": "10", "C": "20", "D": "25", "E": "50"},
        gold_answer="C",
    )


def pipeline(initial_answer="C") -> tuple[SpecifyPipeline, MockLLMProvider]:
    prompts = PromptCatalog.from_yaml(PROMPTS)
    generator = MockLLMProvider(
        [
            {"reasoning_steps": ["Dividir distância pelo tempo.", "100/5 = 20."],
             "final_answer": initial_answer},
            {"reasoning_steps": ["Usar v=d/t.", "Substituir 100 m e 5 s.", "v=20 m/s."],
             "final_answer": "C"},
        ],
        model="mock-generator",
    )
    judge = MockLLMProvider(
        [
            {"approved": True, "reasons": []},
            {"approved": True, "reasons": []},
        ],
        model="mock-judge",
    )
    return (
        SpecifyPipeline(
            InitialCoTGenerator(generator, prompts),
            SpecifyAgent(generator, prompts),
            EvolutionSuccessJudge(judge, prompts),
            CorrectnessJudge(judge, prompts),
            generator_provider=generator,
            judge_provider=judge,
            execution_id="test-execution",
        ),
        generator,
    )


def test_initial_generation_does_not_expose_gold_answer():
    flow, provider = pipeline()
    record = flow.process(question())
    initial_user_prompt = provider.calls[0]["messages"][1]["content"]
    assert '"gold_answer"' not in initial_user_prompt
    assert record.status == RecordStatus.ACCEPTED
    assert record.root_id == record.parent_id == "enem_2015_042"
    assert record.validation.accepted


def test_wrong_initial_answer_is_preserved_as_rejected():
    flow, _provider = pipeline(initial_answer="B")
    record = flow.process(question())
    assert record.status == RecordStatus.REJECTED
    assert "initial_answer_mismatch" in record.validation.rejection_reasons
    assert record.reasoning.initial_cot


def test_pipeline_writes_incrementally_and_resumes(tmp_path):
    source, output = tmp_path / "source.jsonl", tmp_path / "output.jsonl"
    append_jsonl(source, question())
    first, _ = pipeline()
    assert first.run(source, output) == 1
    second, _ = pipeline()
    assert second.run(source, output) == 0
    records = list(read_jsonl(output))
    assert len(records) == 1
    assert records[0]["generation_metadata"]["execution_id"] == "test-execution"
