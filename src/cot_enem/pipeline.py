"""Incremental Phase 3 pipeline for initial CoT and Specify evolution."""

from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from cot_enem.agents.specify import SpecifyAgent
from cot_enem.dataset.schema import (
    EvolvedRecord,
    GenerationMetadata,
    NormalizedQuestion,
    QuestionContent,
    Reasoning,
    RecordStatus,
    Strategy,
    ValidationResult,
)
from cot_enem.generation.initial_cot import InitialCoTGenerator
from cot_enem.generation.prompting import render_steps
from cot_enem.providers.base import LLMProvider
from cot_enem.utils.hashing import stable_hash
from cot_enem.utils.jsonl import append_jsonl, read_jsonl
from cot_enem.validation.answer_validator import AnswerValidator
from cot_enem.validation.correctness_judge import CorrectnessJudge
from cot_enem.validation.evolution_judge import EvolutionSuccessJudge
from cot_enem.validation.format_validator import FormatValidator


class SpecifyPipeline:
    """Generate and persist one Specify descendant for each eligible root item."""

    def __init__(
        self,
        generator: InitialCoTGenerator,
        agent: SpecifyAgent,
        evolution_judge: EvolutionSuccessJudge,
        correctness_judge: CorrectnessJudge,
        *,
        generator_provider: LLMProvider,
        judge_provider: LLMProvider,
        execution_id: str | None = None,
        temperature: float = 0.2,
    ) -> None:
        self.generator = generator
        self.agent = agent
        self.evolution_judge = evolution_judge
        self.correctness_judge = correctness_judge
        self.generator_provider = generator_provider
        self.judge_provider = judge_provider
        self.execution_id = execution_id or str(uuid4())
        self.temperature = temperature
        self.answer_validator = AnswerValidator()
        self.format_validator = FormatValidator()

    def run(
        self,
        input_path: str | Path,
        output_path: str | Path,
        *,
        limit: int | None = None,
        eligible_only: bool = True,
    ) -> int:
        output = Path(output_path)
        existing_ids = {
            item["id"] for item in read_jsonl(output) if isinstance(item.get("id"), str)
        }
        written = 0
        for raw in read_jsonl(input_path):
            question = NormalizedQuestion.model_validate(raw)
            if eligible_only and not question.eligible:
                continue
            record_id = self._record_id(question.id)
            if record_id in existing_ids:
                continue
            append_jsonl(output, self.process(question, record_id=record_id))
            existing_ids.add(record_id)
            written += 1
            if limit is not None and written >= limit:
                break
        return written

    def process(self, question: NormalizedQuestion, *, record_id: str | None = None) -> EvolvedRecord:
        initial = self.generator.generate(question)
        improved = self.agent.evolve(question, initial)
        initial_format = self.format_validator.validate(initial)
        improved_format = self.format_validator.validate(improved)
        initial_answer = self.answer_validator.validate(initial.final_answer, question.gold_answer)
        improved_answer = self.answer_validator.validate(improved.final_answer, question.gold_answer)
        evolution = self.evolution_judge.judge(question, initial, improved)
        correctness = self.correctness_judge.judge(question, improved)

        reasons = [
            *initial_format.reasons,
            *improved_format.reasons,
            *([] if initial_answer.matches else [initial_answer.reason]),
            *([] if improved_answer.matches else ["specified_answer_mismatch"]),
            *([] if evolution.approved else [f"evolution: {reason}" for reason in evolution.reasons]),
            *(
                []
                if correctness.approved
                else [f"correctness: {reason}" for reason in correctness.reasons]
            ),
        ]
        reasons = [reason for reason in reasons if reason]
        accepted = (
            initial_format.valid
            and improved_format.valid
            and initial_answer.matches
            and improved_answer.matches
            and evolution.approved
            and correctness.approved
        )
        generator_model = getattr(self.generator_provider, "model", type(self.generator_provider).__name__)
        judge_model = getattr(self.judge_provider, "model", type(self.judge_provider).__name__)
        return EvolvedRecord(
            id=record_id or self._record_id(question.id),
            root_id=question.id,
            parent_id=question.id,
            generation=1,
            strategy=Strategy.SPECIFY,
            question=QuestionContent(
                header=question.header,
                statement=question.statement,
                alternatives=question.alternatives,
                gold_answer=question.gold_answer,
            ),
            reasoning=Reasoning(
                initial_cot=render_steps(initial.reasoning_steps),
                improved_cot=render_steps(improved.reasoning_steps),
                predicted_answer=improved.final_answer,
            ),
            generation_metadata=GenerationMetadata(
                generator_model=str(generator_model),
                judge_model=str(judge_model),
                prompt_version=self.agent.prompt.version,
                temperature=self.temperature,
                timestamp=datetime.now(timezone.utc),
                execution_id=self.execution_id,
            ),
            validation=ValidationResult(
                format_valid=initial_format.valid and improved_format.valid,
                answer_matches_gold=initial_answer.matches and improved_answer.matches,
                correctness_verified=correctness.approved,
                evolution_success=evolution.approved,
                enem_style_valid=True,
                accepted=accepted,
                rejection_reasons=reasons,
            ),
            status=RecordStatus.ACCEPTED if accepted else RecordStatus.REJECTED,
        )

    @staticmethod
    def _record_id(root_id: str) -> str:
        digest = stable_hash({"root_id": root_id, "strategy": "specify", "generation": 1})[:16]
        return f"cotenem_{digest}"
