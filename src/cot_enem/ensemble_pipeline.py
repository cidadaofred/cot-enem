"""Three-pass Specify pipeline with heterogeneous sequential majority judges."""

from datetime import datetime, timezone
from pathlib import Path
import re
from uuid import uuid4

from pydantic import Field

from cot_enem.agents.specify import SpecifyAgent
from cot_enem.config import PromptCatalog
from cot_enem.dataset.schema import (
    EvolvedRecord,
    GenerationMetadata,
    JudgeVote,
    NormalizedQuestion,
    QuestionContent,
    Reasoning,
    RecordStatus,
    Strategy,
    StrictModel,
    ValidationResult,
)
from cot_enem.generation.initial_cot import InitialCoTGenerator
from cot_enem.generation.prompting import render_steps
from cot_enem.generation.schema import InitialCoT, JudgeDecision, SpecifiedCoT
from cot_enem.providers.base import LLMProvider
from cot_enem.utils.hashing import stable_hash
from cot_enem.utils.jsonl import append_jsonl, read_jsonl
from cot_enem.validation.answer_validator import AnswerValidator
from cot_enem.validation.correctness_judge import CorrectnessJudge
from cot_enem.validation.evolution_judge import EvolutionSuccessJudge
from cot_enem.validation.format_validator import FormatValidator


class SpecifyCandidate(StrictModel):
    id: str
    root: NormalizedQuestion
    initial: InitialCoT
    improved: SpecifiedCoT
    generator_model: str
    prompt_version: str
    temperature: float
    timestamp: datetime
    execution_id: str


class EnsembleVote(StrictModel):
    candidate_id: str
    root_id: str
    judge_model: str
    evolution: JudgeDecision
    correctness: JudgeDecision
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class SpecifyEnsemblePipeline:
    """Persist candidates, sequential judge votes, then majority-voted records."""

    def __init__(self, prompts: PromptCatalog, *, temperature: float = 0.2) -> None:
        self.prompts = prompts
        self.temperature = temperature
        self.format_validator = FormatValidator()
        self.answer_validator = AnswerValidator()

    @staticmethod
    def candidate_id(root_id: str) -> str:
        digest = stable_hash(
            {"root_id": root_id, "strategy": "specify", "generation": 1}
        )[:16]
        return f"cotenem_{digest}"

    def generate_candidates(
        self,
        input_path: str | Path,
        candidate_path: str | Path,
        provider: LLMProvider,
        *,
        limit: int | None = None,
    ) -> int:
        existing = {
            item["id"] for item in read_jsonl(candidate_path) if isinstance(item.get("id"), str)
        }
        generator = InitialCoTGenerator(
            provider, self.prompts, temperature=self.temperature
        )
        agent = SpecifyAgent(provider, self.prompts, temperature=self.temperature)
        execution_id = str(uuid4())
        written = 0
        for raw in read_jsonl(input_path):
            question = NormalizedQuestion.model_validate(raw)
            if not question.eligible:
                continue
            candidate_id = self.candidate_id(question.id)
            if candidate_id in existing:
                continue
            initial = generator.generate(question)
            improved = agent.evolve(question, initial)
            append_jsonl(
                candidate_path,
                SpecifyCandidate(
                    id=candidate_id,
                    root=question,
                    initial=initial,
                    improved=improved,
                    generator_model=str(getattr(provider, "model", type(provider).__name__)),
                    prompt_version=agent.prompt.version,
                    temperature=self.temperature,
                    timestamp=datetime.now(timezone.utc),
                    execution_id=execution_id,
                ),
            )
            existing.add(candidate_id)
            written += 1
            if limit is not None and written >= limit:
                break
        return written

    def import_candidates(
        self,
        input_path: str | Path,
        existing_result_path: str | Path,
        candidate_path: str | Path,
        *,
        limit: int | None = None,
    ) -> int:
        """Reuse prior Qwen outputs so only the filtering method changes."""

        if not Path(existing_result_path).exists():
            raise FileNotFoundError(
                f"prior Specify results not found: {existing_result_path}"
            )
        roots = {
            item["id"]: NormalizedQuestion.model_validate(item)
            for item in read_jsonl(input_path)
        }
        existing = {
            item["id"] for item in read_jsonl(candidate_path) if isinstance(item.get("id"), str)
        }
        written = 0
        for raw in read_jsonl(existing_result_path):
            record = EvolvedRecord.model_validate(raw)
            if record.strategy != Strategy.SPECIFY or record.id in existing:
                continue
            root = roots.get(record.root_id)
            if root is None:
                raise ValueError(f"normalized root not found for prior record: {record.root_id}")
            initial_steps = self._parse_rendered_steps(record.reasoning.initial_cot)
            improved_steps = self._parse_rendered_steps(record.reasoning.improved_cot or "")
            append_jsonl(
                candidate_path,
                SpecifyCandidate(
                    id=record.id,
                    root=root,
                    initial=InitialCoT(
                        reasoning_steps=initial_steps,
                        final_answer=record.reasoning.predicted_answer,
                    ),
                    improved=SpecifiedCoT(
                        reasoning_steps=improved_steps,
                        final_answer=record.reasoning.predicted_answer,
                    ),
                    generator_model=record.generation_metadata.generator_model,
                    prompt_version=record.generation_metadata.prompt_version,
                    temperature=record.generation_metadata.temperature,
                    timestamp=record.generation_metadata.timestamp,
                    execution_id=record.generation_metadata.execution_id,
                ),
            )
            existing.add(record.id)
            written += 1
            if limit is not None and written >= limit:
                break
        return written

    @staticmethod
    def _parse_rendered_steps(value: str) -> list[str]:
        steps = [
            re.sub(r"^\s*\d+\.\s*", "", line).strip()
            for line in value.splitlines()
            if line.strip()
        ]
        if not steps:
            raise ValueError("prior record has no reusable reasoning steps")
        return steps

    def collect_votes(
        self,
        candidate_path: str | Path,
        vote_path: str | Path,
        provider: LLMProvider,
    ) -> int:
        judge_model = str(getattr(provider, "model", type(provider).__name__))
        existing = {
            (item.get("candidate_id"), item.get("judge_model"))
            for item in read_jsonl(vote_path)
        }
        evolution_judge = EvolutionSuccessJudge(provider, self.prompts)
        correctness_judge = CorrectnessJudge(provider, self.prompts)
        written = 0
        for raw in read_jsonl(candidate_path):
            candidate = SpecifyCandidate.model_validate(raw)
            key = (candidate.id, judge_model)
            if key in existing:
                continue
            vote = EnsembleVote(
                candidate_id=candidate.id,
                root_id=candidate.root.id,
                judge_model=judge_model,
                evolution=evolution_judge.judge(
                    candidate.root, candidate.initial, candidate.improved
                ),
                correctness=correctness_judge.judge(
                    candidate.root, candidate.improved
                ),
            )
            append_jsonl(vote_path, vote)
            existing.add(key)
            written += 1
        return written

    def aggregate(
        self,
        candidate_path: str | Path,
        vote_path: str | Path,
        output_path: str | Path,
        judge_models: list[str],
    ) -> int:
        if len(judge_models) != 3 or len(set(judge_models)) != 3:
            raise ValueError("majority voting requires exactly three distinct judge models")
        votes_by_candidate: dict[str, dict[str, EnsembleVote]] = {}
        for raw in read_jsonl(vote_path):
            vote = EnsembleVote.model_validate(raw)
            if vote.judge_model in judge_models:
                votes_by_candidate.setdefault(vote.candidate_id, {})[vote.judge_model] = vote
        existing = {
            item["id"] for item in read_jsonl(output_path) if isinstance(item.get("id"), str)
        }
        written = 0
        for raw in read_jsonl(candidate_path):
            candidate = SpecifyCandidate.model_validate(raw)
            if candidate.id in existing:
                continue
            model_votes = votes_by_candidate.get(candidate.id, {})
            if set(model_votes) != set(judge_models):
                continue
            ordered_votes = [model_votes[model] for model in judge_models]
            evolution_success = sum(vote.evolution.approved for vote in ordered_votes) >= 2
            correctness_verified = sum(vote.correctness.approved for vote in ordered_votes) >= 2
            initial_format = self.format_validator.validate(candidate.initial)
            improved_format = self.format_validator.validate(candidate.improved)
            initial_answer = self.answer_validator.validate(
                candidate.initial.final_answer, candidate.root.gold_answer
            )
            improved_answer = self.answer_validator.validate(
                candidate.improved.final_answer, candidate.root.gold_answer
            )
            judge_votes = [
                JudgeVote(
                    judge_model=vote.judge_model,
                    evolution_success=vote.evolution.approved,
                    correctness_verified=vote.correctness.approved,
                    evolution_reasons=vote.evolution.reasons,
                    correctness_reasons=vote.correctness.reasons,
                )
                for vote in ordered_votes
            ]
            reasons = [
                *initial_format.reasons,
                *improved_format.reasons,
                *([] if initial_answer.matches else [initial_answer.reason]),
                *([] if improved_answer.matches else ["specified_answer_mismatch"]),
                *([] if evolution_success else ["evolution: majority_rejected"]),
                *([] if correctness_verified else ["correctness: majority_rejected"]),
            ]
            accepted = (
                initial_format.valid
                and improved_format.valid
                and initial_answer.matches
                and improved_answer.matches
                and evolution_success
                and correctness_verified
            )
            append_jsonl(
                output_path,
                EvolvedRecord(
                    id=candidate.id,
                    root_id=candidate.root.id,
                    parent_id=candidate.root.id,
                    generation=1,
                    strategy=Strategy.SPECIFY,
                    question=QuestionContent(
                        header=candidate.root.header,
                        statement=candidate.root.statement,
                        alternatives=candidate.root.alternatives,
                        gold_answer=candidate.root.gold_answer,
                    ),
                    reasoning=Reasoning(
                        initial_cot=render_steps(candidate.initial.reasoning_steps),
                        improved_cot=render_steps(candidate.improved.reasoning_steps),
                        predicted_answer=candidate.improved.final_answer,
                    ),
                    generation_metadata=GenerationMetadata(
                        generator_model=candidate.generator_model,
                        judge_model="majority:" + ",".join(judge_models),
                        prompt_version=candidate.prompt_version,
                        temperature=candidate.temperature,
                        timestamp=candidate.timestamp,
                        execution_id=candidate.execution_id,
                    ),
                    validation=ValidationResult(
                        format_valid=initial_format.valid and improved_format.valid,
                        answer_matches_gold=initial_answer.matches and improved_answer.matches,
                        correctness_verified=correctness_verified,
                        evolution_success=evolution_success,
                        enem_style_valid=True,
                        accepted=accepted,
                        rejection_reasons=[reason for reason in reasons if reason],
                        judge_votes=judge_votes,
                    ),
                    status=RecordStatus.ACCEPTED if accepted else RecordStatus.REJECTED,
                ),
            )
            existing.add(candidate.id)
            written += 1
        return written
