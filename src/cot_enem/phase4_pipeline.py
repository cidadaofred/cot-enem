"""Independent Complicate/Diversify branches over reusable evolution seeds."""

from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from pydantic import Field

from cot_enem.agents.complicate import ComplicateAgent
from cot_enem.agents.diversify import DiversifyAgent
from cot_enem.agents.question_evolution import QuestionEvolutionAgent
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
from cot_enem.ensemble_pipeline import EnsembleVote, SpecifyCandidate
from cot_enem.generation.prompting import render_steps
from cot_enem.generation.schema import EvolvedQuestionCoT, InitialCoT
from cot_enem.providers.base import LLMProvider
from cot_enem.utils.hashing import stable_hash
from cot_enem.utils.jsonl import append_jsonl, read_jsonl
from cot_enem.validation.correctness_judge import CorrectnessJudge
from cot_enem.validation.format_validator import FormatValidator
from cot_enem.validation.question_evolution_judge import (
    QuestionEvolutionSuccessJudge,
)


class EvolutionSeed(StrictModel):
    """Strategy-neutral input that can later represent any accepted generation."""

    id: str
    root_id: str
    parent_id: str
    generation: int = Field(ge=0)
    question: NormalizedQuestion
    cot: InitialCoT
    generator_model: str
    source_artifact_id: str


class QuestionEvolutionCandidate(StrictModel):
    id: str
    root_id: str
    parent_id: str
    generation: int = Field(ge=1)
    strategy: Strategy
    source: NormalizedQuestion
    initial: InitialCoT
    evolved: EvolvedQuestionCoT
    generator_model: str
    prompt_version: str
    temperature: float
    timestamp: datetime
    execution_id: str


class QuestionEvolutionEnsemblePipeline:
    """Generate and majority-filter one independent question-evolution branch."""

    def __init__(
        self,
        prompts: PromptCatalog,
        strategy: Strategy,
        *,
        temperature: float = 0.2,
    ) -> None:
        if strategy not in {Strategy.COMPLICATE, Strategy.DIVERSIFY}:
            raise ValueError("Phase 4 supports complicate or diversify")
        self.prompts = prompts
        self.strategy = strategy
        self.temperature = temperature
        self.format_validator = FormatValidator()

    @staticmethod
    def initial_seed_id(root_id: str) -> str:
        digest = stable_hash({"root_id": root_id, "artifact": "initial_cot"})[:16]
        return f"cotseed_{digest}"

    def candidate_id(self, seed: EvolutionSeed) -> str:
        digest = stable_hash(
            {
                "root_id": seed.root_id,
                "parent_id": seed.id,
                "strategy": self.strategy.value,
                "generation": seed.generation + 1,
            }
        )[:16]
        return f"cotenem_{digest}"

    def import_phase3_seeds(
        self,
        phase3_candidates: str | Path,
        seed_path: str | Path,
        *,
        limit: int | None = None,
    ) -> int:
        """Extract only frozen initial CoTs; never depend on Specify improvements."""

        existing = {
            item["id"] for item in read_jsonl(seed_path) if isinstance(item.get("id"), str)
        }
        written = 0
        for index, raw in enumerate(read_jsonl(phase3_candidates), start=1):
            if limit is not None and index > limit:
                break
            source = SpecifyCandidate.model_validate(raw)
            seed_id = self.initial_seed_id(source.root.id)
            if seed_id in existing:
                continue
            append_jsonl(
                seed_path,
                EvolutionSeed(
                    id=seed_id,
                    root_id=source.root.id,
                    parent_id=source.root.id,
                    generation=0,
                    question=source.root,
                    cot=source.initial,
                    generator_model=source.generator_model,
                    source_artifact_id=source.id,
                ),
            )
            existing.add(seed_id)
            written += 1
        return written

    def import_accepted_results(
        self,
        result_path: str | Path,
        seed_path: str | Path,
        *,
        limit: int | None = None,
    ) -> int:
        """Optional adapter for future article-like iterative evolution rounds."""

        existing = {
            item["id"] for item in read_jsonl(seed_path) if isinstance(item.get("id"), str)
        }
        written = 0
        selected = 0
        for raw in read_jsonl(result_path):
            result = EvolvedRecord.model_validate(raw)
            if not result.validation.accepted:
                continue
            selected += 1
            if limit is not None and selected > limit:
                break
            seed_id = f"cotseed_{stable_hash({'parent_id': result.id})[:16]}"
            if seed_id in existing:
                continue
            append_jsonl(
                seed_path,
                EvolutionSeed(
                    id=seed_id,
                    root_id=result.root_id,
                    parent_id=result.id,
                    generation=result.generation,
                    question=NormalizedQuestion(
                        id=result.id,
                        header=result.question.header,
                        statement=result.question.statement,
                        alternatives=result.question.alternatives,
                        gold_answer=result.question.gold_answer,
                    ),
                    cot=InitialCoT(
                        reasoning_steps=self._parse_steps(
                            result.reasoning.improved_cot
                            or result.reasoning.initial_cot
                        ),
                        final_answer=result.reasoning.predicted_answer,
                    ),
                    generator_model=result.generation_metadata.generator_model,
                    source_artifact_id=result.id,
                ),
            )
            existing.add(seed_id)
            written += 1
        return written

    @staticmethod
    def _parse_steps(value: str) -> list[str]:
        steps = [
            line.split(". ", 1)[1].strip()
            if line.strip().split(". ", 1)[0].isdigit() and ". " in line
            else line.strip()
            for line in value.splitlines()
            if line.strip()
        ]
        if not steps:
            raise ValueError("evolution seed has no reasoning steps")
        return steps

    def _agent(self, provider: LLMProvider) -> QuestionEvolutionAgent:
        agent_type = (
            ComplicateAgent
            if self.strategy == Strategy.COMPLICATE
            else DiversifyAgent
        )
        return agent_type(provider, self.prompts, temperature=self.temperature)

    def generate_candidates(
        self,
        seed_path: str | Path,
        candidate_path: str | Path,
        provider: LLMProvider,
        *,
        limit: int | None = None,
    ) -> int:
        existing = {
            item["id"] for item in read_jsonl(candidate_path) if isinstance(item.get("id"), str)
        }
        agent = self._agent(provider)
        execution_id = str(uuid4())
        written = 0
        for index, raw in enumerate(read_jsonl(seed_path), start=1):
            if limit is not None and index > limit:
                break
            seed = EvolutionSeed.model_validate(raw)
            candidate_id = self.candidate_id(seed)
            if candidate_id in existing:
                continue
            evolved = agent.evolve(seed.question, seed.cot)
            self._evolved_question(candidate_id, seed, evolved)
            append_jsonl(
                candidate_path,
                QuestionEvolutionCandidate(
                    id=candidate_id,
                    root_id=seed.root_id,
                    parent_id=seed.id,
                    generation=seed.generation + 1,
                    strategy=self.strategy,
                    source=seed.question,
                    initial=seed.cot,
                    evolved=evolved,
                    generator_model=str(
                        getattr(provider, "model", type(provider).__name__)
                    ),
                    prompt_version=agent.prompt.version,
                    temperature=self.temperature,
                    timestamp=datetime.now(timezone.utc),
                    execution_id=execution_id,
                ),
            )
            existing.add(candidate_id)
            written += 1
        return written

    @staticmethod
    def _evolved_question(
        candidate_id: str,
        seed: EvolutionSeed | QuestionEvolutionCandidate,
        evolved: EvolvedQuestionCoT,
    ) -> NormalizedQuestion:
        return NormalizedQuestion(
            id=candidate_id,
            year=seed.question.year if isinstance(seed, EvolutionSeed) else seed.source.year,
            area=seed.question.area if isinstance(seed, EvolutionSeed) else seed.source.area,
            header=evolved.header,
            statement=evolved.statement,
            alternatives=evolved.alternatives,
            gold_answer=evolved.final_answer,
        )

    def collect_votes(
        self,
        candidate_path: str | Path,
        vote_path: str | Path,
        provider: LLMProvider,
        *,
        limit: int | None = None,
    ) -> int:
        judge_model = str(getattr(provider, "model", type(provider).__name__))
        existing = {
            (item.get("candidate_id"), item.get("judge_model"))
            for item in read_jsonl(vote_path)
        }
        evolution_judge = QuestionEvolutionSuccessJudge(provider, self.prompts)
        correctness_judge = CorrectnessJudge(provider, self.prompts)
        written = 0
        for index, raw in enumerate(read_jsonl(candidate_path), start=1):
            if limit is not None and index > limit:
                break
            candidate = QuestionEvolutionCandidate.model_validate(raw)
            key = (candidate.id, judge_model)
            if key in existing:
                continue
            evolved_question = self._evolved_question(
                candidate.id, candidate, candidate.evolved
            )
            append_jsonl(
                vote_path,
                EnsembleVote(
                    candidate_id=candidate.id,
                    root_id=candidate.root_id,
                    judge_model=judge_model,
                    evolution=evolution_judge.judge(
                        candidate.strategy,
                        candidate.source,
                        candidate.initial,
                        candidate.evolved,
                        evolved_question,
                    ),
                    correctness=correctness_judge.judge(
                        evolved_question, candidate.evolved
                    ),
                ),
            )
            existing.add(key)
            written += 1
        return written

    def aggregate(
        self,
        candidate_path: str | Path,
        vote_path: str | Path,
        output_path: str | Path,
        judge_models: list[str],
        *,
        limit: int | None = None,
    ) -> int:
        if len(judge_models) != 3 or len(set(judge_models)) != 3:
            raise ValueError("majority voting requires exactly three distinct judges")
        votes_by_candidate: dict[str, dict[str, EnsembleVote]] = {}
        for raw in read_jsonl(vote_path):
            vote = EnsembleVote.model_validate(raw)
            if vote.judge_model in judge_models:
                votes_by_candidate.setdefault(vote.candidate_id, {})[
                    vote.judge_model
                ] = vote
        existing = {
            item["id"] for item in read_jsonl(output_path) if isinstance(item.get("id"), str)
        }
        written = 0
        for index, raw in enumerate(read_jsonl(candidate_path), start=1):
            if limit is not None and index > limit:
                break
            candidate = QuestionEvolutionCandidate.model_validate(raw)
            if candidate.id in existing:
                continue
            model_votes = votes_by_candidate.get(candidate.id, {})
            if set(model_votes) != set(judge_models):
                continue
            ordered = [model_votes[model] for model in judge_models]
            evolution_success = sum(vote.evolution.approved for vote in ordered) >= 2
            correctness_verified = (
                sum(vote.correctness.approved for vote in ordered) >= 2
            )
            evolved_format = self.format_validator.validate(candidate.evolved)
            accepted = (
                evolved_format.valid
                and evolution_success
                and correctness_verified
            )
            reasons = [
                *evolved_format.reasons,
                *([] if evolution_success else ["evolution: majority_rejected"]),
                *([] if correctness_verified else ["correctness: majority_rejected"]),
            ]
            append_jsonl(
                output_path,
                EvolvedRecord(
                    id=candidate.id,
                    root_id=candidate.root_id,
                    parent_id=candidate.parent_id,
                    generation=candidate.generation,
                    strategy=candidate.strategy,
                    question=QuestionContent(
                        header=candidate.evolved.header,
                        statement=candidate.evolved.statement,
                        alternatives=candidate.evolved.alternatives,
                        gold_answer=candidate.evolved.final_answer,
                    ),
                    reasoning=Reasoning(
                        initial_cot=render_steps(candidate.initial.reasoning_steps),
                        improved_cot=render_steps(candidate.evolved.reasoning_steps),
                        predicted_answer=candidate.evolved.final_answer,
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
                        format_valid=evolved_format.valid,
                        answer_matches_gold=True,
                        correctness_verified=correctness_verified,
                        evolution_success=evolution_success,
                        enem_style_valid=True,
                        accepted=accepted,
                        rejection_reasons=[reason for reason in reasons if reason],
                        judge_votes=[
                            JudgeVote(
                                judge_model=vote.judge_model,
                                evolution_success=vote.evolution.approved,
                                correctness_verified=vote.correctness.approved,
                                evolution_reasons=vote.evolution.reasons,
                                correctness_reasons=vote.correctness.reasons,
                            )
                            for vote in ordered
                        ],
                    ),
                    status=(
                        RecordStatus.ACCEPTED if accepted else RecordStatus.REJECTED
                    ),
                ),
            )
            existing.add(candidate.id)
            written += 1
        return written
