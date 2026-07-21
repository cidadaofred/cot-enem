"""Command-line entry points available in the current phase."""
import argparse
import json
from pathlib import Path

from cot_enem.agents.specify import SpecifyAgent
from cot_enem.config import PromptCatalog, load_env_file
from cot_enem.configuration import load_application_config
from cot_enem.dataset.repository import QuestionRepository
from cot_enem.dataset.schema import Strategy
from cot_enem.ensemble_pipeline import SpecifyEnsemblePipeline
from cot_enem.generation.initial_cot import InitialCoTGenerator
from cot_enem.observability import configure_logging
from cot_enem.phase4_pipeline import QuestionEvolutionEnsemblePipeline
from cot_enem.phase6 import finalize_phase6
from cot_enem.pipeline import SpecifyPipeline
from cot_enem.providers.base import LLMProvider
from cot_enem.providers.huggingface_provider import HuggingFaceProvider
from cot_enem.providers.openai_compatible import OpenAICompatibleProvider
from cot_enem.runtime.context import build_execution_context
from cot_enem.runtime.diagnostics import environment_summary, verify_environment
from cot_enem.validation.correctness_judge import CorrectnessJudge
from cot_enem.validation.evolution_judge import EvolutionSuccessJudge

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _project_path(path: str) -> Path:
    candidate = Path(path)
    if candidate.exists() or candidate.is_absolute():
        return candidate
    return PROJECT_ROOT / candidate


def _add_runtime_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", help="Environment/profile YAML merged over defaults")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--precision", choices=["auto", "fp32", "fp16", "bf16"])
    parser.add_argument("--output-dir")


def _load_context(args: argparse.Namespace):
    overrides = {
        "runtime.device": getattr(args, "device", None),
        "runtime.precision": getattr(args, "precision", None),
        "storage.base_path": getattr(args, "output_dir", None),
    }
    loaded = load_application_config(
        default_path=_project_path("configs/default.yaml"),
        profile_path=_project_path(args.config) if getattr(args, "config", None) else None,
        cli_overrides=overrides,
    )
    context = build_execution_context(loaded)
    logging_config = context.loaded_config.config.logging
    log_file = (
        context.output_directory / "logs" / "cot-enem.jsonl"
        if logging_config.save_to_file
        else None
    )
    logger = configure_logging(
        level=logging_config.level,
        log_file=log_file,
        structured_file=logging_config.structured,
    )
    logger.info(
        "runtime_resolved",
        extra={
            "environment": context.environment.environment,
            "device": context.device.device,
            "precision": context.device.precision,
            "model": context.loaded_config.config.model.name,
        },
    )
    return context


def _build_providers(context) -> tuple[LLMProvider, LLMProvider]:
    """Build providers without loading the same Hugging Face model twice."""
    config = context.loaded_config.config.model
    if config.provider == "huggingface":
        provider = HuggingFaceProvider(
            model=config.name,
            device=context.device.device,
            precision=context.device.precision,
            quantization=config.quantization,
            max_new_tokens=config.max_new_tokens,
        )
        return provider, provider
    if config.provider == "openai_compatible":
        return (
            OpenAICompatibleProvider(
                model=config.name,
                timeout=config.timeout_seconds,
                max_attempts=config.max_attempts,
            ),
            OpenAICompatibleProvider(
                model=config.judge_name or config.name,
                timeout=config.timeout_seconds,
                max_attempts=config.max_attempts,
            ),
        )
    raise ValueError("provider='mock' is reserved for tests and cannot run from the CLI")


def _build_huggingface_provider(
    context,
    model: str,
    *,
    max_new_tokens: int | None = None,
) -> HuggingFaceProvider:
    config = context.loaded_config.config.model
    return HuggingFaceProvider(
        model=model,
        device=context.device.device,
        precision=context.device.precision,
        quantization=config.quantization,
        max_new_tokens=max_new_tokens or config.max_new_tokens,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="cot-enem")
    commands = parser.add_subparsers(dest="command", required=True)
    verify = commands.add_parser("verify", help="Validate runtime prerequisites")
    _add_runtime_arguments(verify)
    info = commands.add_parser("info", help="Show detected environment and resolved config")
    _add_runtime_arguments(info)
    prepare = commands.add_parser("prepare", help="Normalize ENEM XML into JSONL")
    prepare.add_argument("--input", required=True)
    prepare.add_argument("--output", required=True)
    prepare.add_argument("--year", type=int)
    evolve = commands.add_parser("evolve", help="Generate CoT-ENEM evolved records")
    evolve.add_argument("--strategies", nargs="+", choices=["specify"], required=True)
    evolve.add_argument("--input", required=True)
    evolve.add_argument("--output", required=True)
    evolve.add_argument("--limit", type=int)
    evolve.add_argument("--prompts", default="config/prompts.yaml")
    _add_runtime_arguments(evolve)
    ensemble = commands.add_parser(
        "ensemble-specify",
        help="Generate Specify candidates and apply three sequential majority judges",
    )
    ensemble.add_argument("--input", required=True)
    ensemble.add_argument("--candidates", required=True)
    ensemble.add_argument("--votes", required=True)
    ensemble.add_argument("--output", required=True)
    ensemble.add_argument(
        "--existing-results",
        help="Import prior Specify records instead of regenerating Qwen candidates",
    )
    ensemble.add_argument("--limit", type=int)
    ensemble.add_argument("--prompts", default="config/prompts.yaml")
    _add_runtime_arguments(ensemble)
    phase4 = commands.add_parser(
        "ensemble-evolve",
        help="Run an independent Complicate or Diversify majority-voted branch",
    )
    phase4.add_argument(
        "--strategy",
        choices=["complicate", "diversify"],
        required=True,
    )
    phase4.add_argument("--seeds", required=True)
    phase4_source = phase4.add_mutually_exclusive_group(required=True)
    phase4_source.add_argument(
        "--initial-candidates",
        help="Extract frozen initial CoTs from Phase 3 candidates",
    )
    phase4_source.add_argument(
        "--parent-results",
        help="Use accepted prior-generation results for iterative evolution",
    )
    phase4.add_argument("--candidates", required=True)
    phase4.add_argument("--votes", required=True)
    phase4.add_argument("--output", required=True)
    phase4.add_argument("--limit", type=int)
    phase4.add_argument("--prompts", default="config/prompts.yaml")
    _add_runtime_arguments(phase4)
    finalize = commands.add_parser(
        "finalize",
        help="Consolidate accepted/rejected records and produce final CPU-only reports",
    )
    finalize.add_argument("--normalized", required=True)
    finalize.add_argument("--specify", required=True)
    finalize.add_argument("--complicate", required=True)
    finalize.add_argument("--diversify", required=True)
    finalize.add_argument("--output-dir", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    load_env_file(_project_path(".env"))
    args = build_parser().parse_args(argv)
    if args.command == "verify":
        context = _load_context(args)
        checks = verify_environment(context)
        for check in checks:
            print(f"[{'OK' if check.ok else 'FAIL'}] {check.name}: {check.detail}")
        return 0 if all(check.ok for check in checks) else 1
    if args.command == "info":
        context = _load_context(args)
        print(json.dumps(environment_summary(context), ensure_ascii=False, indent=2))
        return 0
    if args.command == "prepare":
        repository = QuestionRepository()
        count = repository.prepare(args.input, args.output, year=args.year)
        print(f"records_written={count} records_rejected={repository.last_rejected_count}")
    elif args.command == "finalize":
        summary = finalize_phase6(
            args.normalized,
            args.specify,
            args.complicate,
            args.diversify,
            args.output_dir,
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    elif args.command == "evolve":
        context = _load_context(args)
        config = context.loaded_config.config
        prompts = PromptCatalog.from_yaml(_project_path(args.prompts))
        generator_provider, judge_provider = _build_providers(context)
        pipeline = SpecifyPipeline(
            InitialCoTGenerator(
                generator_provider, prompts, temperature=config.model.temperature
            ),
            SpecifyAgent(generator_provider, prompts, temperature=config.model.temperature),
            EvolutionSuccessJudge(judge_provider, prompts),
            CorrectnessJudge(judge_provider, prompts),
            generator_provider=generator_provider,
            judge_provider=judge_provider,
            temperature=config.model.temperature,
        )
        count = pipeline.run(
            args.input,
            args.output,
            limit=args.limit if args.limit is not None else config.pipeline.limit,
        )
        print(
            f"environment={context.environment.environment} "
            f"device={context.device.device} precision={context.device.precision}"
        )
        print(f"records_written={count} execution_id={pipeline.execution_id}")
    elif args.command == "ensemble-specify":
        context = _load_context(args)
        config = context.loaded_config.config
        if config.model.provider != "huggingface":
            raise ValueError("ensemble-specify currently requires provider='huggingface'")
        if len(config.model.judge_names) != 3 or len(set(config.model.judge_names)) != 3:
            raise ValueError(
                "model.judge_names must contain exactly three distinct Hugging Face models"
            )
        prompts = PromptCatalog.from_yaml(_project_path(args.prompts))
        ensemble = SpecifyEnsemblePipeline(
            prompts, temperature=config.model.temperature
        )
        effective_limit = (
            args.limit if args.limit is not None else config.pipeline.limit
        )
        if args.existing_results:
            generated = ensemble.import_candidates(
                args.input,
                args.existing_results,
                args.candidates,
                limit=effective_limit,
            )
        else:
            generator_provider = _build_huggingface_provider(context, config.model.name)
            print(
                f"phase=candidate_generation role=generator "
                f"model={config.model.name}",
                flush=True,
            )
            generated = ensemble.generate_candidates(
                args.input,
                args.candidates,
                generator_provider,
                limit=effective_limit,
            )
            generator_provider.unload()
        print(f"candidates_written={generated}")
        for judge_index, judge_model in enumerate(config.model.judge_names, start=1):
            print(
                f"phase=judge_voting role=judge "
                f"judge={judge_index}/3 model={judge_model}",
                flush=True,
            )
            judge_provider = _build_huggingface_provider(context, judge_model)
            votes = ensemble.collect_votes(
                args.candidates,
                args.votes,
                judge_provider,
                limit=effective_limit,
            )
            judge_provider.unload()
            print(f"judge_model={judge_model} votes_written={votes}")
        aggregated = ensemble.aggregate(
            args.candidates,
            args.votes,
            args.output,
            config.model.judge_names,
            limit=effective_limit,
        )
        print(f"records_written={aggregated} voting=majority judges=3")
    elif args.command == "ensemble-evolve":
        context = _load_context(args)
        config = context.loaded_config.config
        if config.model.provider != "huggingface":
            raise ValueError("ensemble-evolve currently requires provider='huggingface'")
        if len(config.model.judge_names) != 3 or len(set(config.model.judge_names)) != 3:
            raise ValueError(
                "model.judge_names must contain exactly three distinct Hugging Face models"
            )
        strategy = Strategy(args.strategy)
        prompts = PromptCatalog.from_yaml(_project_path(args.prompts))
        pipeline = QuestionEvolutionEnsemblePipeline(
            prompts,
            strategy,
            temperature=config.model.temperature,
        )
        effective_limit = (
            args.limit if args.limit is not None else config.pipeline.limit
        )
        if args.initial_candidates:
            seeds = pipeline.import_phase3_seeds(
                args.initial_candidates,
                args.seeds,
                limit=effective_limit,
            )
            seed_source = "phase3_initial_cot"
        else:
            seeds = pipeline.import_accepted_results(
                args.parent_results,
                args.seeds,
                limit=effective_limit,
            )
            seed_source = "accepted_parent_results"
        print(f"seeds_written={seeds} source={seed_source}")
        phase4_max_new_tokens = max(config.model.max_new_tokens, 1024)
        generator_provider = _build_huggingface_provider(
            context,
            config.model.name,
            max_new_tokens=phase4_max_new_tokens,
        )
        print(
            f"phase=candidate_generation strategy={strategy.value} "
            f"model={config.model.name} max_new_tokens={phase4_max_new_tokens}",
            flush=True,
        )
        generated = pipeline.generate_candidates(
            args.seeds,
            args.candidates,
            generator_provider,
            limit=effective_limit,
        )
        generator_provider.unload()
        print(f"candidates_written={generated}")
        for judge_index, judge_model in enumerate(config.model.judge_names, start=1):
            print(
                f"phase=judge_voting strategy={strategy.value} "
                f"judge={judge_index}/3 model={judge_model}",
                flush=True,
            )
            judge_provider = _build_huggingface_provider(context, judge_model)
            votes = pipeline.collect_votes(
                args.candidates,
                args.votes,
                judge_provider,
                limit=effective_limit,
            )
            judge_provider.unload()
            print(f"judge_model={judge_model} votes_written={votes}")
        aggregated = pipeline.aggregate(
            args.candidates,
            args.votes,
            args.output,
            config.model.judge_names,
            limit=effective_limit,
        )
        print(
            f"records_written={aggregated} strategy={strategy.value} "
            "voting=majority judges=3"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
