"""Command-line entry points available in the current phase."""
import argparse
import json
from pathlib import Path

from cot_enem.agents.specify import SpecifyAgent
from cot_enem.config import PromptCatalog, load_env_file
from cot_enem.configuration import load_application_config
from cot_enem.dataset.repository import QuestionRepository
from cot_enem.generation.initial_cot import InitialCoTGenerator
from cot_enem.observability import configure_logging
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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
