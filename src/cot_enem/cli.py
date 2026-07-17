"""Command-line entry points available in the current phase."""
import argparse
import os

from cot_enem.agents.specify import SpecifyAgent
from cot_enem.config import PromptCatalog, load_env_file
from cot_enem.dataset.repository import QuestionRepository
from cot_enem.generation.initial_cot import InitialCoTGenerator
from cot_enem.pipeline import SpecifyPipeline
from cot_enem.providers.openai_compatible import OpenAICompatibleProvider
from cot_enem.validation.correctness_judge import CorrectnessJudge
from cot_enem.validation.evolution_judge import EvolutionSuccessJudge

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="cot-enem")
    commands = parser.add_subparsers(dest="command", required=True)
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
    return parser

def main(argv: list[str] | None = None) -> int:
    load_env_file()
    args = build_parser().parse_args(argv)
    if args.command == "prepare":
        count = QuestionRepository().prepare(args.input, args.output, year=args.year)
        print(f"records_written={count}")
    elif args.command == "evolve":
        prompts = PromptCatalog.from_yaml(args.prompts)
        generator_provider = OpenAICompatibleProvider()
        judge_provider = OpenAICompatibleProvider(model=os.getenv("JUDGE_MODEL") or None)
        pipeline = SpecifyPipeline(
            InitialCoTGenerator(generator_provider, prompts),
            SpecifyAgent(generator_provider, prompts),
            EvolutionSuccessJudge(judge_provider, prompts),
            CorrectnessJudge(judge_provider, prompts),
            generator_provider=generator_provider,
            judge_provider=judge_provider,
        )
        count = pipeline.run(args.input, args.output, limit=args.limit)
        print(f"records_written={count} execution_id={pipeline.execution_id}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
