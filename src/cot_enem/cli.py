"""Command-line entry points available in the current phase."""
import argparse
from cot_enem.dataset.repository import QuestionRepository

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="cot-enem")
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare", help="Normalize ENEM XML into JSONL")
    prepare.add_argument("--input", required=True)
    prepare.add_argument("--output", required=True)
    prepare.add_argument("--year", type=int)
    return parser

def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "prepare":
        count = QuestionRepository().prepare(args.input, args.output, year=args.year)
        print(f"records_written={count}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
