"""Thin convenience wrapper; project logic remains under src."""
import sys
from cot_enem.cli import main

if __name__ == "__main__":
    raise SystemExit(main(["prepare", "--input", sys.argv[1], "--output", sys.argv[2]]))
