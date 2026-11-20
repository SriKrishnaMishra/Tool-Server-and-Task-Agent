"""Command-line entry point for the Bodhan task agent."""

import argparse
import math
from pathlib import Path

from .agent import run_task


def _non_negative_float(value: str) -> float:
    try:
        parsed = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a number") from exc
    if parsed < 0 or not math.isfinite(parsed):
        raise argparse.ArgumentTypeError("must be a finite, non-negative number")
    return parsed


def _non_negative_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be an integer") from exc
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be non-negative")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    """Build the stable CLI parser."""
    parser = argparse.ArgumentParser(description="Run a bounded Bodhan language task.")
    parser.add_argument("--task", required=True, help="Natural-language task to perform.")
    parser.add_argument("--files", required=True, type=Path, help="Input file or directory.")
    parser.add_argument("--out", required=True, type=Path, help="Output directory.")
    parser.add_argument("--max-steps", default=30, type=_non_negative_int)
    parser.add_argument("--max-cost-inr", default=20.0, type=_non_negative_float)
    parser.add_argument("--max-seconds", default=300.0, type=_non_negative_float)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Parse arguments and execute one task run."""
    parser = build_parser()
    args = parser.parse_args(argv)
    if not args.task.strip():
        parser.error("--task must not be empty")
    return run_task(
        args.task,
        args.files,
        args.out,
        max_steps=args.max_steps,
        max_cost_inr=args.max_cost_inr,
        max_seconds=args.max_seconds,
    )


if __name__ == "__main__":
    raise SystemExit(main())
