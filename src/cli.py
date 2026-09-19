"""ABOUTME: Command-line argument parsing for the function-calling tool."""

import argparse
from dataclasses import dataclass
from pathlib import Path

DEFAULT_FUNCTIONS_DEFINITION = Path("data/input/functions_definition.json")
DEFAULT_INPUT = Path("data/input/function_calling_tests.json")
DEFAULT_OUTPUT = Path("data/output/function_calling_results.json")
DEFAULT_MODEL_NAME = "Qwen/Qwen3-0.6B"


@dataclass
class CliArgs:
    functions_definition: Path
    input: Path
    output: Path
    model_name: str
    max_tokens: int
    verbose: bool
    stream: bool


def parse_args(argv: list[str] | None = None) -> CliArgs:
    parser = argparse.ArgumentParser(
        prog="python -m src",
        description="Translate natural-language prompts into structured "
        "function calls using constrained decoding.",
    )
    parser.add_argument(
        "--functions_definition",
        type=Path,
        default=DEFAULT_FUNCTIONS_DEFINITION,
        help="Path to the function definitions JSON file"
        f"(default: {DEFAULT_FUNCTIONS_DEFINITION})",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help=f"Path to the prompts JSON file (default: {DEFAULT_INPUT})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Path to write results to (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default=DEFAULT_MODEL_NAME,
        help=f"Hugging Face model identifier (default: {DEFAULT_MODEL_NAME})",
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=64,
        help="Safety cap on generated tokens per field (default: 64)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable debug-level logging",
    )
    parser.add_argument(
        "--stream",
        action="store_true",
        help="Print constrained output as it is generated",
    )
    args = parser.parse_args(argv)
    return CliArgs(
        functions_definition=args.functions_definition,
        input=args.input,
        output=args.output,
        model_name=args.model_name,
        max_tokens=args.max_tokens,
        verbose=args.verbose,
        stream=args.stream,
    )
