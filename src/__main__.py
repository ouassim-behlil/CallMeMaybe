"""ABOUTME: Entry point for `python -m src` — wires CLI args, I/O, the model,
and the pipeline together with a single top-level safety net."""

import logging
import sys
import time

from .cli import parse_args
from .io_utils import (
    DataFileError,
    load_function_definitions,
    load_prompts,
    save_output
)
from .pipeline import run_pipeline


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    logger = logging.getLogger(__name__)

    try:
        functions_file = load_function_definitions(args.functions_definition)
        prompts_file = load_prompts(args.input)
    except DataFileError as exc:
        logger.error("%s", exc)
        return 1

    # Imported lazily so `--help` and file-validation errors above return
    # instantly, without paying the cost of loading the model first.
    try:
        from llm_sdk import Small_LLM_Model
    except ImportError as exc:
        logger.error(
            "Could not import llm_sdk: %s",
            exc
        )
        return 1

    logger.info("Loading model %r ...", args.model_name)
    start = time.monotonic()
    try:
        model = Small_LLM_Model(model_name=args.model_name)
    except Exception as exc:
        logger.error("Failed to load model %r: %s", args.model_name, exc)
        return 1
    logger.info("Model loaded in %.1fs", time.monotonic() - start)

    try:
        start = time.monotonic()
        results = run_pipeline(
            model,
            functions_file.root,
            prompts_file.root,
            max_tokens=args.max_tokens,
            stream=args.stream,
        )
        elapsed = time.monotonic() - start
        logger.info("Generation finished in %.1fs", elapsed)

        save_output(args.output, results)
        logger.info("Wrote %d results to %s", len(results), args.output)
    except DataFileError as exc:
        logger.error("%s", exc)
        return 1
    except Exception:
        # Last-resort net: nothing below this should ever be an unhandled
        # crash reaching the reviewer's terminal. Log the full traceback
        # for our own debugging, but still exit cleanly.
        logger.exception("Unexpected error during generation")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
