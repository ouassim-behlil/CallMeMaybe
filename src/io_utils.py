import json
from pathlib import Path
from typing import TypeVar, Type, List

from pydantic import BaseModel, ValidationError

from src import FunctionsDefinitionFile, PromptsFile, OutputItem, OutputFile
T = TypeVar("T", bound=BaseModel)

class DataFileError(Exception):
    """Raised for any problem loading or validating an input/output file.

        Carries a human-readable message; callers (the CLI) are responsible
        for printing it and choosing an exit code. Never raised with a bare
        traceback-only message — always say *which file* and *what* was wrong.
        """

def read_json(path: Path) -> object:
    """Read and parser a JSON file and raise a DataFileError if it fails."""

    try:
        with path.open(mode="r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError as exc:
        raise DataFileError(f"File not found {path}") from exc
    except PermissionError as exc:
        raise DataFileError(f"Permission denied {path}") from exc
    except json.JSONDecodeError as exc:
        raise DataFileError(f"Invalid JSON {path}: (line {exc.lineno}, column {exc.colno}): {exc.msg}")
    except UnicodeDecodeError as exc:
        raise DataFileError(f"File is not valid UTF-8 {path}") from exc

def _validate(model_cls: Type[T], raw: object, path: Path) -> T:
    """Validate raw JSON Data against a pydantic model, wrapping errors."""

    try:
       return model_cls.model_validate(raw)
    except ValidationError as exc:
        raise DataFileError(f"Scheme error in {path}:\n{exc}") from exc

def load_function_definitions(path: Path) -> FunctionsDefinitionFile:
    """ Load and validate functions_definition.json """

    raw = read_json(path)
    return _validate(FunctionsDefinitionFile, raw, path)

def load_prompts(path: Path) -> PromptsFile:
    """ Load and validate prompts.json """

    raw = read_json(path)
    return _validate(PromptsFile, raw, path)

def save_output(path: Path, items: List[OutputItem]) -> None:
    """ Save the items in output_file.json."""

    path.parent.mkdir(parents=True, exist_ok=True)
    payload = OutputFile(root=items).model_dump(mode="json")

    try:
        with path.open(mode="w", encoding="utf-8") as f:
            f.write(json.dumps(payload, indent=4, sort_keys=True))
    except PermissionError as exc:
        raise DataFileError(f"Permission denied {path}") from exc

