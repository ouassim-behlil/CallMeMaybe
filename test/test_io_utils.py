import json
import pytest
from pathlib import Path
from pydantic import ValidationError

from src.schema import (
    FunctionDefinition,
    ParameterSpec,
    ReturnSpec,
    FunctionsDefinitionFile,
    PromptItem,
    PromptsFile,
    OutputItem,
    OutputFile,
)
from src.io_utils import (
    load_function_definitions,
    load_prompts,
    save_output,
    DataFileError,
    _read_json,
    _validate,
)


# -----------------------------
# Helpers
# -----------------------------

def write_json(tmp_path: Path, filename: str, data: object) -> Path:
    path = tmp_path / filename
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


# -----------------------------
# _read_json tests
# -----------------------------

def test_read_json_valid(tmp_path):
    path = write_json(tmp_path, "valid.json", {"a": 1})
    assert _read_json(path) == {"a": 1}


def test_read_json_file_not_found(tmp_path):
    with pytest.raises(DataFileError):
        _read_json(tmp_path / "missing.json")


def test_read_json_invalid_json(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text("{ invalid json", encoding="utf-8")

    with pytest.raises(DataFileError) as exc:
        _read_json(path)

    assert "Invalid JSON" in str(exc.value)


# -----------------------------
# _validate tests
# -----------------------------

def test_validate_success():
    raw = [{"name": "f", "description": "d",
            "parameters": {"x": {"type": "number"}},
            "returns": {"type": "string"}}]

    result = _validate(FunctionsDefinitionFile, raw, Path("dummy"))
    assert isinstance(result, FunctionsDefinitionFile)
    assert result.root[0].name == "f"


def test_validate_failure():
    raw = [{"name": "f", "description": "d",
            "parameters": {"x": {"type": "INVALID"}},
            "returns": {"type": "string"}}]

    with pytest.raises(DataFileError) as exc:
        _validate(FunctionsDefinitionFile, raw, Path("dummy"))

    assert "Scheme error" in str(exc.value)


# -----------------------------
# load_function_definitions tests
# -----------------------------

def test_load_function_definitions(tmp_path):
    raw = [{
        "name": "add",
        "description": "Adds numbers",
        "parameters": {"a": {"type": "number"}, "b": {"type": "number"}},
        "returns": {"type": "number"}
    }]
    path = write_json(tmp_path, "functions.json", raw)

    result = load_function_definitions(path)
    assert isinstance(result, FunctionsDefinitionFile)
    assert result.root[0].name == "add"


# -----------------------------
# load_prompts tests
# -----------------------------

def test_load_prompts(tmp_path):
    raw = [{"prompt": "Say hello"}]
    path = write_json(tmp_path, "prompts.json", raw)

    result = load_prompts(path)
    assert isinstance(result, PromptsFile)
    assert result.root[0].prompt == "Say hello"


# -----------------------------
# save_output tests
# -----------------------------

def test_save_output(tmp_path):
    items = [
        OutputItem(prompt="p", name="f", parameters={"x": 1})
    ]

    path = tmp_path / "out" / "output.json"
    save_output(path, items)

    # File must exist
    assert path.exists()

    # Validate JSON structure
    data = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(data, list)
    assert data[0]["prompt"] == "p"
    assert data[0]["parameters"]["x"] == 1


def test_save_output_permission_error(tmp_path, monkeypatch):
    items = [OutputItem(prompt="p", name="f", parameters={"x": 1})]
    path = tmp_path / "out" / "output.json"

    def fake_open(*args, **kwargs):
        raise PermissionError("nope")

    monkeypatch.setattr(Path, "open", fake_open)

    with pytest.raises(DataFileError):
        save_output(path, items)


# -----------------------------
# Model-specific tests
# -----------------------------

def test_function_definition_param_types():
    fd = FunctionDefinition(
        name="test",
        description="desc",
        parameters={"x": ParameterSpec(type="number")},
        returns=ReturnSpec(type="string")
    )
    assert fd.param_types() == {"x": "number"}


def test_prompts_file_validation():
    pf = PromptsFile.model_validate([{"prompt": "hello"}])
    assert pf.root[0].prompt == "hello"


def test_output_file_validation():
    of = OutputFile.model_validate([
        {"prompt": "p", "name": "n", "parameters": {"a": 1}}
    ])
    assert of.root[0].name == "n"
