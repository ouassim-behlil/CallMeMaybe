from pydantic import BaseModel, RootModel
from typing import Any, Dict, List, Literal

PARAM_TYPE = Literal["number", "string", "boolean"]


class ParameterSpec(BaseModel):
    type: PARAM_TYPE


class ReturnSpec(BaseModel):
    type: PARAM_TYPE


class FunctionDefinition(BaseModel):
    name: str
    description: str
    parameters: Dict[str, ParameterSpec]
    returns: ReturnSpec

    def param_types(self) -> dict[str, PARAM_TYPE]:
        return {k: v.type for k, v in self.parameters.items()}


class PromptItem(BaseModel):
    prompt: str


class OutputItem(BaseModel):
    prompt: str
    name: str
    parameters: Dict[str, Any]


class FunctionsDefinitionFile(RootModel[list[FunctionDefinition]]):
    """Validates the top-level JSON array in functions_definition.json."""


class PromptsFile(RootModel[List[PromptItem]]):
    """Validates the top-level JSON array in prompts_file.json."""


class OutputFile(RootModel[List[OutputItem]]):
    """Validates the top-level JSON array in output_file.json."""


def validate_output_against_schema(
    item: OutputItem, functions_by_name: dict[str, FunctionDefinition]
) -> None:
    """Raise ValueError if parameters do not match the function schema."""
    func = functions_by_name.get(item.name)
    if func is None:
        raise ValueError(f"Unknown function name: {item.name!r}")

    expected = func.param_types()
    if set(item.parameters) != set(expected):
        raise ValueError(
            f"Parameter mismatch for {item.name!r}: "
            f"expected {sorted(expected)}, got {sorted(item.parameters)}"
        )

    for key, expected_type in expected.items():
        value = item.parameters[key]
        valid = (
            isinstance(value, (int, float)) and not isinstance(value, bool)
            if expected_type == "number"
            else isinstance(value, str)
            if expected_type == "string"
            else isinstance(value, bool)
        )
        if not valid:
            raise ValueError(
                f"{key!r} expected {expected_type}, got {type(value)}"
            )
