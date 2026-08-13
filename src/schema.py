from pydantic import BaseModel, RootModel
from typing import Literal, Dict, Any, List

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

    def param_types(self):
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
    pass