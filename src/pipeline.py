"""ABOUTME: Orchestration layer — turns a natural-language prompt into a
validated OutputItem by running two constrained-decoding phases (pick the
function, then fill its parameters) against the loaded model."""

import json
import logging

from llm_sdk import Small_LLM_Model

from .decoder import GenerationError, encode_prompt, generate_constrained
from .grammar import (
    GrammarState,
    function_name_grammar,
    parameters_object_grammar,
)
from .schema import (
    FunctionDefinition,
    OutputItem,
    PromptItem,
    validate_output_against_schema,
)
from .tokenizer_utils import Vocabulary

logger = logging.getLogger(__name__)

NAME_PROMPT_TEMPLATE = (
    "<|im_start|>system\n"
    "You are a function-calling assistant. Choose the single most "
    "appropriate function from the list below. Answer with its exact "
    "name only — no punctuation.\n\n"
    "Available functions:\n"
    "{functions_block}<|im_end|>\n"
    "<|im_start|>user\n"
    "{prompt}<|im_end|>\n"
    "<|im_start|>assistant\n"
)

PARAMS_PROMPT_SUFFIX = (
    "<|im_end|>\n"  # Close the previous assistant message.
    "<|im_start|>user\n"
    "Now provide arguments for {chosen_name} as a JSON object. "
    "Extract values exactly from the user's request. Match each value "
    "to the correct parameter name. Return JSON only.<|im_end|>\n"
    "<|im_start|>assistant\n"
)


def format_functions_block(functions: list[FunctionDefinition]) -> str:
    """Render the function list as a compact, model-readable block.

    This text only *guides* the model towards the right choice — it has no
    bearing on output validity, since that's guaranteed by the grammar
    regardless of what the model "understands" from this description.
    """
    lines: list[str] = []
    for f in functions:
        params = ", ".join(f"{k}: {v.type}" for k, v in f.parameters.items())
        lines.append(f"- {f.name}({params}): {f.description}")
    return "\n".join(lines)


class PromptPipeline:
    """Holds everything that's expensive to build once and reused across
    every prompt in the run: the model, the vocabulary/trie, and the
    grammar objects for each function.

    Grammar states are immutable (see grammar.py) — `step()` returns new
    instances rather than mutating `self` — so it's safe to build each
    grammar exactly once here and hand the *same* starting instance to
    every prompt's generate_constrained call.
    """

    def __init__(
        self,
        model: Small_LLM_Model,
        vocab: Vocabulary,
        functions: list[FunctionDefinition],
        max_tokens: int = 64,
        stream: bool = False,
    ) -> None:
        if not functions:
            raise ValueError("At least one function definition is required")

        self._model = model
        self._vocab = vocab
        self._max_tokens = max_tokens
        self._stream = stream
        self._functions_by_name = {f.name: f for f in functions}
        self._functions_block = format_functions_block(functions)

        self._name_grammar: GrammarState = function_name_grammar(
            [f.name for f in functions]
        )
        self._param_grammars: dict[str, GrammarState] = {
            f.name: parameters_object_grammar(f.param_types())
            for f in functions
        }

    def process_prompt(self, prompt_item: PromptItem) -> OutputItem:
        """Run both generation phases for one prompt and return a validated
        OutputItem. Raises GenerationError on failure — the caller decides
        whether that's fatal or skippable.
        """
        name_prompt_text = NAME_PROMPT_TEMPLATE.format(
            functions_block=self._functions_block, prompt=prompt_item.prompt
        )
        name_input_ids = encode_prompt(self._model, name_prompt_text)

        name_result = generate_constrained(
            self._model,
            self._vocab,
            name_input_ids,
            self._name_grammar,
            max_tokens=self._max_tokens,
            stream=self._stream,
        )
        chosen_name = name_result.text
        if self._stream:
            print()

        param_grammar = self._param_grammars.get(chosen_name)
        if param_grammar is None:
            # Should be unreachable: the name grammar only ever completes
            # on one of the known function names. Treat it as a hard
            # error rather than silently guessing, since it means the
            # trie/grammar pairing is broken.
            raise GenerationError(
                f"Model selected unknown function name {chosen_name!r}"
            )

        selected_function = self._functions_by_name[chosen_name]
        schema = {
            name: parameter.type
            for name, parameter in selected_function.parameters.items()
        }
        params_prompt_text = (
            name_prompt_text
            + chosen_name
            + PARAMS_PROMPT_SUFFIX.format(
                chosen_name=chosen_name,
                schema=json.dumps(schema),
                prompt=prompt_item.prompt,
            )
        )
        params_input_ids = encode_prompt(self._model, params_prompt_text)

        params_result = generate_constrained(
            self._model,
            self._vocab,
            params_input_ids,
            param_grammar,
            max_tokens=self._max_tokens,
            stream=self._stream,
        )
        if self._stream:
            print()

        try:
            parameters = json.loads(params_result.text)
        except json.JSONDecodeError as exc:
            # Should be unreachable if the grammar is correct — the whole
            # point of constrained decoding is that this can't happen.
            # Surface it loudly rather than writing bad output.
            raise GenerationError(
                f"Grammar produced unparsable JSON: {params_result.text!r}"
            ) from exc

        # JSON parses whole-number values as ints. Normalize declared number
        # parameters to floats so the output consistently stores 4 as 4.0.
        parameters = {
            key: (
                float(value)
                if selected_function.parameters[key].type == "number"
                else value
            )
            for key, value in parameters.items()
        }

        item = OutputItem(
            prompt=prompt_item.prompt,
            name=chosen_name,
            parameters=parameters,
        )
        validate_output_against_schema(item, self._functions_by_name)
        return item


def run_pipeline(
    model: Small_LLM_Model,
    functions: list[FunctionDefinition],
    prompts: list[PromptItem],
    max_tokens: int = 64,
    stream: bool = False,
) -> list[OutputItem]:
    """Process every prompt, skipping (and logging) individual failures so
    one bad prompt never aborts the whole run.
    """
    vocab = Vocabulary.from_model(model)
    pipeline = PromptPipeline(
        model,
        vocab,
        functions,
        max_tokens=max_tokens,
        stream=stream,
    )

    results: list[OutputItem] = []
    failures = 0
    for i, prompt_item in enumerate(prompts):
        try:
            results.append(pipeline.process_prompt(prompt_item))
        except GenerationError as exc:
            failures += 1
            logger.warning(
                "Prompt %d (%r) failed: %s",
                i,
                prompt_item.prompt,
                exc,
            )

    logger.info(
        "Processed %d prompts: %d succeeded, %d failed",
        len(prompts),
        len(results),
        failures,
    )
    return results
