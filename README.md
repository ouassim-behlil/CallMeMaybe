*This project has been created as part of the 42 curriculum by obehlil.*

# CallMeMaybe

## Description

CallMeMaybe is a learning project that explores function calling with a small
local language model. Given a natural-language request, the program selects the
most appropriate function and generates its arguments as structured JSON.

The project uses a Hugging Face causal language model, a tokenizer vocabulary,
and a character-level grammar. The grammar constrains generation so that the
function name and generated parameters follow the expected syntax and schema.
The repository also contains a small Moulinette utility for generating exercise
data and grading generated function calls.

This project is intended for learning and experimentation rather than
production use.

## Instructions

### Requirements

- Python 3.10 or newer
- `uv` recommended for dependency management
- Internet access on the first run to download the Hugging Face model
- Enough memory for the selected model

### Installation

From the repository root:

```bash
uv sync
```

Alternatively, use the provided Makefile:

```bash
make install
```

To activate the generated virtual environment manually:

```bash
source .venv/bin/activate
```

### Running the project

Run the constrained function-calling pipeline with the default input files:

```bash
uv run python -m src
```

The equivalent Makefile command is:

```bash
make run
```

To stream constrained output while it is generated:

```bash
uv run python -m src --stream
```

The default files are:

```text
data/input/functions_definition.json
data/input/function_calling_tests.json
data/output/function_calling_results.json
```

Useful command-line options can be displayed with:

```bash
uv run python -m src --help
```

For example:

```bash
uv run python -m src \
  --functions_definition data/input/functions_definition.json \
  --input data/input/function_calling_tests.json \
  --output data/output/function_calling_results.json \
  --model_name Qwen/Qwen3-0.6B \
  --max_tokens 64 \
  --stream
```

### Running the tests

The Makefile provides:

```bash
make lint
make lint-strict
make debug
make clean
```

### Running the Moulinette

The Moulinette is a separate package located in the `moulinette/` directory.
From the repository root, install it in the active environment if necessary:

```bash
uv pip install -e ./moulinette
```

Generate exercise files:

```bash
uv run --active python -m moulinette \
  prepare_exercises \
  --output data \
  --set public
```

Grade generated answers:

```bash
uv run --active python -m moulinette \
  grade_student_answers \
  data/output/function_calling_results.json \
  --set public
```

Use `--set private` to grade against the private exercise set.

## Algorithm explanation

The constrained decoder works in several stages:

1. The input prompt and function definitions are formatted into a model
   context.
2. The model generates a function name. A `LiteralChoiceState` grammar only
   allows characters that can lead to one of the known function names.
3. The selected function's parameter grammar is built from its schema.
   Punctuation, JSON keys, and separators are represented by literal grammar
   states.
4. Values are generated with type-specific states:
   - `NumberState` accepts JSON numbers and rejects invalid leading zeros or
     incomplete fractions.
   - `StringContentState` accepts JSON string content and handles escapes.
   - `LiteralChoiceState` handles fixed alternatives such as `true` and
     `false`.
5. The tokenizer vocabulary is decoded into text and indexed in a prefix trie.
   At each generation step, the trie walks every token that can be fully
   consumed by the current grammar state.
6. The decoder asks the model for next-token logits, filters them to the
   grammar-allowed token IDs, and chooses the highest-logit valid token.
7. Generation stops when the grammar is complete or raises a controlled error
   when no valid token exists or the token limit is reached.
8. The generated JSON is parsed and validated against the function schema
   before it is written to the output file.

This approach constrains syntax and declared types. It does not prove that a
model-selected string or regular expression is semantically correct, so the
model output still needs evaluation for task-specific meaning.

## Design decisions

- **Character-level grammar:** Character transitions make it possible to
  constrain multi-character tokenizer tokens without depending on a specific
  tokenizer implementation.
- **Prefix trie:** The trie avoids scanning the complete vocabulary from
  scratch for every grammar transition and supports tokens that share prefixes.
- **Deterministic argmax:** The highest-logit allowed token is selected instead
  of sampling, making experiments reproducible and avoiding invalid-token
  randomness.
- **Immutable grammar states:** Each transition returns a new state. This makes
  grammar instances safe to reuse for multiple prompts.
- **Two generation phases:** Selecting the function separately from generating
  its parameters keeps the grammar focused and makes the process easier to
  inspect.
- **Schema validation after decoding:** Valid JSON alone is not enough, so the
  generated parameter names and Python value types are checked before output.
- **Streaming is optional:** `--stream` exposes progress during inference while
  preserving normal non-streaming execution.
- **Learning-oriented scope:** The implementation favors readable components
  and explicit state transitions over production-level optimization.

## Performance analysis

Model inference is the dominant cost. Each generated token requires a model
forward pass, so generation speed depends primarily on the selected model,
device, and available memory. CPU inference is simpler but slower; GPU or MPS
hardware can reduce latency when supported by the local PyTorch installation.

The vocabulary trie is built once when the pipeline starts and reused across
prompts. Grammar objects are also prepared once per function. This avoids
rebuilding the same static structures for every request.

The decoder is reliable for syntactic constraints: invalid function names,
malformed JSON punctuation, unsupported value types, and incomplete grammar
paths are rejected during generation. Semantic accuracy remains dependent on
the model and prompt quality. A small model may select the correct function
but extract an incorrect regular expression or replacement string, so 100%
semantic accuracy cannot be guaranteed by grammar constraints alone.

## Challenges faced

- **Mapping tokens to grammar characters:** Tokenizers emit variable-length
  tokens, so the trie must verify an entire token against the grammar before
  allowing it.
- **Grammar hand-off:** Variable-length values such as numbers and strings must
  be able to finish and transition to the next JSON delimiter. This is handled
  by the epsilon-style hand-off in `ConcatState`.
- **JSON number rules:** JSON numbers allow integer and fractional forms but
  reject invalid leading-zero forms and incomplete decimal values.
- **Model quality versus structural validity:** Constrained decoding fixes
  structure but cannot infer every user intention correctly. More explicit
  prompts and post-generation validation are needed for semantic reliability.
- **Local model setup:** The model and tokenizer may need to be downloaded from
  Hugging Face before the first run, and unauthenticated Hub access can be
  rate-limited.
- **Debugging generation:** Optional character streaming and the `pdb`
  Makefile target make it easier to inspect progress and diagnose dead ends.

## Testing strategy

The test suite is organized around the main correctness boundaries:

- Grammar state tests verify literal matching, choices, number syntax, prefix
  handling, and immutable transitions.
- Tokenizer utility tests verify vocabulary inversion, token decoding, trie
  construction, shared prefixes, duplicate token IDs, and rejected paths.
- Decoder tests use a dummy model and vocabulary to validate deterministic
  constrained generation without downloading a real model.
- I/O tests verify JSON loading, Pydantic validation, output creation, and
  error handling.
- Manual Moulinette runs compare generated calls with expected exercise
  corrections and provide an end-to-end check of function selection and
  parameter execution.

Run the automated tests with:

```bash
uv run pytest
```

## Example usage

Run the complete pipeline with streamed output:

```bash
uv run python -m src --stream
```

Use a custom output path:

```bash
uv run python -m src \
  --input data/input/function_calling_tests.json \
  --functions_definition data/input/functions_definition.json \
  --output data/output/my_results.json
```

For a request such as:

```text
What is the sum of 2 and 3?
```

the output has the following shape:

```json
{
  "prompt": "What is the sum of 2 and 3?",
  "name": "fn_add_numbers",
  "parameters": {
    "a": 2.0,
    "b": 3.0
  }
}
```

## Resources

- [Hugging Face Transformers documentation](https://huggingface.co/docs/transformers/)
- [Hugging Face Hub documentation](https://huggingface.co/docs/huggingface_hub/)
- [PyTorch documentation](https://pytorch.org/docs/)
- [Python `json` documentation](https://docs.python.org/3/library/json.html)
- [Python `argparse` documentation](https://docs.python.org/3/library/argparse.html)
- [Pydantic documentation](https://docs.pydantic.dev/)
- [Trie data structure overview](https://en.wikipedia.org/wiki/Trie)
- [JSON specification](https://www.json.org/)

AI assistance was used as a learning and development aid. It helped explain
constrained decoding, inspect and debug Python code, identify grammar and
pipeline bugs, improve prompts, add optional streaming output, create the
Makefile, and draft this documentation. The implementation, decisions, and
validation remain the responsibility of the project author.
