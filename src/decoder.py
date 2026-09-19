"""ABOUTME: Constrained generation loop — ties the SDK's raw logits together
with a GrammarState and the vocab trie to produce guaranteed-valid text,
one token at a time."""

import sys
from dataclasses import dataclass
from typing import cast

import numpy as np
from llm_sdk import Small_LLM_Model

from .grammar import GrammarState
from .tokenizer_utils import Vocabulary, decode_token


class GenerationError(Exception):
    """Raised when constrained generation can't proceed or complete.

    Two distinct failure modes are folded into this one exception,
    distinguished by message: a dead end (the grammar allows no valid next
    token — almost certainly a bug in how the grammar was built, not a
    model problem) and a runaway generation (max_tokens exceeded without
    the grammar ever reaching completion — a safety net so a bug can never
    hang the whole run).
    """


@dataclass
class GenerationResult:
    text: str
    """The literal text produced (e.g. 'fn_add_numbers' or '{"a": 2.0}')."""

    token_ids: list[int]
    """Just the newly generated token ids, in order (prompt ids excluded)."""


def _select_next_token(
    logits: list[float],
    candidate_ids: list[int],
) -> int:
    """Pick the highest-logit token among the grammar-allowed candidates.

    Deterministic argmax: this task has one correct structural answer per
    field (the schema), so there's no benefit to sampling/temperature —
    only risk of picking an unnecessarily-worse token.
    """
    if not candidate_ids:
        raise GenerationError(
            "Dead end: no vocabulary token is a valid continuation of the "
            "current grammar state. This indicates a grammar/tokenizer "
            "bug, not a model quality issue."
        )
    logits_arr = np.asarray(logits, dtype=np.float64)
    candidates_arr = np.asarray(candidate_ids, dtype=np.int64)
    best_local_idx = int(np.argmax(logits_arr[candidates_arr]))
    return int(candidates_arr[best_local_idx])


def generate_constrained(
    model: Small_LLM_Model,
    vocab: Vocabulary,
    input_ids: list[int],
    grammar_state: GrammarState,
    max_tokens: int = 64,
    stream: bool = False,
) -> GenerationResult:
    """Generate tokens under `grammar_state` until it completes.

    `input_ids` is the full context (prompt + anything generated in
    earlier phases) — the caller owns and extends this across phases so
    the model stays aware of what it already committed to (e.g. which
    function it picked, when generating parameters next).

    Stopping condition: as soon as `grammar_state.is_complete` is True.
    This works even for variable-length fields (numbers, strings) without
    any special-cased "should I stop now?" logic, because the *trie walk
    itself* offers both "extend this field" and "move on to what follows
    it" as valid next tokens whenever both are grammatically legal (see
    ConcatState's epsilon-closure in grammar.py) — the model's own logits
    decide which one it prefers, exactly like real constrained decoding
    should.
    """
    current_ids = list(input_ids)
    generated_ids: list[int] = []
    state = grammar_state
    text_parts: list[str] = []

    if state.is_complete:
        # Degenerate but legal case: e.g. a function with zero parameters,
        # whose grammar is just the literal "{}" and might already be
        # complete before any token is generated... in practice this only
        # happens if the caller passes an already-finished state, but
        # guard it anyway rather than looping needlessly.
        return GenerationResult(text="", token_ids=[])

    for _ in range(max_tokens):
        logits = model.get_logits_from_input_ids(current_ids)

        candidate_ids = vocab.trie.tokens_matching(
            start_state=state,
            transition=lambda s, c: s.step(c),
        )
        chosen_id = _select_next_token(logits, candidate_ids)

        chosen_text = decode_token(vocab.id_to_token[chosen_id])
        for ch in chosen_text:
            next_state = state.step(ch)
            if next_state is None:
                # Should be unreachable: tokens_matching only returns
                # tokens whose full text is walkable from `state`. If this
                # fires, the trie and the grammar have gone out of sync.
                raise GenerationError(
                    f"Internal inconsistency: token {chosen_id!r} "
                    f"({chosen_text!r}) was offered as valid but character "
                    f"{ch!r} was rejected by the grammar mid-token."
                )
            state = next_state

        current_ids.append(chosen_id)
        generated_ids.append(chosen_id)
        text_parts.append(chosen_text)
        if stream:
            for ch in chosen_text:
                sys.stdout.write(ch)
                sys.stdout.flush()

        if state.is_complete:
            return GenerationResult(
                text="".join(text_parts),
                token_ids=generated_ids,
            )

    raise GenerationError(
        f"Generation did not complete within max_tokens={max_tokens}. "
        f"Partial output: {''.join(text_parts)!r}"
    )


def encode_prompt(model: Small_LLM_Model, text: str) -> list[int]:
    """Turn a prompt string into a flat list of input ids.

    The SDK's `encode` returns a 2D tensor (batch size 1); this is the one
    place we touch it, so callers everywhere else only ever deal in plain
    `list[int]`.
    """
    tensor = model.encode(text)
    return cast(list[int], tensor[0].tolist())


def step_grammar_string(
    grammar_state: GrammarState,
    text: str,
) -> GrammarState | None:
    """Advance a grammar state through a literal character string.

    Returns the updated grammar state if every character is accepted.
    Returns None if the string cannot be consumed by the current grammar.
    """
    state = grammar_state
    for ch in text:
        next_state = state.step(ch)
        if next_state is None:
            return None
        state = next_state
    return state


class ConstrainedDecoder:
    """Small compatibility wrapper expected by the project tests.

    The real pipeline code uses `generate_constrained`, but this class keeps
    the older API working for simple deterministic generation tasks.
    """

    def __init__(self, model: Small_LLM_Model, vocab: Vocabulary) -> None:
        self.model = model
        self.vocab = vocab

    def generate(
        self,
        prompt: str,
        grammar: GrammarState,
        max_new_tokens: int = 64,
        stream: bool = False,
    ) -> str:
        current_ids = encode_prompt(self.model, prompt)
        state = grammar
        text_parts: list[str] = []

        for _ in range(max_new_tokens):
            logits = self.model.get_logits_from_input_ids(current_ids)
            candidate_ids = self.vocab.trie.tokens_matching(
                start_state=state,
                transition=lambda s, ch: s.step(ch),
            )
            if not candidate_ids:
                raise GenerationError(
                    "Dead end: no vocabulary token is a valid continuation "
                    "of the current grammar state."
                )

            chosen_id = _select_next_token(logits, candidate_ids)
            chosen_text = decode_token(self.vocab.id_to_token[chosen_id])

            for ch in chosen_text:
                next_state = state.step(ch)
                if next_state is None:
                    raise GenerationError(
                        f"Internal inconsistency: token {chosen_id!r} "
                        f"({chosen_text!r}) was offered as valid but "
                        f"character {ch!r} was rejected mid-token."
                    )
                state = next_state

            current_ids.append(chosen_id)
            text_parts.append(chosen_text)
            if stream:
                for ch in chosen_text:
                    sys.stdout.write(ch)
                    sys.stdout.flush()

            if state.is_complete:
                return "".join(text_parts)

        raise GenerationError(
            f"Generation did not complete within max_new_tokens="
            f"{max_new_tokens}. Partial output: {''.join(text_parts)!r}"
        )
