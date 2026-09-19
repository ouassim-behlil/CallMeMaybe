from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import Optional


class GrammarState(ABC):
    @abstractmethod
    def step(self, char: str) -> Optional["GrammarState"]:
        """Consume one character and return the next state."""

    @property
    @abstractmethod
    def is_complete(self) -> bool:
        """Whether stopping now produces a valid value."""


class LiteralState(GrammarState):
    """Match one fixed string exactly."""

    def __init__(self, remaining: str) -> None:
        self.remaining: str = remaining

    def step(self, char: str) -> Optional["GrammarState"]:
        if self.remaining and self.remaining[0] == char:
            return LiteralState(self.remaining[1:])
        return None

    @property
    def is_complete(self) -> bool:
        return not self.remaining


class LiteralChoiceState(GrammarState):
    """Match exactly one of several fixed strings."""

    def __init__(self, remaining_candidates: frozenset[str]) -> None:
        self._remaining = remaining_candidates

    @classmethod
    def from_options(cls, options: list[str]) -> "LiteralChoiceState":
        normalized = [option for option in options if option is not None]
        for first in normalized:
            for second in normalized:
                if first != second and second.startswith(first):
                    raise ValueError(
                        f"Ambiguous literal choices: {first!r} is a prefix "
                        f"of {second!r}"
                    )
        return cls(frozenset(normalized))

    def step(self, char: str) -> Optional["GrammarState"]:
        next_candidates = frozenset(
            candidate[1:]
            for candidate in self._remaining
            if candidate and candidate[0] == char
        )
        return LiteralChoiceState(next_candidates) if next_candidates else None

    @property
    def is_complete(self) -> bool:
        return "" in self._remaining


class NumberState(GrammarState):
    """Match JSON numbers without exponents."""

    _START, _ZERO, _INT, _AFTER_DOT, _FRAC = (
        "start",
        "zero",
        "int",
        "after_dot",
        "frac",
    )

    def __init__(self, phase: str = _START, seen_minus: bool = False) -> None:
        self._phase = phase
        self._seen_minus = seen_minus

    def step(self, char: str) -> Optional["GrammarState"]:
        phase = self._phase
        if phase == self._START:
            if char == "-" and not self._seen_minus:
                return NumberState(self._START, True)
            if char == "0":
                return NumberState(self._ZERO)
            if char.isdigit():
                return NumberState(self._INT)
            return None
        if phase == self._ZERO:
            return NumberState(self._AFTER_DOT) if char == "." else None
        if phase == self._INT:
            if char.isdigit():
                return NumberState(self._INT)
            if char == ".":
                return NumberState(self._AFTER_DOT)
            return None
        if phase == self._AFTER_DOT:
            return NumberState(self._FRAC) if char.isdigit() else None
        if phase == self._FRAC:
            return NumberState(self._FRAC) if char.isdigit() else None
        return None

    @property
    def is_complete(self) -> bool:
        return self._phase in (self._ZERO, self._INT, self._FRAC)


class StringContentState(GrammarState):
    """Match the characters between JSON string quotes."""

    def __init__(self, in_escape: bool = False) -> None:
        self._in_escape = in_escape

    def step(self, char: str) -> Optional["GrammarState"]:
        if self._in_escape:
            if char in '\\"/bfnrt':
                return StringContentState()
            return None
        if char == '"':
            return None
        if char == "\\":
            return StringContentState(in_escape=True)
        if ord(char) < 0x20:
            return None
        return StringContentState()

    @property
    def is_complete(self) -> bool:
        return not self._in_escape


class ConcatState(GrammarState):
    """Sequence several sub-grammars with epsilon-style hand-off."""

    def __init__(self, parts: list[GrammarState], index: int = 0) -> None:
        self._parts = parts
        self._index = index

    def step(self, char: str) -> Optional["ConcatState"]:
        index = self._index
        parts = self._parts
        while index < len(parts):
            next_state = parts[index].step(char)
            if next_state is not None:
                new_parts = list(parts)
                new_parts[index] = next_state
                return ConcatState(parts=new_parts, index=index)
            if parts[index].is_complete and index + 1 < len(parts):
                index += 1
                continue
            return None
        return None

    @property
    def is_complete(self) -> bool:
        index = self._index
        while index < len(self._parts):
            if not self._parts[index].is_complete:
                return False
            index += 1
        return True


def string_value_grammar() -> ConcatState:
    """Return a quoted JSON string grammar."""
    return ConcatState(
        [LiteralState('"'), StringContentState(), LiteralState('"')]
    )


def value_grammar_for_type(param_type: str) -> GrammarState:
    """Return the grammar for one parameter value."""
    if param_type == "number":
        return NumberState()
    if param_type == "string":
        return string_value_grammar()
    if param_type == "boolean":
        return LiteralChoiceState.from_options(["true", "false"])
    raise ValueError(f"Unsupported parameter type: {param_type!r}")


def function_name_grammar(names: list[str]) -> LiteralChoiceState:
    """Return a grammar for one of the known function names."""
    return LiteralChoiceState.from_options(names)


def parameters_object_grammar(param_types: Mapping[str, str]) -> ConcatState:
    """Return a JSON object grammar for the declared parameter types."""
    keys = list(param_types.keys())
    parts: list[GrammarState] = [LiteralState("{")]
    for index, key in enumerate(keys):
        parts.append(LiteralState(f'"{key}": '))
        parts.append(value_grammar_for_type(param_types[key]))
        if index < len(keys) - 1:
            parts.append(LiteralState(", "))
    parts.append(LiteralState("}"))
    return ConcatState(parts)
