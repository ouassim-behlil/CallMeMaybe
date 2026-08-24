from abc import ABC, abstractmethod
from typing import Optional

class GrammarState(ABC):
     
    @abstractmethod
    def step(self, char: str) -> Optional["GrammarState"]:
        """ Consume One character. Return the next state or None if invalid. """


    @property
    @abstractmethod
    def is_complete(self) -> bool:
        
        """Whether stopping generation *right now* would yield valid JSON
        for this fragment (not necessarily the only valid stopping point —
        e.g. an integer is complete after each digit, since more digits
        could still follow)."""


class LiteralState(GrammarState):
    """Matches one fixed string exactly, e.g. punctuation or a key name."""

    def __init__(self, remaining: str) -> None:
        self.remaining:str = remaining

    def step(self, char: str) -> Optional["GrammarState"]:
        if self.remaining and self.remaining[0] == char:
            return LiteralState(self.remaining[1:])
        return None

    @property
    def is_complete(self) -> bool:
        return len(self.remaining) == 0


class LiteralChoiceState(GrammarState):
    """Matches exactly one of several fixed strings — e.g. function names,
    or JSON's `true`/`false`."""

    def __init__(self, remaining_candidates: frozenset[str]) -> None:
        self._remaining = remaining_candidates


    @classmethod
    def from_options(cls, options: list[str]) -> "LiteralChoiceState":

        for a in options:
            for b in options:
                if a != b and b.startswith(a):
                    raise ValueError(
                        f"Ambiguous literal choices: {a!r} is a prefix of {b!r}"
                    )
        return cls(frozenset(options))

    def step(self, char: str) -> GrammarState | None:
        nxt = frozenset(c[1:] for c in self._remaining if c and c[0] == char)
        return LiteralChoiceState(nxt) if nxt else None

    @property
    def is_complete(self) -> bool:
        return "" in self._remaining


class NumberState(GrammarState):
    """JSON number grammar: -?(0|[1-9][0-9]*)(\\.[0-9]+)?  (no exponents).

    Explicitly forbids leading zeros followed by more digits (`01` is not
    valid JSON — Python's json.loads rejects it), since the whole point of
    this engine is guaranteeing output that actually parses.
    """

    _START, _ZERO, _INT, _AFTER_DOT, _FRAC = (
        "start", "zero", "int", "after_dot", "frac",
    )

    def __init__(self, phase: str = _START, seen_minus: bool = False) -> None:
        self._phase = phase
        self._seen_minus = seen_minus


    def step(self, char: str) -> GrammarState | None:
        p = self._phase
        if p == self._START:
            if char == "-" and not self._seen_minus:
                return NumberState(self._START, True)
            if char == "0":
                return NumberState(self._ZERO)
            if char.isdigit():
                return NumberState(self._INT)
            return None
        if p == self._ZERO:
            if char == ".":
                return NumberState(self._AFTER_DOT)
            return None
        if p == self._INT:
            if char.isdigit():
                return NumberState(self._INT)
            if char == ".":
                return NumberState(self._AFTER_DOT)
            return None
        if p == self._AFTER_DOT:
            if char.isdigit():
                return NumberState(self._FRAC)
            return None
        if p == self._FRAC:
            if char.isdigit():
                return NumberState(self._FRAC)
            return None
        


    @property
    def is_complete(self) -> bool:
        return self._phase in (self._ZERO, self._FRAC, self._INT)
    

        