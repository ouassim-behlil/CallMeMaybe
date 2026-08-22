from pathlib import Path
from typing import cast, Callable, Optional, TypeVar
from dataclasses import dataclass

from src import read_json, DataFileError
from llm_sdk import Small_LLM_Model

StateT = TypeVar("StateT")


def _bytes_to_unicode() -> dict[int, str]:
    """ Reproduce the GPT-2 byte-to-unicode mapping used in Qwen"""

    bs = list(
        list(range(ord('!'), ord('~') + 1)) +
        list(range(ord("¡"), ord("¬") + 1)) +
        list(range(ord("®"), ord("ÿ") + 1))
    )

    cs = bs[:]
    n = 0

    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, (chr(c) for c in cs)))

_BYTE_ENCODER: dict[int, str] = _bytes_to_unicode()
_BYTE_DECODER: dict[str, int] = {c: u for u, c in _BYTE_ENCODER.items()}

def load_vocab(vocab_path: Path) -> dict[str, int]:
    """ Load the vocabulary from the given path"""
    raw = read_json(vocab_path)
    if not isinstance(raw, dict):
        raise DataFileError("Expected vocab to be a dictionary.")
    return cast(dict[str, int], raw)


def build_id_to_token(vocab: dict[str, int]) -> dict[int, str]:
    """ invert token->id into id->raw token string. """
    return {
        token_id: token
        for token, token_id in vocab.items()
    }

def decode_token(raw_token: str) -> str:
    """Convert raw token (e.g. 'Ġhello') into literal text (' hello'). """

    try:
        raw_bytes = bytes(_BYTE_DECODER[ch] for ch in raw_token)
    except KeyError as exc:
        raise ValueError(f"Unrecognized byte-encoding char in token {raw_token!r}") from exc
    return raw_bytes.decode(encoding="utf-8", errors="replace")


class TrieNode:
    __slots__ = ("children", "token_ids")

    def __init__(self) -> None:
        self.children: dict[str, "TrieNode"] = {}
        self.token_ids: list[int] = []

class VocabTrie:
    """ Prefix trie over decoded token text. """

    def __init__(self) -> None:
        self._root: "TrieNode" = TrieNode()

    def _insert(self, text: str, token_id: int) -> None:
        node = self._root
        for ch in text:
            node = node.children.setdefault(ch, TrieNode())
        node.token_ids.append(token_id)


    @classmethod
    def build(cls, id_to_token: dict[int, str]) -> "VocabTrie":

        trie = cls()

        for token_id, token in id_to_token.items():
            text = decode_token(token)
            if text:
                trie._insert(text, token_id)

        return trie


    def _walk(
            self,
            node: TrieNode,
            state: StateT,
            transition: Callable[[StateT, str], Optional[StateT]],
            is_acceptable_end: Callable[[StateT], bool],
            results: list[int],
    ) -> None:
        """ Backtrack the VocabTrie and adds acceptable token ids to results. """

        if node.token_ids and is_acceptable_end(state):
            results.extend(node.token_ids)

        for ch, child in node.children.items():
            next_state = transition(state, ch)
            if next_state is not None:
                self._walk(child, next_state, transition, is_acceptable_end, results)

    def tokens_matching(
            self,
            start_state: StateT,
            transition: Callable[[StateT, str], Optional[StateT]],
            is_acceptable_end: Callable[[StateT], bool] = lambda _s: True,
    ) -> list[int]:

        results: list[int] = []
        self._walk(self._root, start_state, transition, is_acceptable_end, results)
        return results

@dataclass(slots=True)
class Vocabulary:
    """Bundle of everything decoder.py needs, built once at startup."""

    id_to_token: dict[int, str]
    trie: "VocabTrie"

    @classmethod
    def from_model(cls, model:Small_LLM_Model) -> "Vocabulary":
        vocab_path = Path(model.get_path_to_vocab_file())
        vocab = load_vocab(vocab_path)
        id_to_token = build_id_to_token(vocab)
        trie = VocabTrie.build(id_to_token)
        return cls(id_to_token=id_to_token, trie=trie)
