
from src.tokenizer_utils import (
    build_id_to_token,
    decode_token,
    TrieNode,
    VocabTrie
)


def test_build_id_to_token():
    vocab = {
        "hello": 0,
        "world": 1,
        "!": 2,
    }
    result = build_id_to_token(vocab)
    assert result == {
        0: "hello",
        1: "world",
        2: "!",
    }
def test_build_id_to_token_empty_vocab():
    vocab: dict[str, int] = {}

    result = build_id_to_token(vocab)

    assert result == {}


def test_build_id_to_token_single_entry():
    vocab = {"hello": 42}

    result = build_id_to_token(vocab)

    assert result == {42: "hello"}


def test_build_id_to_token_preserves_raw_token():
    vocab = {
        "hello world": 0,
        "don't": 1,
        "😊": 2,
        "": 3,
    }

    result = build_id_to_token(vocab)

    assert result == {
        0: "hello world",
        1: "don't",
        2: "😊",
        3: "",
    }


def test_build_id_to_token_duplicate_ids_last_token_wins():
    vocab = {
        "first": 0,
        "second": 0,
    }

    result = build_id_to_token(vocab)

    assert result == {0: "second"}

def test_decode_token_converts_space_marker():
    assert decode_token("Ġhello") == " hello"


def test_decode_token_without_marker():
    assert decode_token("hello") == "hello"


def test_decode_token_empty_string():
    assert decode_token("") == ""


def test_decode_token_preserves_unicode():
    assert decode_token("cafÃ©") == "café"


def test_decode_token_preserves_multiple_special_characters():
    assert decode_token("ĠhelloĠworld") == " hello world"


def test_trie_node_starts_empty():
    node = TrieNode()

    assert node.children == {}
    assert node.token_ids == []


def test_tokens_matching_returns_all_tokens():
    trie = VocabTrie.build({
        0: "cat",
        1: "car",
        2: "dog",
    })

    def transition(state: str, ch: str) -> str:
        return state + ch

    result = trie.tokens_matching(
        start_state="",
        transition=transition,
    )

    assert result == [0, 1, 2]


def test_empty_vocab_returns_no_matches():
    trie = VocabTrie.build({})

    result = trie.tokens_matching(
        start_state="",
        transition=lambda state, ch: state + ch,
    )

    assert result == []


def test_matches_token_that_is_prefix_of_another_token():
    trie = VocabTrie.build({
        0: "a",
        1: "ab",
        2: "abc",
    })

    result = trie.tokens_matching(
        start_state="",
        transition=lambda state, ch: state + ch,
    )

    assert result == [0, 1, 2]

def test_matches_tokens_with_shared_prefix():
    trie = VocabTrie.build({
        0: "cat",
        1: "car",
        2: "can",
        3: "dog",
    })

    result = trie.tokens_matching(
        start_state="",
        transition=lambda state, ch: state + ch,
    )

    assert result == [0, 1, 2, 3]

def test_multiple_token_ids_for_same_text_are_returned():
    trie = VocabTrie.build({
        10: "hello",
        20: "hello",
    })

    result = trie.tokens_matching(
        start_state="",
        transition=lambda state, ch: state + ch,
    )

    assert result == [10, 20]

def test_transition_can_reject_a_path():
    trie = VocabTrie.build({
        0: "cat",
        1: "dog",
    })

    def transition(state: str, ch: str):
        if ch == "c":
            return None

        return state + ch

    result = trie.tokens_matching(
        start_state="",
        transition=transition,
    )

    assert result == [1]

def test_is_acceptable_end_filters_tokens():
    trie = VocabTrie.build({
        0: "cat",
        1: "car",
        2: "dog",
    })

    def transition(state: str, ch: str):
        return state + ch

    def is_acceptable(state: str) -> bool:
        return state.endswith("t")

    result = trie.tokens_matching(
        start_state="",
        transition=transition,
        is_acceptable_end=is_acceptable,
    )

    assert result == [0]


def test_transition_can_use_arbitrary_state():
    trie = VocabTrie.build({
        0: "cat",
        1: "dog",
    })

    def transition(state: tuple[str, ...], ch: str) -> tuple[str, ...]:
        return state + (ch,)

    result = trie.tokens_matching(
        start_state=(),
        transition=transition,
        is_acceptable_end=lambda state: len(state) == 3,
    )

    assert result == [0, 1]

def test_empty_tokens_are_not_matched():
    trie = VocabTrie.build({
        0: "",
        1: "hello",
    })

    result = trie.tokens_matching(
        start_state="",
        transition=lambda state, ch: state + ch,
    )

    assert result == [1]


def test_matches_are_returned_in_vocab_order():
    trie = VocabTrie.build({
        5: "cat",
        2: "dog",
        9: "bird",
    })

    result = trie.tokens_matching(
        start_state="",
        transition=lambda state, ch: state + ch,
    )

    assert result == [5, 2, 9]
