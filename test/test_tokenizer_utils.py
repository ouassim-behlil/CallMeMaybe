import pytest
from pathlib import Path

from src import DataFileError
from src.tokenizer_utils import (
    _bytes_to_unicode,
    _BYTE_ENCODER,
    _BYTE_DECODER,
    load_vocab,
)

def test_bytes_to_unicode_length():
    mapping = _bytes_to_unicode()
    assert len(mapping) == 256

def test_byte_encoder_decoder_roundtrip():
    for byte in range(256):
        encoded = _BYTE_ENCODER[byte]
        decoded = _BYTE_DECODER[encoded]
        assert decoded == byte

def test_load_vocab_valid(monkeypatch):
    fake_vocab = {"hello": 1, "world": 2}

    def mock_read_json(path):
        return fake_vocab

    monkeypatch.setattr("src.tokenizer_utils.read_json", mock_read_json)

    result = load_vocab(Path("dummy.json"))
    assert result == fake_vocab

def test_load_vocab_invalid(monkeypatch):
    def mock_read_json(path):
        return ["not", "a", "dict"]

    monkeypatch.setattr("src.tokenizer_utils.read_json", mock_read_json)

    with pytest.raises(DataFileError):
        load_vocab(Path("dummy.json"))
