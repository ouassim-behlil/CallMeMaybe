from pathlib import Path
from typing import cast

from src import read_json, DataFileError


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

