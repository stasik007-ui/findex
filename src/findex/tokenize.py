from collections.abc import Iterator
import re
import unicodedata


_TOKEN_RE = re.compile(r"[^\W_]+(?:['\u2019-][^\W_]+)*", re.UNICODE)


def tokenize(text: str) -> Iterator[str]:
    """Yield normalized word-like tokens without materializing the input."""
    normalized = unicodedata.normalize("NFC", text).casefold()
    yield from (match.group(0) for match in _TOKEN_RE.finditer(normalized))