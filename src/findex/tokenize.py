import re
import unicodedata
from collections.abc import Iterator

TOKEN_PATTERN: re.Pattern[str] = re.compile(r"[^\W_]+(?:['\u2019-][^\W_]+)*", re.UNICODE)


def tokenize(text: str) -> Iterator[str]:
    """Yield normalized word-like tokens without materializing the input."""
    normalized = unicodedata.normalize("NFC", text).casefold()
    yield from (match.group(0) for match in TOKEN_PATTERN.finditer(normalized))
