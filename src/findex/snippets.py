import re
import unicodedata
from collections.abc import Callable
from pathlib import Path

from findex.index import DocMeta
from findex.tokenize import TOKEN_PATTERN


def extract_snippet(
    meta: DocMeta,
    query_terms: tuple[str, ...],
    *,
    window: int = 80,
    left_delim: str = "[",
    right_delim: str = "]",
    escape_fn: Callable[[str], str] | None = None,
) -> str:
    """Extract a +-window character snippet around the first matched term and highlight terms."""
    path = Path(meta.path)
    if not path.exists():
        return ""

    if path.suffix.lower() == ".jsonl":
        return _highlight_terms(
            meta.title,
            query_terms,
            left_delim=left_delim,
            right_delim=right_delim,
            escape_fn=escape_fn,
        )

    text = path.read_text(encoding="utf-8", errors="replace")
    clean_terms = tuple(dict.fromkeys(t for t in query_terms if t))
    if not clean_terms:
        compact = " ".join(text.split())
        snippet = compact[: window * 2] + ("..." if len(compact) > window * 2 else "")
        return escape_fn(snippet) if escape_fn is not None else snippet

    normalized = unicodedata.normalize("NFC", text)
    center = _find_best_offset(normalized, clean_terms)

    start = max(0, center - window)
    end = min(len(normalized), center + window)
    fragment = " ".join(normalized[start:end].split())

    prefix = "..." if start > 0 else ""
    suffix = "..." if end < len(normalized) else ""
    highlighted = _highlight_terms(
        fragment,
        clean_terms,
        left_delim=left_delim,
        right_delim=right_delim,
        escape_fn=escape_fn,
    )
    return f"{prefix}{highlighted}{suffix}"


def _find_best_offset(text: str, terms: tuple[str, ...]) -> int:
    term_set = set(terms)
    for match in TOKEN_PATTERN.finditer(text):
        if match.group(0).casefold() in term_set:
            return match.start()

    lowered = text.casefold()
    positions = [pos for t in terms if (pos := lowered.find(t)) != -1]
    return min(positions) if positions else 0


def _highlight_terms(
    fragment: str,
    terms: tuple[str, ...],
    *,
    left_delim: str = "[",
    right_delim: str = "]",
    escape_fn: Callable[[str], str] | None = None,
) -> str:
    escape: Callable[[str], str] = escape_fn if escape_fn is not None else lambda s: s
    if not terms:
        return escape(fragment)

    pattern = re.compile(
        r"(?<![^\W_])(" + "|".join(re.escape(t) for t in terms) + r")(?![^\W_])",
        re.IGNORECASE | re.UNICODE,
    )
    parts: list[str] = []
    last_end = 0
    for match in pattern.finditer(fragment):
        parts.append(escape(fragment[last_end : match.start()]))
        parts.append(f"{left_delim}{escape(match.group(0))}{right_delim}")
        last_end = match.end()
    parts.append(escape(fragment[last_end:]))
    return "".join(parts)
