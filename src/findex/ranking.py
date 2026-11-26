from __future__ import annotations

import math
import re
import unicodedata
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from findex.index import Index, Posting

_WORD_SPAN_RE = re.compile(r"\w+(?:['’\-]\w+)*", flags=re.UNICODE)


@dataclass(frozen=True, slots=True, order=True)
class SearchResult:
    doc_id: int = field(compare=False)
    score: float
    title: str = field(compare=False)
    snippet: str = field(default="", compare=False)

    def __str__(self) -> str:
        base = f"[{self.doc_id:>5}] ({self.score:.4f}) {self.title}"
        if self.snippet:
            return f"{base}\n        {self.snippet}"
        return base


class ResultList(list[SearchResult]):
    """A result list whose slices remain result lists."""

    def __getitem__(self, item):
        value = super().__getitem__(item)
        return ResultList(value) if isinstance(item, slice) else value

    def __repr__(self) -> str:
        return f"ResultList({list.__repr__(self)})"


@runtime_checkable
class Scorer(Protocol):
    """Structural protocol for term-document relevance scorers."""

    def score(self, term: str, posting: Posting, index: Index) -> float: ...


@dataclass(frozen=True, slots=True)
class TfIdf:
    """Sublinear TF * smoothed IDF scorer."""

    def score(self, term: str, posting: Posting, index: Index) -> float:
        n_docs = index.num_docs
        df = index.df(term)
        if n_docs == 0 or df == 0 or posting.tf <= 0:
            return 0.0
        tf_weight = 1.0 + math.log(posting.tf)
        idf = math.log((n_docs + 1.0) / df)
        return tf_weight * idf

    def __call__(self, term: str, posting: Posting, index: Index) -> float:
        return self.score(term, posting, index)


@dataclass(frozen=True, slots=True)
class BM25:
    """Okapi BM25 scorer with term-frequency saturation and length normalization."""

    k1: float = 1.5
    b: float = 0.75

    def score(self, term: str, posting: Posting, index: Index) -> float:
        n_docs = index.num_docs
        df = index.df(term)
        if n_docs == 0 or df == 0 or posting.tf <= 0:
            return 0.0

        idf = math.log(1.0 + (n_docs - df + 0.5) / (df + 0.5))
        dl = index.doc_length(posting.doc_id)
        avgdl = index.avg_doc_length or 1.0
        tf = posting.tf
        denom = tf + self.k1 * (1.0 - self.b + self.b * (dl / avgdl))
        return idf * (tf * (self.k1 + 1.0)) / denom

    def __call__(self, term: str, posting: Posting, index: Index) -> float:
        return self.score(term, posting, index)


def make_snippet(
    text: str,
    terms: Sequence[str],
    positions: Sequence[int] = (),
    *,
    window: int = 80,
) -> str:
    """Extract a +-window character snippet around the best match and highlight terms."""
    if not text:
        return ""

    norm_text = unicodedata.normalize("NFC", text)
    center: int | None = None

    if positions:
        target_pos = positions[0]
        for idx, match in enumerate(_WORD_SPAN_RE.finditer(norm_text)):
            if idx == target_pos:
                center = match.start()
                break

    if center is None and terms:
        folded = norm_text.casefold()
        for term in terms:
            found = folded.find(term.casefold())
            if found != -1:
                center = found
                break

    if center is None:
        snippet = norm_text[: window * 2].strip()
        return re.sub(r"\s+", " ", snippet)

    start = max(0, center - window)
    end = min(len(norm_text), center + window)
    raw_slice = re.sub(r"\s+", " ", norm_text[start:end]).strip()

    if terms:
        escaped = sorted({re.escape(t) for t in terms if t}, key=len, reverse=True)
        if escaped:
            hl_re = re.compile(r"\b(" + "|".join(escaped) + r")\b", flags=re.IGNORECASE)
            raw_slice = hl_re.sub(r"**\1** [\1]", raw_slice)

    prefix = "..." if start > 0 else ""
    suffix = "..." if end < len(norm_text) else ""
    return f"{prefix}{raw_slice}{suffix}"


def precision_at_k(
    retrieved_doc_ids: Sequence[int],
    relevant_doc_ids: set[int],
    k: int = 5,
) -> float:
    """Compute Precision@k for a ranked list of document IDs."""
    if k <= 0:
        return 0.0
    top_k = retrieved_doc_ids[:k]
    if not top_k:
        return 0.0
    hits = sum(1 for doc_id in top_k if doc_id in relevant_doc_ids)
    return hits / k


def explain_score(scorer: Scorer, term: str, posting: Posting, index: Index) -> str:
    """Return a concise explanation of one term/document score."""
    value = scorer.score(term, posting, index)
    return f"{type(scorer).__name__}({term!r}, doc={posting.doc_id}) = {value:.6f}"