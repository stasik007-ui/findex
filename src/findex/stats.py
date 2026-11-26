from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from findex.corpus import iter_documents
from findex.index import Index
from findex.tokenize import tokenize


@dataclass(frozen=True, slots=True)
class IndexStats:
    num_docs: int
    unique_terms: int
    total_tokens: int
    avg_doc_length: float
    hapax_count: int
    hapax_ratio: float
    top_terms: tuple[tuple[str, int, int], ...]  # (term, df, collection_tf)


def compute_index_stats(index: Index, *, top_n: int = 10) -> IndexStats:
    """Compute corpus and vocabulary statistics directly from a built Index."""
    total_tokens = sum(index.doc_lengths.values())
    collection_tf: dict[str, int] = {}
    doc_freq: dict[str, int] = {}

    for term, plist in index.items():
        doc_freq[term] = len(plist)
        collection_tf[term] = sum(p.tf for p in plist)

    hapax_count = sum(1 for tf in collection_tf.values() if tf == 1)
    unique_terms = len(index)
    hapax_ratio = (hapax_count / unique_terms) if unique_terms > 0 else 0.0

    ranked_terms = sorted(
        collection_tf.keys(),
        key=lambda t: (-doc_freq[t], -collection_tf[t], t),
    )[:top_n]
    top_terms = tuple((t, doc_freq[t], collection_tf[t]) for t in ranked_terms)

    return IndexStats(
        num_docs=index.num_docs,
        unique_terms=unique_terms,
        total_tokens=total_tokens,
        avg_doc_length=index.avg_doc_length,
        hapax_count=hapax_count,
        hapax_ratio=hapax_ratio,
        top_terms=top_terms,
    )


def compute_corpus_stats(root: Path, *, top_n: int = 10) -> tuple[int, int, list[tuple[str, int]]]:
    """Stream a raw corpus directory and compute (doc_count, total_tokens, top_terms)."""
    doc_count = 0
    counts: Counter[str] = Counter()
    for doc in iter_documents(root):
        doc_count += 1
        counts.update(tokenize(doc.text))
    return doc_count, sum(counts.values()), counts.most_common(top_n)