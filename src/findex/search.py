from __future__ import annotations

import argparse
import functools
import heapq
import itertools
import json
import logging
import weakref
from collections import defaultdict
from operator import itemgetter
from pathlib import Path
from typing import NamedTuple

from findex.ast import parse, positive_terms
from findex.corpus import Document
from findex.index import Index, Posting, build_index
from findex.ranking import (
    BM25,
    ResultList,
    Scorer,
    SearchResult,
    TfIdf,
    make_snippet,
    precision_at_k,
)
from findex.store import open_index
from findex.utils import timed

logger = logging.getLogger(__name__)
DEFAULT_SCORER = BM25()


class _IndexCacheKey:
    """Hashable weak-reference key so lru_cache never leaks Index instances."""

    def __init__(self, index: Index) -> None:
        self._id = id(index)
        self._ref = weakref.ref(index)

    @property
    def index(self) -> Index | None:
        return self._ref()

    def __hash__(self) -> int:
        return hash(self._id)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, _IndexCacheKey):
            return False
        return self._id == other._id and self._ref() is other._ref()


class CacheInfo(NamedTuple):
    hits: int
    misses: int
    maxsize: int | None
    currsize: int


def _score_term(scorer: Scorer, term: str, posting: Posting, index: Index) -> float:
    return scorer.score(term, posting, index)


@functools.lru_cache(maxsize=256)
def _cached_ranked_ids(
    index_key: _IndexCacheKey,
    query: str,
    scorer: Scorer,
    k: int,
) -> tuple[tuple[int, float], ...]:
    index = index_key.index
    if index is None or k <= 0:
        return ()

    tree = parse(query)
    if tree is None:
        return ()

    matched_docs = tree.evaluate(index)
    if not matched_docs:
        return ()

    terms = positive_terms(tree)
    if not terms:
        return tuple((doc_id, 0.0) for doc_id in sorted(matched_docs)[:k])

    scores: defaultdict[int, float] = defaultdict(float)
    for term in terms:
        if term not in index:
            continue
        for posting in index[term]:
            if posting.doc_id in matched_docs:
                scores[posting.doc_id] += _score_term(scorer, term, posting, index)

    top_pairs = heapq.nlargest(k, scores.items(), key=itemgetter(1))
    return tuple(top_pairs)


def search_cache_info() -> CacheInfo:
    """Return lru_cache statistics for query evaluation."""
    info = _cached_ranked_ids.cache_info()
    return CacheInfo(info.hits, info.misses, info.maxsize, info.currsize)


def search_cache_hits() -> int:
    """Return the number of cache hits accumulated by ranked searches."""
    return _cached_ranked_ids.cache_info().hits


def clear_search_cache() -> None:
    """Clear the query result cache."""
    _cached_ranked_ids.cache_clear()


@timed
def search(
    index: Index,
    query: str,
    *,
    scorer: Scorer = DEFAULT_SCORER,
    k: int = 10,
    include_snippets: bool = True,
    with_snippets: bool | None = None,
) -> list[SearchResult]:
    """Parse query, evaluate matching docs, rank with `scorer`, and return top-k."""
    if with_snippets is not None:
        include_snippets = with_snippets

    before = _cached_ranked_ids.cache_info()
    ranked_pairs = _cached_ranked_ids(_IndexCacheKey(index), query, scorer, k)
    after = _cached_ranked_ids.cache_info()
    if after.hits > before.hits:
        logger.info("Cache HIT for query %r (%s)", query, after)
    else:
        logger.info("Cache MISS for query %r (%s)", query, after)

    tree = parse(query)
    terms = positive_terms(tree)

    results: list[SearchResult] = []
    for doc_id, doc_score in ranked_pairs:
        meta = index.doc_meta[doc_id]
        snippet = ""
        if include_snippets:
            snippet = _build_doc_snippet(index, doc_id, meta.path, meta.source_id, terms)
        results.append(
            SearchResult(
                doc_id=doc_id,
                score=doc_score,
                title=meta.title,
                snippet=snippet,
            )
        )
    return ResultList(results)


def boolean_search(index: Index, query: str) -> list[int]:
    """Return matching document IDs in deterministic document order."""
    tree = parse(query)
    return sorted(tree.evaluate(index)) if tree is not None else []


def _build_doc_snippet(
    index: Index,
    doc_id: int,
    path_str: str,
    source_id: str,
    terms: list[str],
) -> str:
    positions: tuple[int, ...] = ()
    for term in terms:
        if term in index:
            for posting in index[term]:
                if posting.doc_id == doc_id and posting.positions:
                    positions = posting.positions
                    break
        if positions:
            break

    path = Path(path_str)
    if not path.is_file():
        return ""

    if path.suffix.lower() == ".jsonl":
        raw_text = _read_jsonl_doc_text(path, source_id)
    else:
        try:
            raw_text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return ""

    return make_snippet(raw_text, terms, positions=positions, window=80)


def _read_jsonl_doc_text(path: Path, source_id: str) -> str:
    if ":" not in source_id:
        return ""
    _, _, line_str = source_id.rpartition(":")
    if not line_str.isdigit():
        return ""
    line_idx = int(line_str) - 1
    if line_idx < 0:
        return ""
    try:
        with path.open("r", encoding="utf-8", errors="replace") as fh:
            raw_line = next(itertools.islice(fh, line_idx, line_idx + 1), "")
        if not raw_line:
            return ""
        record = json.loads(raw_line)
        return str(record.get("text") or record.get("content") or record.get("body") or "")
    except (OSError, json.JSONDecodeError):
        return ""


def run_sanity_checks(index: Index) -> None:
    """Demonstrate the three M2 ranking sanity checks."""
    if len(index) < 2 or index.num_docs < 2:
        print("Need at least 2 terms and 2 documents in the index for sanity checks.")
        return

    by_df = sorted(index.keys(), key=lambda t: index.df(t))
    rare_term = by_df[0]
    common_term = by_df[-1]

    print("=== M2 Ranking Sanity Checks ===")
    # 1. Rare term outranks common term
    r_rare = search(
        index,
        f"{rare_term} OR {common_term}",
        scorer=BM25(),
        k=3,
        include_snippets=False,
    )
    print(
        f"1. Rare ({rare_term!r}, df={index.df(rare_term)}) "
        f"vs Common ({common_term!r}, df={index.df(common_term)}):"
    )
    for r in r_rare:
        print(f"   doc_id={r.doc_id} score={r.score:.4f} title={r.title!r}")

    # 2. BM25 saturation
    sat_docs = [
        Document("tf1.txt", Path("tf1.txt"), "alpha beta gamma delta"),
        Document("tf10.txt", Path("tf10.txt"), ("alpha " * 10) + "beta gamma delta"),
        Document("tf20.txt", Path("tf20.txt"), ("alpha " * 20) + "beta gamma delta"),
    ]
    sat_idx = build_index(sat_docs)
    bm25 = BM25()
    s1 = bm25.score("alpha", sat_idx["alpha"][0], sat_idx)
    s10 = bm25.score("alpha", sat_idx["alpha"][1], sat_idx)
    s20 = bm25.score("alpha", sat_idx["alpha"][2], sat_idx)
    print(
        f"2. BM25 saturation: tf=1 -> {s1:.4f}, tf=10 -> {s10:.4f}, "
        f"tf=20 -> {s20:.4f} (20th rep delta: {s20 - s10:.4f})"
    )

    # 3. Document length normalization
    len_docs = [
        Document("short.txt", Path("short.txt"), "needle in haystack"),
        Document("long.txt", Path("long.txt"), "needle " + ("filler " * 60)),
    ]
    len_idx = build_index(len_docs)
    len_res = search(len_idx, "needle", scorer=BM25(), k=2, include_snippets=False)
    print(
        f"3. Length normalization: short (len={len_idx.doc_length(0)}) "
        f"score={len_res[0].score:.4f} vs long (len={len_idx.doc_length(1)}) "
        f"score={len_res[1].score:.4f}"
    )


def evaluate_queries(index: Index, eval_file: Path) -> None:
    """Compute Precision@5 for TF-IDF vs BM25 from {query: [relevant_doc_ids]} JSON."""
    judgments: dict[str, list[int]] = json.loads(eval_file.read_text(encoding="utf-8"))
    tfidf_scores: list[float] = []
    bm25_scores: list[float] = []

    print("| # | Query | TF-IDF P@5 | BM25 P@5 |")
    print("| ---: | --- | ---: | ---: |")
    for idx_num, (q, rel_list) in enumerate(judgments.items(), start=1):
        rel_set = set(rel_list)
        r_tfidf = [r.doc_id for r in search(index, q, scorer=TfIdf(), k=5, include_snippets=False)]
        r_bm25 = [r.doc_id for r in search(index, q, scorer=BM25(), k=5, include_snippets=False)]
        p_tfidf = precision_at_k(r_tfidf, rel_set, k=5)
        p_bm25 = precision_at_k(r_bm25, rel_set, k=5)
        tfidf_scores.append(p_tfidf)
        bm25_scores.append(p_bm25)
        print(f"| {idx_num} | `{q}` | {p_tfidf:.2f} | {p_bm25:.2f} |")

    mean_tfidf = sum(tfidf_scores) / len(tfidf_scores) if tfidf_scores else 0.0
    mean_bm25 = sum(bm25_scores) / len(bm25_scores) if bm25_scores else 0.0
    print(
        f"| **Mean** | **All {len(judgments)} queries** | "
        f"**{mean_tfidf:.3f}** | **{mean_bm25:.3f}** |"
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="python -m findex.search",
        description="Ranked search with Boolean/phrase AST queries, TF-IDF/BM25, and snippets.",
    )
    parser.add_argument("index_path", type=Path, help="Path to saved index file")
    parser.add_argument("query", nargs="?", default="", help="Search query")
    parser.add_argument(
        "--scorer",
        choices=("bm25", "tfidf"),
        default="bm25",
        help="Relevance ranking model (default: bm25)",
    )
    parser.add_argument("-k", type=int, default=10, help="Number of results to return")
    parser.add_argument(
        "--sanity-checks",
        action="store_true",
        help="Run the three M2 ranking sanity checks",
    )
    parser.add_argument(
        "--cache-demo",
        action="store_true",
        help="Run the query twice to demonstrate lru_cache hit in timing logs",
    )
    parser.add_argument(
        "--eval",
        type=Path,
        default=None,
        help="Path to JSON file with 10 hand-labeled queries for Precision@5 table",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s [%(name)s]: %(message)s",
    )
    scorer: Scorer = BM25() if args.scorer == "bm25" else TfIdf()

    with open_index(args.index_path) as index:
        if args.sanity_checks:
            run_sanity_checks(index)
            return

        if args.eval is not None:
            evaluate_queries(index, args.eval)
            return

        results = search(index, args.query, scorer=scorer, k=args.k)
        if args.cache_demo:
            _ = search(index, args.query, scorer=scorer, k=args.k)
            print(f"Cache stats: {search_cache_info()}")

        print(f"Index:   {index!r}")
        print(f"Query:   {args.query!r} | Scorer: {type(scorer).__name__} | Hits: {len(results)}\n")
        for res in results:
            print(res)


if __name__ == "__main__":
    main()
