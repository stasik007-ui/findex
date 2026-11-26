from pathlib import Path
import tempfile

from findex.index import build_index
from findex.ranking import BM25, TfIdf
from findex.search import clear_search_cache, search


def run_sanity_checks() -> tuple[float, float, float]:
    """Verify the three core IR ranking invariants on a controlled corpus."""
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        for i in range(5):
            (root / f"common_{i}.txt").write_text(
                f"Doc {i}\npython language tutorial guide", encoding="utf-8"
            )
        (root / "rare.txt").write_text(
            "Rare Doc\npython bm25 ranking algorithm", encoding="utf-8"
        )
        (root / "rust_1.txt").write_text("Rust 1\nrust systems", encoding="utf-8")
        (root / "rust_20.txt").write_text(
            "Rust 20\n" + ("rust " * 20) + "systems", encoding="utf-8"
        )
        (root / "short.txt").write_text("Short Doc\nsearch engine", encoding="utf-8")
        (root / "long.txt").write_text(
            "Long Doc\nsearch " + ("filler word text " * 60), encoding="utf-8"
        )

        idx = build_index(root)
        bm25 = BM25(k1=1.5, b=0.75)

        res_rare = search(idx, "python OR bm25", scorer=bm25, k=3, with_snippets=False)
        assert res_rare[0].doc_id == 6

        r1 = search(idx, "rust", scorer=bm25, k=2, with_snippets=False)
        score_map = {r.doc_id: r.score for r in r1}
        saturation_ratio = score_map[8] / score_map[7]
        assert saturation_ratio < 2.5

        res_len = search(idx, "search", scorer=bm25, k=2, with_snippets=False)
        assert res_len[0].doc_id == 9 and res_len[1].doc_id == 5

        return res_rare[0].score, saturation_ratio, res_len[0].score


def compute_precision_at_5() -> tuple[float, float]:
    """Compute Mean Precision@5 over 10 labeled benchmark queries for TF-IDF vs BM25."""
    corpus_docs = {
        "d0.txt": "Python Iterators\npython generators yield lazy evaluation protocol",
        "d1.txt": "Inverted Index\ninverted index postings dictionary hash table",
        "d2.txt": "BM25 Ranking\nbm25 ranking algorithm term frequency saturation length",
        "d3.txt": "TF-IDF Baseline\ntfidf weighting rare terms inverse document frequency",
        "d4.txt": "Boolean Queries\nboolean search recursive descent parser phrase queries",
        "d5.txt": "Unicode Text\nunicode nfc normalization casefold tokenization regex",
        "d6.txt": "Memory Optimization\ndataclasses slots array memory layout tracemalloc",
        "d7.txt": "Context Managers\ncontext manager protocol enter exit decorators wraps",
        "d8.txt": "Search Snippets\nsnippets window highlight term positions phrase",
        "d9.txt": "Long Overview\npython search engine " + ("general notes text " * 50),
    }

    benchmark_queries: list[tuple[str, set[int]]] = [
        ("python generators", {0}),
        ("inverted index postings", {1}),
        ("bm25 ranking saturation", {2}),
        ("tfidf rare terms", {3}),
        ("recursive descent parser", {4}),
        ("unicode nfc casefold", {5}),
        ("slots array memory", {6}),
        ("context manager decorators", {7}),
        ("snippets highlight positions", {8}),
        ("python search", {0, 8, 9}),
    ]

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        for fname, content in corpus_docs.items():
            (root / fname).write_text(content, encoding="utf-8")

        idx = build_index(root)
        clear_search_cache()

        tfidf_p5: list[float] = []
        bm25_p5: list[float] = []

        for q, relevant in benchmark_queries:
            res_tfidf = search(idx, q, scorer=TfIdf(), k=5, with_snippets=False)
            res_bm25 = search(idx, q, scorer=BM25(), k=5, with_snippets=False)

            hits_tfidf = sum(1 for r in res_tfidf if r.doc_id in relevant)
            hits_bm25 = sum(1 for r in res_bm25 if r.doc_id in relevant)

            denom = min(5, len(relevant))
            tfidf_p5.append(hits_tfidf / denom)
            bm25_p5.append(hits_bm25 / denom)

        return sum(tfidf_p5) / len(tfidf_p5), sum(bm25_p5) / len(bm25_p5)