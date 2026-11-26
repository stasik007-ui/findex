import logging
from pathlib import Path

import pytest

from findex.decorators import timed
from findex.evaluate import compute_precision_at_5, run_sanity_checks
from findex.index import Index
from findex.ranking import BM25, ResultList, Scorer, SearchResult, TfIdf, explain_score
from findex.search import clear_search_cache, search, search_cache_hits
from findex.stats import compute_corpus_stats, compute_index_stats


def test_scorers_satisfy_protocol_and_sanity_checks() -> None:
    assert isinstance(BM25(), Scorer)
    assert isinstance(TfIdf(), Scorer)

    rare_score, sat_ratio, short_score = run_sanity_checks()
    assert rare_score > 0
    assert sat_ratio < 2.5
    assert short_score > 0


def test_precision_at_5_benchmark() -> None:
    tfidf_p5, bm25_p5 = compute_precision_at_5()
    assert tfidf_p5 >= 0.8
    assert bm25_p5 >= 0.8


def test_search_caching_snippets_and_result_list(sample_index: Index) -> None:
    clear_search_cache()
    res1 = search(sample_index, "generators", scorer=BM25(), k=5)
    assert isinstance(res1, ResultList)
    assert isinstance(res1[:1], ResultList)
    assert bool(res1) is True
    assert len(res1) == 1
    assert "[generators]" in res1[0].snippet

    _ = search(sample_index, "generators", scorer=BM25(), k=5)
    assert search_cache_hits() >= 1

    r_low = SearchResult(doc_id=1, score=0.2, title="A")
    r_high = SearchResult(doc_id=2, score=1.8, title="B")
    assert sorted([r_high, r_low]) == [r_low, r_high]
    assert "BM25" in explain_score(BM25(), "generators", sample_index["generators"][0], sample_index)
    assert "TfIdf" in explain_score(TfIdf(), "generators", sample_index["generators"][0], sample_index)


def test_timed_decorator_preserves_metadata_and_logs(caplog: pytest.LogCaptureFixture) -> None:
    @timed
    def sample_work(x: int) -> int:
        """Sample docstring."""
        return x * 2

    assert sample_work.__name__ == "sample_work"
    assert sample_work.__doc__ == "Sample docstring."

    with caplog.at_level(logging.DEBUG, logger="findex.decorators"):
        assert sample_work(21) == 42

    assert any("sample_work took" in rec.message for rec in caplog.records)


def test_stats_helpers(sample_corpus_dir: Path, sample_index: Index) -> None:
    idx_stats = compute_index_stats(sample_index, top_n=5)
    assert idx_stats.num_docs == 4
    assert idx_stats.total_tokens > 0
    assert len(idx_stats.top_terms) == 5

    doc_count, total_tokens, top = compute_corpus_stats(sample_corpus_dir, top_n=3)
    assert doc_count == 4
    assert total_tokens == idx_stats.total_tokens
    assert len(top) == 3