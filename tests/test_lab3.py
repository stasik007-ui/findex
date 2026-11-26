from pathlib import Path

import pytest

from findex.index import Index, build_index, open_index
from findex.query import And, Not, Or, Phrase, Term, explain_query, parse
from findex.ranking import BM25, ResultList, Scorer, SearchResult, TfIdf, explain_score
from findex.search import _cached_ranked_ids, search  # pyright: ignore[reportPrivateUsage]
from findex.store import save


def test_index_mapping_protocol_and_context_manager(tmp_path: Path) -> None:
    (tmp_path / "a.txt").write_text("First\npython search", encoding="utf-8")
    (tmp_path / "b.txt").write_text("Second\npython engine ranking", encoding="utf-8")

    idx = build_index(tmp_path)
    assert isinstance(idx, Index)
    assert len(idx) == 6
    assert "python" in idx
    assert "absent" not in idx
    assert list(idx) == ["first", "python", "search", "second", "engine", "ranking"]
    assert idx.num_docs == 2
    assert idx.avg_doc_length == pytest.approx(3.5)
    assert idx.df("python") == 2
    assert repr(idx) == "Index(terms=6, docs=2)"

    with pytest.raises(KeyError):
        _ = idx["absent"]

    idx_path = tmp_path / "test.bin"
    save(idx, idx_path)

    with pytest.raises(RuntimeError), open_index(idx_path) as loaded_ix:
        assert len(loaded_ix) == 6
        raise RuntimeError("Simulated failure")

    assert loaded_ix.closed is True
    assert len(loaded_ix) == 0


def test_query_parser_ast_and_operators(tmp_path: Path) -> None:
    assert parse("a OR b c") == Or(Term("a"), And(Term("b"), Term("c")))
    assert parse('python AND (async OR await) NOT java "exact phrase"') == And(
        And(
            And(Term("python"), Or(Term("async"), Term("await"))),
            Not(Term("java")),
        ),
        Phrase(("exact", "phrase")),
    )

    composed = (Term("python") | Term("rust")) & ~Term("java")
    assert composed == And(Or(Term("python"), Term("rust")), Not(Term("java")))
    assert "OR" in explain_query(composed)

    (tmp_path / "d0.txt").write_text("Doc 0\nevent loop in python", encoding="utf-8")
    (tmp_path / "d1.txt").write_text("Doc 1\nloop of the event", encoding="utf-8")
    idx = build_index(tmp_path, with_positions=True)

    assert parse('"event loop"').evaluate(idx) == {0}  # type: ignore[union-attr]


def test_scorers_caching_snippets_and_result_list(tmp_path: Path) -> None:
    (tmp_path / "d0.txt").write_text(
        "Intro\nLearning python programming with generators", encoding="utf-8"
    )
    (tmp_path / "d1.txt").write_text(
        "Guide\nAdvanced search engine with BM25 ranking", encoding="utf-8"
    )
    idx = build_index(tmp_path)

    assert isinstance(BM25(), Scorer)
    assert isinstance(TfIdf(), Scorer)

    _cached_ranked_ids.cache_clear()
    res1 = search(idx, "generators", scorer=BM25(), k=5)
    assert isinstance(res1, ResultList)
    assert isinstance(res1[:1], ResultList)
    assert len(res1) == 1
    assert "[generators]" in res1[0].snippet

    _ = search(idx, "generators", scorer=BM25(), k=5)
    assert _cached_ranked_ids.cache_info().hits >= 1

    r_low = SearchResult(doc_id=1, score=0.5, title="Low")
    r_high = SearchResult(doc_id=2, score=2.5, title="High")
    assert sorted([r_high, r_low]) == [r_low, r_high]
    assert "BM25" in explain_score(BM25(), "generators", idx["generators"][0], idx)
