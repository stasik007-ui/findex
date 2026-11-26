from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest
from hypothesis import given
from hypothesis import strategies as st

from findex.corpus import iter_documents
from findex.index import DocMeta, Index, Posting, build_index, open_index
from findex.store import load, save


def test_posting_and_docmeta_are_frozen_slotted_and_hashable() -> None:
    p = Posting(doc_id=1, tf=3, positions=(0, 4, 9))
    m = DocMeta(doc_id=1, source_id="a.txt", path="data/a.txt", title="Title")

    assert len({p, Posting(doc_id=1, tf=3, positions=(0, 4, 9))}) == 1
    assert len({m, DocMeta(doc_id=1, source_id="a.txt", path="data/a.txt", title="Title")}) == 1
    assert not hasattr(p, "__dict__")
    assert not hasattr(m, "__dict__")

    with pytest.raises(FrozenInstanceError):
        p.tf = 5  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        m.title = "New"  # type: ignore[misc]


def test_index_mapping_protocol_and_properties(sample_index: Index) -> None:
    assert sample_index.num_docs == 4
    assert len(sample_index) > 10
    assert "python" in sample_index
    assert "nonexistent_term_xyz" not in sample_index
    assert sample_index.df("python") == 2
    assert sample_index.avg_doc_length > 0
    assert repr(sample_index) == f"Index(terms={len(sample_index)}, docs=4)"

    iterated_terms = list(sample_index)
    assert len(iterated_terms) == len(sample_index)

    with pytest.raises(KeyError):
        _ = sample_index["nonexistent_term_xyz"]


def test_open_index_context_manager_releases_on_exception(saved_index_path: Path) -> None:
    with pytest.raises(RuntimeError), open_index(saved_index_path) as loaded:
        assert loaded.num_docs == 4
        assert not loaded.closed
        raise RuntimeError("boom")

    assert loaded.closed is True
    assert len(loaded) == 0


def test_build_index_without_positions_and_limit(sample_corpus_dir: Path) -> None:
    idx = build_index(sample_corpus_dir, limit=2, with_positions=False)
    assert idx.num_docs == 2
    for plist in idx.postings.values():
        for posting in plist:
            assert posting.positions == ()


@pytest.mark.parametrize("fmt", ["binary", "pickle"])
def test_store_roundtrip_formats(tmp_path: Path, sample_index: Index, fmt: str) -> None:
    assert fmt in {"binary", "pickle"}
    out = tmp_path / f"idx.{fmt}"
    save(sample_index, out, fmt=fmt)
    restored = load(out)

    assert restored.postings == sample_index.postings
    assert restored.doc_lengths == sample_index.doc_lengths
    assert restored.doc_meta == sample_index.doc_meta


def test_corpus_skips_undecodable_file_and_missing_root(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    bad_file = tmp_path / "broken.txt"
    bad_file.write_bytes(b"\xff\xfe\xfd")
    good_file = tmp_path / "ok.txt"
    good_file.write_text("Valid document", encoding="utf-8")

    with caplog.at_level("WARNING"):
        docs = list(iter_documents(tmp_path))

    assert len(docs) == 1
    assert any("undecodable" in rec.message for rec in caplog.records)

    with pytest.raises(FileNotFoundError):
        list(iter_documents(tmp_path / "does_not_exist"))


@given(
    st.lists(
        st.text(alphabet=st.characters(blacklist_categories=("Cs",)), min_size=1, max_size=60),
        min_size=1,
        max_size=8,
    )
)
def test_property_roundtrip_and_sorted_postings(
    tmp_path_factory: pytest.TempPathFactory, docs: list[str]
) -> None:
    corpus_dir = tmp_path_factory.mktemp("prop_corpus")
    for i, body in enumerate(docs):
        (corpus_dir / f"doc_{i}.txt").write_text(body, encoding="utf-8")

    idx = build_index(corpus_dir, with_positions=True)
    for plist in idx.postings.values():
        doc_ids = [p.doc_id for p in plist]
        assert doc_ids == sorted(doc_ids)

    bin_path = corpus_dir / "roundtrip.bin"
    save(idx, bin_path, fmt="binary")
    restored = load(bin_path, fmt="binary")
    assert restored.postings == idx.postings
    assert restored.doc_lengths == idx.doc_lengths
