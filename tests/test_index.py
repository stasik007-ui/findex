from dataclasses import FrozenInstanceErrorfrom pathlib import Pathimport pytestfrom findex.index import DocMeta, Posting, build_indexfrom findex.search import searchfrom findex.store import load, savedef test_posting_and_docmeta_are_frozen_slotted_and_hashable() -> None:
    p = Posting(doc_id=1, tf=3, positions=(0, 4, 9))
    m = DocMeta(doc_id=1, source_id="a.txt", path="data/a.txt", title="Title")

    # Хешованість (можна класти у set і використовувати як ключ dict)
    assert len({p, Posting(doc_id=1, tf=3, positions=(0, 4, 9))}) == 1
    assert len({m, DocMeta(doc_id=1, source_id="a.txt", path="data/a.txt", title="Title")}) == 1

    # Відсутність __dict__ завдяки slots=True
    assert not hasattr(p, "__dict__")
    assert not hasattr(m, "__dict__")

    # Заборона зміни полів (frozen=True)
    with pytest.raises(FrozenInstanceError):
        p.tf = 5  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        m.title = "New"  # type: ignore[misc]


def test_build_index_sorted_postings_and_boolean_search(tmp_path: Path) -> None:
    (tmp_path / "d0.txt").write_text("Doc Zero\npython search engine", encoding="utf-8")
    (tmp_path / "d1.txt").write_text("Doc One\npython database sql", encoding="utf-8")
    (tmp_path / "d2.txt").write_text("Doc Two\nsearch engine ranking", encoding="utf-8")
    (tmp_path / "d3.txt").write_text("Doc Three\nrust systems programming", encoding="utf-8")

    idx = build_index(tmp_path, with_positions=True)

    # Перевірка монотонного сортування за doc_id
    for plist in idx.postings.values():
        doc_ids = [p.doc_id for p in plist]
        assert doc_ids == sorted(doc_ids)

    queries = [
        "python search",
        "python AND search",
        "python OR rust",
        "python NOT sql",
        "NOT python",
        "search OR rust NOT engine",
    ]
    for q in queries:
        res_merge = search(idx, q, engine="merge")
        res_set = search(idx, q, engine="set")
        assert res_merge == res_set

    assert search(idx, "python search", engine="merge") == [0]
    assert search(idx, "python OR rust", engine="merge") == [0, 1, 3]
    assert search(idx, "python NOT sql", engine="merge") == [0]
    assert search(idx, "NOT python", engine="merge") == [2, 3]


@pytest.mark.parametrize("fmt", ["binary", "pickle"])
def test_store_save_and_load_roundtrip(tmp_path: Path, fmt: str) -> None:
    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir()
    (corpus_dir / "a.txt").write_text("Перший\nпривіт світ python", encoding="utf-8")
    (corpus_dir / "b.txt").write_text("Другий\nсвіт даних", encoding="utf-8")

    original = build_index(corpus_dir, with_positions=True)
    out_file = tmp_path / f"index.{ 'pkl' if fmt == 'pickle' else 'bin' }"

    save(original, out_file, fmt=fmt)
    restored = load(out_file)

    assert restored.doc_meta == original.doc_meta
    assert restored.doc_lengths == original.doc_lengths
    assert restored.postings == original.postings