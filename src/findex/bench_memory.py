import argparseimport gcimport pickleimport tempfileimport timeimport tracemallocfrom array import arrayfrom collections import Counter, defaultdictfrom dataclasses import dataclassfrom itertools import islicefrom pathlib import Pathfrom findex.corpus import iter_documentsfrom findex.tokenize import tokenize@dataclass
class PlainPosting:
    doc_id: int
    tf: int


@dataclass(slots=True)
class SlottedPosting:
    doc_id: int
    tf: int


def _build_plain(root: Path, limit: int | None) -> dict[str, list[PlainPosting]]:
    postings: defaultdict[str, list[PlainPosting]] = defaultdict(list)
    for doc_id, doc in enumerate(islice(iter_documents(root), limit)):
        for term, tf in Counter(tokenize(doc.text)).items():
            postings[term].append(PlainPosting(doc_id=doc_id, tf=tf))
    return dict(postings)


def _build_slotted(root: Path, limit: int | None) -> dict[str, list[SlottedPosting]]:
    postings: defaultdict[str, list[SlottedPosting]] = defaultdict(list)
    for doc_id, doc in enumerate(islice(iter_documents(root), limit)):
        for term, tf in Counter(tokenize(doc.text)).items():
            postings[term].append(SlottedPosting(doc_id=doc_id, tf=tf))
    return dict(postings)


def _build_arrays(root: Path, limit: int | None) -> dict[str, tuple[array[int], array[int]]]:
    postings: defaultdict[str, tuple[array[int], array[int]]] = defaultdict(
        lambda: (array("I"), array("I"))
    )
    for doc_id, doc in enumerate(islice(iter_documents(root), limit)):
        for term, tf in Counter(tokenize(doc.text)).items():
            doc_ids, tfs = postings[term]
            doc_ids.append(doc_id)
            tfs.append(tf)
    return dict(postings)


def _measure_variant(name: str, builder) -> tuple[str, float, float, float]:
    gc.collect()
    tracemalloc.start()
    data = builder()
    _, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pkl") as tmp:
        tmp_path = Path(tmp.name)

    try:
        with tmp_path.open("wb") as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
        file_size_mib = tmp_path.stat().st_size / (1024 * 1024)

        del data
        gc.collect()

        t0 = time.perf_counter()
        with tmp_path.open("rb") as f:
            loaded = pickle.load(f)
        load_ms = (time.perf_counter() - t0) * 1000
        del loaded
    finally:
        tmp_path.unlink(missing_ok=True)

    return name, peak_bytes / (1024 * 1024), file_size_mib, load_ms


def main() -> None:
    parser = argparse.ArgumentParser(description="Замір споживання пам'яті для M4.")
    parser.add_argument("corpus", type=Path, help="Шлях до директорії корпусу")
    parser.add_argument("--limit", type=int, default=None, help="Обмеження кількості документів")
    args = parser.parse_args()

    rows = [
        _measure_variant(
            "`list[Posting]` with plain `@dataclass`",
            lambda: _build_plain(args.corpus, args.limit),
        ),
        _measure_variant(
            "`list[Posting]` with `slots=True`",
            lambda: _build_slotted(args.corpus, args.limit),
        ),
        _measure_variant(
            "`array('I')` pairs `(doc_ids, tfs)` — no objects",
            lambda: _build_arrays(args.corpus, args.limit),
        ),
    ]

    print("| Postings representation | Peak memory (build) | Index file size | Load time |")
    print("| :--- | :---: | :---: | :---: |")
    for label, peak_mib, size_mib, load_ms in rows:
        print(f"| {label} | {peak_mib:.2f} MiB | {size_mib:.2f} MiB | {load_ms:.2f} ms |")


if __name__ == "__main__":
    main()