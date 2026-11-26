import argparseimport timeimport tracemallocfrom collections import Counter, defaultdictfrom dataclasses import dataclass, fieldfrom itertools import islicefrom pathlib import Pathfrom findex.corpus import iter_documentsfrom findex.tokenize import tokenize@dataclass(frozen=True, slots=True)
class Posting:
    doc_id: int
    tf: int
    positions: tuple[int, ...] = ()


@dataclass(frozen=True, slots=True)
class DocMeta:
    doc_id: int
    source_id: str
    path: str
    title: str


@dataclass(slots=True)
class InvertedIndex:
    postings: dict[str, list[Posting]] = field(default_factory=dict)
    doc_lengths: dict[int, int] = field(default_factory=dict)
    doc_meta: dict[int, DocMeta] = field(default_factory=dict)


def _extract_title(text: str, fallback: str) -> str:
    for line in text.splitlines():
        stripped = line.strip()
        if stripped:
            return stripped[:120]
    return fallback


def build_index(
    root: Path,
    *,
    limit: int | None = None,
    with_positions: bool = False,
) -> InvertedIndex:
    """Будує інвертований індекс за один прохід по лінивому потоку документів."""
    postings_acc: defaultdict[str, list[Posting]] = defaultdict(list)
    doc_lengths: dict[int, int] = {}
    doc_meta: dict[int, DocMeta] = {}

    docs = islice(iter_documents(root), limit)
    for doc_id, doc in enumerate(docs):
        doc_meta[doc_id] = DocMeta(
            doc_id=doc_id,
            source_id=doc.doc_id,
            path=str(doc.path),
            title=_extract_title(doc.text, doc.path.stem),
        )

        if with_positions:
            term_positions: defaultdict[str, list[int]] = defaultdict(list)
            token_count = 0
            for pos, token in enumerate(tokenize(doc.text)):
                term_positions[token].append(pos)
                token_count += 1
            doc_lengths[doc_id] = token_count
            for term, pos_list in term_positions.items():
                postings_acc[term].append(
                    Posting(doc_id=doc_id, tf=len(pos_list), positions=tuple(pos_list))
                )
        else:
            term_counts = Counter(tokenize(doc.text))
            doc_lengths[doc_id] = sum(term_counts.values())
            for term, tf in term_counts.items():
                postings_acc[term].append(Posting(doc_id=doc_id, tf=tf))

    return InvertedIndex(
        postings=dict(postings_acc),
        doc_lengths=doc_lengths,
        doc_meta=doc_meta,
    )


def main() -> None:
    from findex.store import save

    parser = argparse.ArgumentParser(
        description="Побудова та збереження інвертованого індексу корпусу."
    )
    parser.add_argument("corpus", type=Path, help="Шлях до директорії корпусу або .jsonl")
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("index.bin"),
        help="Шлях до вихідного файлу індексу (за замовчуванням: index.bin)",
    )
    parser.add_argument(
        "--format",
        choices=("binary", "pickle"),
        default=None,
        help="Формат серіалізації (автовизначення за розширенням або binary)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Індексувати лише перші N документів",
    )
    parser.add_argument(
        "--positions",
        action="store_true",
        help="Зберігати кортеж позицій входжень кожного терма",
    )
    args = parser.parse_args()

    tracemalloc.start()
    start_time = time.perf_counter()

    index = build_index(args.corpus, limit=args.limit, with_positions=args.positions)
    save(index, args.out, fmt=args.format)

    elapsed = time.perf_counter() - start_time
    _, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    file_size_mib = args.out.stat().st_size / (1024 * 1024)
    peak_mib = peak_bytes / (1024 * 1024)

    print(f"Indexed documents: {len(index.doc_meta):,}")
    print(f"Vocabulary size:   {len(index.postings):,}")
    print(f"Output file:       {args.out} ({file_size_mib:.2f} MiB)")
    print(f"Elapsed time:      {elapsed:.3f} s")
    print(f"Peak memory:       {peak_mib:.2f} MiB ({peak_bytes:,} bytes)")


if __name__ == "__main__":
    main()