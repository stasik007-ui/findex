import argparse
from collections import Counter
from itertools import islice
from pathlib import Path
import time
import tracemalloc

from findex.corpus import iter_documents
from findex.tokenize import tokenize


def run_lazy(root: Path, limit: int | None) -> tuple[int, int, Counter[str]]:
    """Потоковий однопрохідний конвеєр у константній пам'яті."""
    docs = islice(iter_documents(root), limit)
    counts: Counter[str] = Counter()
    doc_count = 0
    total_tokens = 0

    for doc in docs:
        doc_count += 1
        for token in tokenize(doc.text):
            counts[token] += 1
            total_tokens += 1

    return doc_count, total_tokens, counts


def run_eager(root: Path, limit: int | None) -> tuple[int, int, Counter[str]]:
    """Навмисно жадібна версія (M4): вантажить усі документи й токени у списки."""
    docs = list(islice(iter_documents(root), limit))
    tokenized_docs = [list(tokenize(doc.text)) for doc in docs]
    counts: Counter[str] = Counter(
        token for doc_tokens in tokenized_docs for token in doc_tokens
    )
    doc_count = len(docs)
    total_tokens = sum(len(doc_tokens) for doc_tokens in tokenized_docs)

    return doc_count, total_tokens, counts


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Потоковий підрахунок статистики термів корпусу."
    )
    parser.add_argument("corpus", type=Path, help="Шлях до директорії корпусу або .jsonl")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Обробити лише перші N документів (через itertools.islice)",
    )
    parser.add_argument(
        "--eager",
        action="store_true",
        help="Використати жадібну реалізацію зі списками (для порівняння в M4)",
    )
    args = parser.parse_args()

    tracemalloc.start()
    start_time = time.perf_counter()

    if args.eager:
        doc_count, total_tokens, counts = run_eager(args.corpus, args.limit)
        mode = "eager (lists)"
    else:
        doc_count, total_tokens, counts = run_lazy(args.corpus, args.limit)
        mode = "lazy (generators)"

    elapsed = time.perf_counter() - start_time
    _, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    peak_mib = peak_bytes / (1024 * 1024)

    print(f"Mode:            {mode}")
    print(f"Documents:       {doc_count:,}")
    print(f"Total tokens:    {total_tokens:,}")
    print(f"Vocabulary size: {len(counts):,}")
    print(f"Elapsed time:    {elapsed:.3f} s")
    print(f"Peak memory:     {peak_mib:.2f} MiB ({peak_bytes:,} bytes)")
    print("\nTop 50 terms:")
    for rank, (term, freq) in enumerate(counts.most_common(50), start=1):
        print(f"{rank:>2}. {term:<20} {freq:,}")


if __name__ == "__main__":
    main()