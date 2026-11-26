import argparseimport timeimport tracemallocfrom pathlib import Pathfrom findex.index import InvertedIndexfrom findex.store import loadfrom findex.tokenize import tokenizedef merge_and(a: list[int], b: list[int]) -> list[int]:
    """Перетин двох відсортованих списків doc_id за O(len(a) + len(b))."""
    result: list[int] = []
    i = j = 0
    while i < len(a) and j < len(b):
        if a[i] == b[j]:
            result.append(a[i])
            i += 1
            j += 1
        elif a[i] < b[j]:
            i += 1
        else:
            j += 1
    return result


def merge_or(a: list[int], b: list[int]) -> list[int]:
    """Об'єднання двох відсортованих списків doc_id без дублікатів за O(len(a) + len(b))."""
    result: list[int] = []
    i = j = 0
    while i < len(a) and j < len(b):
        if a[i] == b[j]:
            result.append(a[i])
            i += 1
            j += 1
        elif a[i] < b[j]:
            result.append(a[i])
            i += 1
        else:
            result.append(b[j])
            j += 1
    if i < len(a):
        result.extend(a[i:])
    if j < len(b):
        result.extend(b[j:])
    return result


def merge_not(a: list[int], b: list[int]) -> list[int]:
    """Різниця відсортованих списків (a \\ b) за O(len(a) + len(b))."""
    result: list[int] = []
    i = j = 0
    while i < len(a) and j < len(b):
        if a[i] == b[j]:
            i += 1
            j += 1
        elif a[i] < b[j]:
            result.append(a[i])
            i += 1
        else:
            j += 1
    if i < len(a):
        result.extend(a[i:])
    return result


def _parse_query_tokens(query: str) -> list[str]:
    """Розбирає запит і вставляє неявний оператор AND між сусідніми термами."""
    raw_parts = query.split()
    tokens: list[str] = []
    for part in raw_parts:
        if part in {"AND", "OR", "NOT"}:
            tokens.append(part)
        else:
            tokens.extend(tokenize(part))

    normalized: list[str] = []
    for tok in tokens:
        if normalized:
            prev = normalized[-1]
            prev_is_term = prev not in {"AND", "OR", "NOT"}
            curr_is_term_or_not = tok not in {"AND", "OR"}
            if prev_is_term and curr_is_term_or_not:
                normalized.append("AND")
        normalized.append(tok)
    return normalized


def search(index: InvertedIndex, query: str, *, engine: str = "merge") -> list[int]:
    """Виконує булевий пошук (AND, OR, NOT) за допомогою 'merge' або 'set'."""
    if engine not in {"merge", "set"}:
        raise ValueError(f"Непідтримуваний рушій пошуку: {engine}")

    tokens = _parse_query_tokens(query)
    if not tokens:
        return []

    all_docs = sorted(index.doc_meta.keys())

    if engine == "set":
        return _eval_set(index, tokens, all_docs)
    return _eval_merge(index, tokens, all_docs)


def _get_doc_ids(index: InvertedIndex, term: str) -> list[int]:
    return [p.doc_id for p in index.postings.get(term, ())]


def _eval_merge(index: InvertedIndex, tokens: list[str], all_docs: list[int]) -> list[int]:
    current: list[int] | None = None
    pending_op = "AND"
    negate_next = False

    for tok in tokens:
        if tok in {"AND", "OR"}:
            pending_op = tok
        elif tok == "NOT":
            negate_next = not negate_next
        else:
            rhs = _get_doc_ids(index, tok)
            if current is None:
                current = merge_not(all_docs, rhs) if negate_next else rhs
            elif pending_op == "AND":
                current = merge_not(current, rhs) if negate_next else merge_and(current, rhs)
            else:
                rhs_effective = merge_not(all_docs, rhs) if negate_next else rhs
                current = merge_or(current, rhs_effective)
            pending_op = "AND"
            negate_next = False

    return current or []


def _eval_set(index: InvertedIndex, tokens: list[str], all_docs: list[int]) -> list[int]:
    all_set = set(all_docs)
    current: set[int] | None = None
    pending_op = "AND"
    negate_next = False

    for tok in tokens:
        if tok in {"AND", "OR"}:
            pending_op = tok
        elif tok == "NOT":
            negate_next = not negate_next
        else:
            rhs = set(_get_doc_ids(index, tok))
            if current is None:
                current = (all_set - rhs) if negate_next else rhs
            elif pending_op == "AND":
                current = (current - rhs) if negate_next else (current & rhs)
            else:
                rhs_effective = (all_set - rhs) if negate_next else rhs
                current = current | rhs_effective
            pending_op = "AND"
            negate_next = False

    return sorted(current) if current is not None else []


def run_benchmark(index: InvertedIndex, iterations: int = 1000) -> None:
    """Порівнює швидкість merge та set на 2 найчастіших і 2 найрідкісніших термах."""
    sorted_terms = sorted(index.postings.items(), key=lambda item: len(item[1]))
    if len(sorted_terms) < 2:
        print("Недостатньо термів в індексі для бенчмарку.")
        return

    rare_terms = [t for t, _ in sorted_terms[:2]]
    common_terms = [t for t, _ in sorted_terms[-2:]]

    queries = [
        ("2 найчастіші терми (AND)", f"{common_terms[0]} AND {common_terms[1]}"),
        ("2 найчастіші терми (OR)", f"{common_terms[0]} OR {common_terms[1]}"),
        ("2 найрідкісніші терми (AND)", f"{rare_terms[0]} AND {rare_terms[1]}"),
        ("2 найрідкісніші терми (OR)", f"{rare_terms[0]} OR {rare_terms[1]}"),
    ]

    print(f"Benchmark ({iterations} iterations per query):")
    print(f"  Common terms: {common_terms} (df={[len(index.postings[t]) for t in common_terms]})")
    print(f"  Rare terms:   {rare_terms} (df={[len(index.postings[t]) for t in rare_terms]})\n")

    for label, q in queries:
        t0 = time.perf_counter()
        for _ in range(iterations):
            res_merge = search(index, q, engine="merge")
        t_merge = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        for _ in range(iterations):
            res_set = search(index, q, engine="set")
        t_set = (time.perf_counter() - t0) * 1000

        assert res_merge == res_set
        print(
            f"{label:<30} | hits={len(res_merge):<5} | "
            f"merge={t_merge:7.2f} ms | set={t_set:7.2f} ms"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Булевий пошук по інвертованому індексу.")
    parser.add_argument("index_path", type=Path, help="Шлях до збереженого індексу")
    parser.add_argument("query", nargs="?", default="", help="Пошуковий запит (AND/OR/NOT)")
    parser.add_argument(
        "--engine",
        choices=("merge", "set"),
        default="merge",
        help="Алгоритм виконання запиту: merge або set",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=10,
        help="Максимальна кількість результатів для виводу",
    )
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Запустити порівняння merge проти set на найчастіших і найрідкісніших термах",
    )
    args = parser.parse_args()

    tracemalloc.start()
    start_time = time.perf_counter()

    index = load(args.index_path)

    if args.benchmark:
        run_benchmark(index)
    else:
        doc_ids = search(index, args.query, engine=args.engine)
        elapsed = time.perf_counter() - start_time
        _, peak_bytes = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        print(f"Query:       {args.query!r} (engine={args.engine})")
        print(f"Matches:     {len(doc_ids):,}")
        print(f"Elapsed:     {elapsed * 1000:.2f} ms")
        print(f"Peak memory: {peak_bytes / (1024 * 1024):.2f} MiB\n")

        for doc_id in doc_ids[: args.top]:
            meta = index.doc_meta[doc_id]
            print(f"[{doc_id:>4}] {meta.title} ({meta.source_id})")
        return

    tracemalloc.stop()


if __name__ == "__main__":
    main()