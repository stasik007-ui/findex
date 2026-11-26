import argparse
from dataclasses import dataclass
from pathlib import Path
import tempfile
import time
import tracemalloc

from findex.corpus import iter_documents
from findex.index import build_index
from findex.store import load, save
from findex.tokenize import tokenize


@dataclass
class PlainPosting:
    doc_id: int
    tf: int


@dataclass(slots=True)
class SlottedPosting:
    doc_id: int
    tf: int


def _measure_plain(root: Path) -> float:
    tracemalloc.start()
    postings: dict[str, list[PlainPosting]] = {}
    for doc_id, doc in enumerate(iter_documents(root)):
        counts: dict[str, int] = {}
        for tok in tokenize(doc.text):
            counts[tok] = counts.get(tok, 0) + 1
        for term, tf in counts.items():
            postings.setdefault(term, []).append(PlainPosting(doc_id=doc_id, tf=tf))
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return peak / (1024 * 1024)


def _measure_slotted(root: Path) -> float:
    tracemalloc.start()
    postings: dict[str, list[SlottedPosting]] = {}
    for doc_id, doc in enumerate(iter_documents(root)):
        counts: dict[str, int] = {}
        for tok in tokenize(doc.text):
            counts[tok] = counts.get(tok, 0) + 1
        for term, tf in counts.items():
            postings.setdefault(term, []).append(SlottedPosting(doc_id=doc_id, tf=tf))
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return peak / (1024 * 1024)


def _measure_array(root: Path) -> tuple[float, float, float]:
    tracemalloc.start()
    idx = build_index(root, with_positions=False)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "bench.bin"
        save(idx, out, fmt="binary")
        size_mib = out.stat().st_size / (1024 * 1024)
        t0 = time.perf_counter()
        load(out, fmt="binary")
        load_ms = (time.perf_counter() - t0) * 1000
    return peak / (1024 * 1024), size_mib, load_ms


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark postings memory representations.")
    parser.add_argument("corpus", type=Path, help="Path to corpus directory")
    args = parser.parse_args()

    p_mem = _measure_plain(args.corpus)
    s_mem = _measure_slotted(args.corpus)
    a_mem, a_size, a_load = _measure_array(args.corpus)

    print("| Postings representation | Peak memory (build) | Index file size | Load time |")
    print("| :--- | :---: | :---: | :---: |")
    print(f"| `list[Posting]` plain `@dataclass` | {p_mem:.2f} MiB | — | — |")
    print(f"| `list[Posting]` `slots=True` | {s_mem:.2f} MiB | — | — |")
    print(f"| `array('I')` binary layout | {a_mem:.2f} MiB | {a_size:.2f} MiB | {a_load:.2f} ms |")


if __name__ == "__main__":
    main()