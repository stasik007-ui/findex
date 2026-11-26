from __future__ import annotations

import argparse
import functools
import itertools
import logging
from collections import Counter, defaultdict
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path

from findex.corpus import Document, iter_documents
from findex.tokenize import tokenize
from findex.utils import timed

DocId = int


@dataclass(frozen=True, slots=True)
class Posting:
    doc_id: int
    tf: int
    positions: tuple[int, ...] = ()


@dataclass(frozen=True, slots=True)
class DocMeta:
    doc_id: int
    path: str
    title: str
    source_id: str = ""


class Index(Mapping[str, list[Posting]]):
    """In-memory inverted index implementing Python's Mapping protocol."""

    def __init__(
        self,
        postings: dict[str, list[Posting]],
        doc_lengths: dict[int, int],
        doc_meta: dict[int, DocMeta],
        include_positions: bool = False,
    ) -> None:
        self._postings = postings
        self._doc_lengths = doc_lengths
        self._doc_meta = doc_meta
        self.include_positions = include_positions
        self._closed = False

    def __len__(self) -> int:
        return len(self._postings)

    def __contains__(self, term: object) -> bool:
        return term in self._postings

    def __getitem__(self, term: str) -> list[Posting]:
        return self._postings[term]

    def __iter__(self) -> Iterator[str]:
        return iter(self._postings)

    def __repr__(self) -> str:
        return f"Index(terms={len(self):_}, docs={self.num_docs:_})"

    @property
    def postings(self) -> dict[str, list[Posting]]:
        return self._postings

    @property
    def doc_lengths(self) -> dict[int, int]:
        return self._doc_lengths

    @property
    def doc_meta(self) -> dict[int, DocMeta]:
        return self._doc_meta

    @property
    def num_docs(self) -> int:
        return len(self._doc_lengths)

    @property
    def vocab_size(self) -> int:
        return len(self._postings)

    @functools.cached_property
    def avg_doc_length(self) -> float:
        if self.num_docs == 0:
            return 0.0
        return sum(self._doc_lengths.values()) / self.num_docs

    def doc_length(self, doc_id: int) -> int:
        return self._doc_lengths[doc_id]

    def df(self, term: str) -> int:
        plist = self._postings.get(term)
        return len(plist) if plist is not None else 0

    def close(self) -> None:
        """Release cached properties and mark the index as closed."""
        self.__dict__.pop("avg_doc_length", None)
        self._postings.clear()
        self._doc_lengths.clear()
        self._doc_meta.clear()
        self._closed = True

    @property
    def closed(self) -> bool:
        return self._closed


InvertedIndex = Index


def _extract_title(doc: Document) -> str:
    for raw_line in doc.text.splitlines():
        line = raw_line.strip().lstrip("#").strip()
        if line:
            return line[:120]
    return Path(str(doc.doc_id)).stem or str(doc.doc_id)


def build_index_from_docs(
    docs: Iterable[Document],
    *,
    include_positions: bool = False,
) -> Index:
    """Build an Index in a single streaming pass over documents."""
    postings: defaultdict[str, list[Posting]] = defaultdict(list)
    doc_lengths: dict[int, int] = {}
    doc_meta: dict[int, DocMeta] = {}

    for doc_id, doc in enumerate(docs):
        doc_meta[doc_id] = DocMeta(
            doc_id=doc_id,
            path=str(doc.path),
            title=_extract_title(doc),
            source_id=str(doc.doc_id),
        )

        if include_positions:
            term_positions: defaultdict[str, list[int]] = defaultdict(list)
            token_count = 0
            for pos, token in enumerate(tokenize(doc.text)):
                term_positions[token].append(pos)
                token_count += 1

            doc_lengths[doc_id] = token_count
            for term, pos_list in term_positions.items():
                postings[term].append(
                    Posting(doc_id=doc_id, tf=len(pos_list), positions=tuple(pos_list))
                )
        else:
            term_counts = Counter(tokenize(doc.text))
            doc_lengths[doc_id] = sum(term_counts.values())
            for term, tf in term_counts.items():
                postings[term].append(Posting(doc_id=doc_id, tf=tf))

    return Index(
        postings=dict(postings),
        doc_lengths=doc_lengths,
        doc_meta=doc_meta,
        include_positions=include_positions,
    )


def open_index(path: Path | str, fmt: str | None = None):
    """Load an index as a context manager."""
    from findex.store import open_index as _open_index

    return _open_index(path, fmt=fmt)


@timed
def build_index(
    source: Path | str | Iterable[Document],
    *,
    limit: int | None = None,
    include_positions: bool = False,
    with_positions: bool | None = None,
) -> Index:
    """Stream documents from a path or iterable and build an Index in one pass."""
    if with_positions is not None:
        include_positions = with_positions

    if isinstance(source, (str, Path)):
        docs: Iterable[Document] = iter_documents(Path(source))
    else:
        docs = source

    if limit is not None:
        docs = itertools.islice(docs, limit)

    return build_index_from_docs(docs, include_positions=include_positions)


def main(argv: list[str] | None = None) -> None:
    from findex.store import save

    parser = argparse.ArgumentParser(
        prog="python -m findex.index",
        description="Build and persist an inverted index from a document corpus.",
    )
    parser.add_argument("corpus", type=Path, help="Path to corpus directory or file")
    parser.add_argument("--out", type=Path, default=Path("index.bin"))
    parser.add_argument(
        "--format",
        choices=("pickle", "json", "binary"),
        default=None,
    )
    parser.add_argument(
        "--positions",
        action="store_true",
        help="Store token positions for phrase queries and snippets",
    )
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    index = build_index(args.corpus, limit=args.limit, include_positions=args.positions)
    save(index, args.out, fmt=args.format)
    print(f"Built {index!r} -> {args.out} ({args.out.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
