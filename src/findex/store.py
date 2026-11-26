from __future__ import annotations

import json
import pickle
import struct
from array import array
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path

from findex.index import DocMeta, Index, Posting
from findex.utils import timed

BINARY_MAGIC = b"FIDX0002"


def _infer_format_from_path(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".json":
        return "json"
    if suffix in {".idx", ".fidx"}:
        return "binary"
    return "pickle"


def _detect_format_from_file(path: Path) -> str:
    with path.open("rb") as fh:
        head = fh.read(8)
    if head == BINARY_MAGIC:
        return "binary"
    if head.lstrip().startswith(b"{"):
        return "json"
    return "pickle"


def save(index: Index, path: Path | str, fmt: str | None = None) -> None:
    """Persist an Index to disk in 'pickle', 'json', or 'binary' format."""
    path = Path(path)
    chosen = fmt or _infer_format_from_path(path)

    if chosen == "pickle":
        # SECURITY NOTE: pickle serializes arbitrary Python object graphs.
        # Only load locally generated pickle files; never unpickle untrusted data.
        with path.open("wb") as fh:
            pickle.dump(index, fh, protocol=pickle.HIGHEST_PROTOCOL)
        return

    if chosen == "json":
        payload = {
            "include_positions": index.include_positions,
            "doc_lengths": {str(k): v for k, v in index.doc_lengths.items()},
            "doc_meta": {
                str(k): {
                    "doc_id": m.doc_id,
                    "path": m.path,
                    "title": m.title,
                    "source_id": m.source_id,
                }
                for k, m in index.doc_meta.items()
            },
            "postings": {
                term: [
                    [p.doc_id, p.tf, list(p.positions)]
                    if index.include_positions
                    else [p.doc_id, p.tf]
                    for p in plist
                ]
                for term, plist in index.postings.items()
            },
        }
        path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        return

    if chosen == "binary":
        _save_binary(index, path)
        return

    raise ValueError(f"Unsupported index format: {chosen!r}")


@timed
def load(path: Path | str, fmt: str | None = None) -> Index:
    """Load an Index from disk."""
    path = Path(path)
    chosen = fmt or _detect_format_from_file(path)

    if chosen == "pickle":
        # SECURITY WARNING: pickle.load can execute arbitrary bytecode via __reduce__.
        # Never call this on untrusted files.
        with path.open("rb") as fh:
            loaded = pickle.load(fh)  # noqa: S301
        if not isinstance(loaded, Index):
            raise TypeError(f"Expected Index in {path}, got {type(loaded).__name__}")
        return loaded

    if chosen == "json":
        raw = json.loads(path.read_text(encoding="utf-8"))
        include_positions = bool(raw.get("include_positions", False))
        doc_lengths = {int(k): int(v) for k, v in raw["doc_lengths"].items()}
        doc_meta = {
            int(k): DocMeta(
                doc_id=int(v["doc_id"]),
                path=str(v["path"]),
                title=str(v["title"]),
                source_id=str(v.get("source_id", "")),
            )
            for k, v in raw["doc_meta"].items()
        }
        postings: dict[str, list[Posting]] = {}
        for term, entries in raw["postings"].items():
            if include_positions:
                postings[term] = [
                    Posting(doc_id=int(e[0]), tf=int(e[1]), positions=tuple(e[2])) for e in entries
                ]
            else:
                postings[term] = [Posting(doc_id=int(e[0]), tf=int(e[1])) for e in entries]
        return Index(
            postings=postings,
            doc_lengths=doc_lengths,
            doc_meta=doc_meta,
            include_positions=include_positions,
        )

    if chosen == "binary":
        return _load_binary(path)

    raise ValueError(f"Unsupported index format: {chosen!r}")


@contextmanager
def open_index(path: Path | str, fmt: str | None = None) -> Generator[Index, None, None]:
    """Context manager that loads an Index and guarantees cleanup on exit."""
    index = load(Path(path), fmt=fmt)
    try:
        yield index
    finally:
        index.close()


def _save_binary(index: Index, path: Path) -> None:
    doc_ids_buf = array("I")
    tfs_buf = array("I")
    pos_buf = array("I")
    term_dir: dict[str, tuple[int, int, int, int]] = {}

    for term, plist in index.postings.items():
        start = len(doc_ids_buf)
        pos_start = len(pos_buf)
        for p in plist:
            doc_ids_buf.append(p.doc_id)
            tfs_buf.append(p.tf)
            if index.include_positions:
                pos_buf.extend(p.positions)
        term_dir[term] = (start, len(plist), pos_start, len(pos_buf) - pos_start)

    header_obj = {
        "include_positions": index.include_positions,
        "doc_lengths": {str(k): v for k, v in index.doc_lengths.items()},
        "doc_meta": {
            str(k): [m.doc_id, m.path, m.title, m.source_id] for k, m in index.doc_meta.items()
        },
        "terms": term_dir,
        "total_postings": len(doc_ids_buf),
        "total_positions": len(pos_buf),
    }
    header_bytes = json.dumps(header_obj, ensure_ascii=False).encode("utf-8")

    with path.open("wb") as fh:
        fh.write(BINARY_MAGIC)
        fh.write(struct.pack("<Q", len(header_bytes)))
        fh.write(header_bytes)
        doc_ids_buf.tofile(fh)
        tfs_buf.tofile(fh)
        if index.include_positions:
            pos_buf.tofile(fh)


def _load_binary(path: Path) -> Index:
    with path.open("rb") as fh:
        magic = fh.read(8)
        if magic != BINARY_MAGIC:
            raise ValueError(f"Invalid binary index magic in {path}: {magic!r}")
        (header_len,) = struct.unpack("<Q", fh.read(8))
        header = json.loads(fh.read(header_len).decode("utf-8"))

        total_postings = int(header["total_postings"])
        total_positions = int(header["total_positions"])
        include_positions = bool(header["include_positions"])

        doc_ids_buf = array("I")
        doc_ids_buf.fromfile(fh, total_postings)
        tfs_buf = array("I")
        tfs_buf.fromfile(fh, total_postings)
        pos_buf = array("I")
        if include_positions and total_positions:
            pos_buf.fromfile(fh, total_positions)

    doc_lengths = {int(k): int(v) for k, v in header["doc_lengths"].items()}
    doc_meta = {
        int(k): DocMeta(
            doc_id=int(v[0]),
            path=str(v[1]),
            title=str(v[2]),
            source_id=str(v[3]) if len(v) > 3 else "",
        )
        for k, v in header["doc_meta"].items()
    }

    postings: dict[str, list[Posting]] = {}
    for term, (start, length, pos_start, _) in header["terms"].items():
        plist: list[Posting] = []
        cursor = pos_start
        for i in range(start, start + length):
            tf = tfs_buf[i]
            if include_positions:
                positions = tuple(pos_buf[cursor : cursor + tf])
                cursor += tf
                plist.append(Posting(doc_id=doc_ids_buf[i], tf=tf, positions=positions))
            else:
                plist.append(Posting(doc_id=doc_ids_buf[i], tf=tf))
        postings[term] = plist

    return Index(
        postings=postings,
        doc_lengths=doc_lengths,
        doc_meta=doc_meta,
        include_positions=include_positions,
    )
