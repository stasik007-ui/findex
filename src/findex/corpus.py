from collections.abc import Iterator
from dataclasses import dataclass
import json
import logging
from pathlib import Path
from typing import cast

log = logging.getLogger(__name__)

SUPPORTED_TEXT_EXTS: frozenset[str] = frozenset({".txt", ".md", ".rst"})


@dataclass(frozen=True, slots=True)
class Document:
    doc_id: str
    path: Path
    text: str


def iter_documents(root: Path) -> Iterator[Document]:
    """Lazily yield Document objects from a directory or a single corpus file."""
    if not root.exists():
        raise FileNotFoundError(f"Corpus path does not exist: {root}")

    if root.is_file():
        yield from _iter_single_file(root, root.parent)
        return

    for file_path in sorted(p for p in root.rglob("*") if p.is_file()):
        yield from _iter_single_file(file_path, root)


def _iter_single_file(file_path: Path, base_dir: Path) -> Iterator[Document]:
    suffix = file_path.suffix.lower()
    rel_id = file_path.relative_to(base_dir).as_posix()

    if suffix == ".jsonl":
        try:
            with file_path.open("r", encoding="utf-8") as fh:
                for line_no, raw_line in enumerate(fh, start=1):
                    line = raw_line.strip()
                    if not line:
                        continue
                    try:
                        parsed: object = json.loads(line)
                    except json.JSONDecodeError:
                        log.warning("Skipping malformed JSON at %s:%d", file_path, line_no)
                        continue
                    if not isinstance(parsed, dict):
                        log.warning("Skipping non-object JSONL at %s:%d", file_path, line_no)
                        continue
                    record = cast(dict[str, object], parsed)
                    text_val = record.get("text")
                    if not isinstance(text_val, str):
                        log.warning("Missing string 'text' field at %s:%d", file_path, line_no)
                        continue
                    id_val = record.get("id")
                    doc_key = id_val if isinstance(id_val, str) else f"{rel_id}:{line_no}"
                    yield Document(doc_id=doc_key, path=file_path, text=text_val)
        except UnicodeDecodeError:
            log.warning("Skipping undecodable JSONL file: %s", file_path)
        return

    if suffix in SUPPORTED_TEXT_EXTS:
        try:
            text = file_path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            log.warning("Skipping undecodable text file: %s", file_path)
            return
        yield Document(doc_id=rel_id, path=file_path, text=text)