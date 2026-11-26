import json
import logging
from collections.abc import Iterator
from pathlib import Path
from typing import NamedTuple

logger = logging.getLogger(__name__)





class Document(NamedTuple):

    doc_id: str

    path: Path

    text: str





def iter_documents(root: Path) -> Iterator[Document]:

    """Ліниво віддає документи по одному з директорії .txt або з .jsonl-файлу."""

    if not root.exists():

        raise FileNotFoundError(f"Шлях до корпусу не знайдено: {root}")



    if root.is_file() and root.suffix == ".jsonl":

        yield from _iter_jsonl(root)

        return



    for file_path in root.rglob("*.txt"):

        if not file_path.is_file():

            continue

        try:

            text = file_path.read_text(encoding="utf-8", errors="strict")

        except UnicodeDecodeError as exc:

            logger.warning("Пропущено файл із некоректним UTF-8 (%s): %s", file_path, exc)

            continue

        except OSError as exc:

            logger.warning("Помилка читання файлу (%s): %s", file_path, exc)

            continue



        doc_id = str(file_path.relative_to(root))

        yield Document(doc_id=doc_id, path=file_path, text=text)





def _iter_jsonl(jsonl_path: Path) -> Iterator[Document]:

    """Потокове читання JSON-lines (один рядок = один документ із полем 'text')."""

    with jsonl_path.open(encoding="utf-8", errors="replace") as f:

        for line_num, line in enumerate(f, start=1):

            line = line.strip()

            if not line:

                continue

            try:

                record = json.loads(line)

            except json.JSONDecodeError as exc:

                logger.warning("Невалідний JSON у %s:%d: %s", jsonl_path, line_num, exc)

                continue



            text = record.get("text", "")

            doc_id = str(record.get("id", f"{jsonl_path.name}:{line_num}"))

            yield Document(doc_id=doc_id, path=jsonl_path, text=text)