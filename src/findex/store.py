import json
import pickle
import struct
from array import array
from pathlib import Path

from findex.index import DocMeta, InvertedIndex, Posting

MAGIC = b"FIDXBIN1"





def save(index: InvertedIndex, path: Path, fmt: str | None = None) -> None:

    """Зберігає інвертований індекс на диск у форматі 'binary' або 'pickle'."""

    chosen_fmt = fmt or ("pickle" if path.suffix.lower() in {".pkl", ".pickle"} else "binary")

    if chosen_fmt == "pickle":

        _save_pickle(index, path)

    elif chosen_fmt == "binary":

        _save_binary(index, path)

    else:

        raise ValueError(f"Невідомий формат збереження: {chosen_fmt}")





def load(path: Path, fmt: str | None = None) -> InvertedIndex:

    """Завантажує інвертований індекс із диска."""

    if fmt == "pickle":

        return _load_pickle(path)

    if fmt == "binary":

        return _load_binary(path)



    with path.open("rb") as f:

        magic = f.read(len(MAGIC))

    if magic == MAGIC:

        return _load_binary(path)

    return _load_pickle(path)





def _save_pickle(index: InvertedIndex, path: Path) -> None:

    with path.open("wb") as f:

        pickle.dump(index, f, protocol=pickle.HIGHEST_PROTOCOL)





def _load_pickle(path: Path) -> InvertedIndex:

    # УВАГА ЩОДО БЕЗПЕКИ: модуль pickle під час десеріалізації (pickle.load) здатний

    # виконувати довільний Python-код, закладений у потік байтів (через метод __reduce__).

    # Цей формат придатний виключно для локальних індексів, створених вами особисто.

    # Ніколи не завантажуйте через pickle файли з мережі чи від ненадійних джерел.

    with path.open("rb") as f:

        data = pickle.load(f)

    if not isinstance(data, InvertedIndex):

        raise TypeError("Файл не містить об'єкт InvertedIndex")

    return data





def _save_binary(index: InvertedIndex, path: Path) -> None:

    has_positions = any(

        bool(posting.positions)

        for plist in index.postings.values()

        for posting in plist[:1]

    )



    buf = array("I")

    terms_meta: dict[str, list[int]] = {}



    for term, plist in index.postings.items():

        start_idx = len(buf)

        if has_positions:

            for p in plist:

                buf.append(p.doc_id)

                buf.append(p.tf)

                buf.extend(p.positions)

        else:

            for p in plist:

                buf.append(p.doc_id)

                buf.append(p.tf)

        terms_meta[term] = [start_idx, len(plist)]



    header_obj = {

        "has_positions": has_positions,

        "doc_meta": [

            [m.doc_id, m.source_id, m.path, m.title] for m in index.doc_meta.values()

        ],

        "doc_lengths": list(index.doc_lengths.items()),

        "terms": terms_meta,

    }

    header_bytes = json.dumps(header_obj, ensure_ascii=False, separators=(",", ":")).encode(

        "utf-8"

    )



    with path.open("wb") as f:

        f.write(MAGIC)

        f.write(struct.pack("<Q", len(header_bytes)))

        f.write(header_bytes)

        buf.tofile(f)





def _load_binary(path: Path) -> InvertedIndex:

    raw = path.read_bytes()

    if not raw.startswith(MAGIC):

        raise ValueError(f"Невірний заголовок бінарного індексу у файлі {path}")



    offset = len(MAGIC)

    (header_len,) = struct.unpack_from("<Q", raw, offset)

    offset += 8



    header_obj = json.loads(raw[offset : offset + header_len].decode("utf-8"))

    offset += header_len



    buf = array("I")

    buf.frombytes(memoryview(raw)[offset:])



    has_positions: bool = header_obj["has_positions"]

    doc_meta = {

        int(row[0]): DocMeta(

            doc_id=int(row[0]),

            source_id=str(row[1]),

            path=str(row[2]),

            title=str(row[3]),

        )

        for row in header_obj["doc_meta"]

    }

    doc_lengths = {int(k): int(v) for k, v in header_obj["doc_lengths"]}



    postings: dict[str, list[Posting]] = {}

    for term, (start_idx, count) in header_obj["terms"].items():

        plist: list[Posting] = []

        cursor = start_idx

        if has_positions:

            for _ in range(count):

                doc_id = buf[cursor]

                tf = buf[cursor + 1]

                pos_start = cursor + 2

                pos_end = pos_start + tf

                plist.append(

                    Posting(

                        doc_id=doc_id,

                        tf=tf,

                        positions=tuple(buf[pos_start:pos_end]),

                    )

                )

                cursor = pos_end

        else:

            for _ in range(count):

                plist.append(Posting(doc_id=buf[cursor], tf=buf[cursor + 1]))

                cursor += 2

        postings[term] = plist



    return InvertedIndex(postings=postings, doc_lengths=doc_lengths, doc_meta=doc_meta)