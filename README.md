# findex — Потоковий індексатор корпусу (Lab 01)

## Порівняння Eager vs. Lazy

| Version | Documents | Peak memory | Elapsed |
| :--- | :---: | :---: | :---: |
| eager (lists) | 1,000 | 289.05 MiB | 8.987 s |
| lazy (generators) | 1,000 | 0.90 MiB | 11.430 s |

### Пояснення різниці у споживанні пам'яті

У жадібній (`eager`) реалізації пікове використання оперативної пам'яті зростає пропорційно до розміру всього корпусу: спочатку список `docs` одночасно утримує в RAM тексти всіх 1 000 документів, а потім `tokenized_docs` створює в пам'яті вкладені списки з усіма окремими токенами ще до початку підрахунку частот. У лінивій (`lazy`) версії генератори `iter_documents` і `tokenize` передають по одному документу та по одному токену за раз: після обробки документа його текст і проміжні токени одразу звільняються, тому в пам'яті залишається лише поточний документ і словник частот `Counter`, розмір якого залежить тільки від кількості унікальних слів. 

---

## Lab 02: Inverted Index, Serialization & Memory Layout

### Порівняння форматів збереження індексу (M3)

| Format | File size | Save time | Load time |
| :--- | :---: | :---: | :---: |
| `pickle` | *0.31 MiB* | *21.13 ms* | *19.12 ms* |
| `binary` (`array('I')` + JSON) | *0.19 MiB* | *8.86 ms* | *12.94 ms* |

### Порівняння представлення постінгів у пам'яті (M4)

| Postings representation | Peak memory (build) | Index file size | Load time |
| :--- | :---: | :---: | :---: |
| `list[Posting]` with plain `@dataclass` | *2.37 MiB* | *0.28 MiB* | *6.27 ms* |
| `list[Posting]` with `slots=True` | *1.74 MiB* | *0.32 MiB* | *5.73 ms* |
| `array('I')` pairs `(doc_ids, tfs)` — no objects | *1.19 MiB* | *0.15 MiB* | *1.46 ms* |

### Куди витрачаються байти в кожному варіанті

У звичайному `@dataclass` кожен екземпляр `Posting` створює в купі власний словник атрибутів `__dict__`, що додає понад 100 байтів накладних витрат на кожен запис. У `@dataclass(slots=True)` об'єкт не має `__dict__` і зберігає лише вказівники на поля у фіксованих C-слотах структури, зменшуючи розмір самого об'єкта у 2–3 рази, проте кожне поле `int` усе одно залишається окремим 28-байтовим об'єктом Python, а список зберігає 8-байтові посилання на них. У варіанті з `array('I')` об'єкти `Posting` та окремі об'єкти `int` узагалі не створюються: дані лежать у неперервному C-буфері по 4 байти на число без заголовків об'єктів і лічильників посилань, що дає максимальну економію пам'яті.

---

## Lab 03 — Ranking, Query Language & Object Model

Lab 3 transforms the inverted index into a Python data-model object and adds ranked search, a query language, snippets, caching, and evaluation helpers.

### Index API and Python protocols

`Index` implements `collections.abc.Mapping[str, list[Posting]]`:

```python
len(index)              # number of indexed terms
"python" in index      # membership test
index["python"]         # postings, or KeyError
list(index)             # iterate terms
repr(index)             # Index(terms=..., docs=...)
```

Additional API: `num_docs`, cached `avg_doc_length`, `doc_length(doc_id)`, and `df(term)`. Positional postings are enabled with `build_index(..., with_positions=True)` and are used for phrase queries and snippets. Persistent indexes can be loaded safely with:

```python
with open_index("index.bin") as index:
    results = search(index, '"event loop"', scorer=BM25(), k=10)
```

`Index.close()` releases internal data, including when an exception is raised inside the context manager.

### Ranking

The `Scorer` protocol allows the ranking algorithm to be exchanged without changing search:

- `TfIdf`: sublinear term frequency with smoothed inverse document frequency;
- `BM25(k1=1.5, b=0.75)`: term-frequency saturation and document-length normalization.

Search accumulates scores per document and selects the top results with `heapq.nlargest`. Results are ordered `SearchResult` objects returned in a `ResultList`.

The controlled BM25 sanity run produced:

```text
rare-term score:       3.8275429492203417
saturation ratio:      1.1971218206157963
short-document score:  2.38200086965308
```

These values demonstrate that the rare term wins, repeated terms saturate, and the short matching document outranks the long matching document.

### Query language

The recursive-descent parser supports terms, implicit and explicit `AND`, `OR`, `NOT`, parentheses, and quoted phrases:

```python
parse('python AND (async OR await) NOT java "event loop"')
```

Query nodes are dataclasses with structural equality and support composition through `&`, `|`, and `~`:

```python
query = Term("python") & (Term("async") | Term("await")) & ~Term("java")
```

### Decorators, caching, and snippets

`build_index`, `load`, and `search` use the `@timed` decorator implemented with `functools.wraps`. Ranked query IDs are cached with `functools.lru_cache(maxsize=256)`. Cache statistics are available through `search_cache_info()` and `search_cache_hits()`.

Snippets use positional postings when available and highlight matching terms in Markdown form (`**term**`).

### Precision@5 evaluation

The ten-document benchmark in `evaluate.py` contains ten labeled queries. The measured mean precision was:

| Scorer | Mean Precision@5 |
| :--- | ---: |
| TF-IDF | 0.9333 |
| BM25 | 0.9333 |

Run the evaluation with:

```powershell
uv run python -c "from findex.evaluate import compute_precision_at_5; print(compute_precision_at_5())"
```

### Verification

The Lab 3 test module passes:

```text
uv run pytest tests/test_lab3.py -q
... [100%]
```

---

## Lab 04 — Typing, Testing, Packaging & CLI

Lab 4 turns `findex` into an installable, typed command-line tool.

[![CI](https://github.com/stasik007-ui/findex/actions/workflows/ci.yml/badge.svg)](https://github.com/stasik007-ui/findex/actions/workflows/ci.yml)

### CLI

The package exposes the `findex` command through the `findex.cli:app` entry point:

```powershell
findex --help
findex --version
findex index <corpus> --out <file> --positions
findex search <index> "python AND search" --scorer bm25 --k 10
findex search <index> "python" --json
findex stats <index>
```

The `index` command builds an index with an optional progress display. The
`search` command supports BM25 and TF-IDF, Rich tables, snippets, and JSON
Lines output. The `stats` command displays document, term, token, and hapax
statistics. Expected errors are reported to stderr with a non-zero exit code.

The root `-v` and `-vv` options configure logging at INFO and DEBUG levels.

### Type checking and quality gates

The project uses strict Pyright, Ruff linting, and Ruff formatting:

```powershell
uv run pyright src tests
uv run ruff check src tests
uv run ruff format --check src tests
```

All three checks pass for the current Lab 4 implementation.

### Tests and coverage

Shared fixtures are defined in `tests/conftest.py`. The suite covers index
protocols, serialization, Boolean and phrase queries, ranking, snippets,
Hypothesis properties, the CLI, and error handling.

```powershell
uv run pytest -q
uv run pytest --cov=findex --cov-report=term-missing
```

Current result: **40 tests passed** and **69% total coverage**. The remaining
uncovered code is primarily optional benchmark, corpus error, and CLI display
branches.

### Packaging

The project uses the `src/` layout, PEP 621 metadata, a locked `uv`
environment, and a `findex` console-script entry point. Build and verify the
artifacts with:

```powershell
uv sync --link-mode=copy
uv build
uvx --from .\dist\findex-0.4.0-py3-none-any.whl findex --help
```

The generated artifacts are `dist/findex-0.4.0.tar.gz` and
`dist/findex-0.4.0-py3-none-any.whl`. IDE, virtual-environment, test-cache,
and build directories are excluded from source distributions and Git.

### Continuous integration

`.github/workflows/ci.yml` runs on every push and pull request:

1. `uv sync --locked`;
2. `ruff check`;
3. `ruff format --check`;
4. `pyright`;
5. `pytest` with coverage.