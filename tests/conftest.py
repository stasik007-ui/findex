from pathlib import Path

import pytest
from hypothesis import HealthCheck, settings

from findex.index import Index, build_index
from findex.store import save

settings.register_profile(
    "project",
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
settings.load_profile("project")


@pytest.fixture
def sample_corpus_dir(tmp_path: Path) -> Path:
    documents = {
        "d0.txt": "Python generators are lazy and efficient",
        "d1.txt": "Search uses an inverted index",
        "d2.txt": "Boolean queries combine terms",
        "d3.txt": "Event loop and async Python programming",
    }
    for name, text in documents.items():
        (tmp_path / name).write_text(text, encoding="utf-8")
    return tmp_path


@pytest.fixture
def sample_index(sample_corpus_dir: Path) -> Index:
    return build_index(sample_corpus_dir, with_positions=True)


@pytest.fixture
def saved_index_path(tmp_path: Path, sample_index: Index) -> Path:
    path = tmp_path / "sample.bin"
    save(sample_index, path, fmt="binary")
    return path
