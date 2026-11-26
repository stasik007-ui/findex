from pathlib import Path

from typer.testing import CliRunner

from findex.cli import app

runner = CliRunner()


def test_help() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "full-text search" in result.stdout.lower()


def test_index_search_json_and_stats(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "python.txt").write_text("Python search engine", encoding="utf-8")
    (corpus / "other.txt").write_text("Other document", encoding="utf-8")
    index_path = tmp_path / "index.bin"

    indexed = runner.invoke(
        app,
        ["index", str(corpus), "--out", str(index_path), "--positions"],
    )
    assert indexed.exit_code == 0
    assert index_path.exists()

    searched = runner.invoke(
        app,
        ["search", str(index_path), "python", "--json", "--k", "1"],
    )
    assert searched.exit_code == 0
    assert "Python search engine" in searched.stdout
    assert '"doc_id": 1' in searched.stdout

    stats = runner.invoke(app, ["stats", str(index_path)])
    assert stats.exit_code == 0
    assert "Documents:" in stats.stdout
    assert "Unique terms:" in stats.stdout


def test_cli_reports_expected_errors(tmp_path: Path) -> None:
    result = runner.invoke(app, ["search", str(tmp_path / "missing.bin"), "python"])
    assert result.exit_code != 0
    assert "error:" in result.stderr.lower()
