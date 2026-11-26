from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Annotated, Literal

import typer
from rich.console import Console
from rich.progress import Progress
from rich.table import Table

from findex.index import build_index
from findex.ranking import BM25, TfIdf
from findex.search import search
from findex.stats import compute_index_stats
from findex.store import open_index, save

app = typer.Typer(help="A small full-text search engine.", no_args_is_help=False)
console = Console()
log = logging.getLogger(__name__)
VERSION = "0.4.0"


@app.callback(invoke_without_command=True)
def main(
    ctx: typer.Context,
    verbose: Annotated[
        int,
        typer.Option("-v", "--verbose", count=True, help="Increase log verbosity."),
    ] = 0,
    version: Annotated[
        bool,
        typer.Option("--version", is_eager=True, help="Show the installed version."),
    ] = False,
) -> None:
    if version:
        typer.echo(f"findex {VERSION}")
        raise typer.Exit()
    if ctx.invoked_subcommand is None:
        typer.echo(ctx.get_help())
        raise typer.Exit()
    level = logging.WARNING if verbose == 0 else logging.INFO if verbose == 1 else logging.DEBUG
    logging.basicConfig(level=level, format="%(levelname)s: %(message)s")


@app.command("index")
def index_command(
    corpus: Annotated[Path, typer.Argument(help="Corpus directory or file")],
    out: Annotated[Path, typer.Option("--out", "-o", help="Output index path")] = Path("index.bin"),
    positions: Annotated[
        bool, typer.Option("--positions", help="Store positions for phrases and snippets")
    ] = False,
    limit: Annotated[int | None, typer.Option("--limit", min=1)] = None,
) -> None:
    """Build and save an inverted index."""
    try:
        with Progress() as progress:
            task = progress.add_task("Indexing...", total=None)
            built = build_index(corpus, limit=limit, with_positions=positions)
            progress.update(task, completed=1)
        save(built, out)
        console.print(f"Indexed {built.num_docs} documents and {len(built)} terms -> {out}")
    except (OSError, ValueError) as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(code=1) from exc


@app.command("search")
def search_command(
    index_path: Annotated[Path, typer.Argument(help="Saved index path")],
    query: Annotated[str, typer.Argument(help="Search query")],
    k: Annotated[int, typer.Option("--k", min=1)] = 10,
    scorer: Annotated[Literal["bm25", "tfidf"], typer.Option("--scorer")] = "bm25",
    json_output: Annotated[bool, typer.Option("--json", help="Output JSON lines")] = False,
) -> None:
    """Search an index using BM25 or TF-IDF ranking."""
    try:
        with open_index(index_path) as loaded:
            model = BM25() if scorer == "bm25" else TfIdf()
            results = search(loaded, query, scorer=model, k=k)
            if json_output:
                for result in results:
                    typer.echo(
                        json.dumps(
                            {
                                "doc_id": result.doc_id,
                                "score": result.score,
                                "title": result.title,
                                "snippet": result.snippet,
                            },
                            ensure_ascii=False,
                        )
                    )
                return
            table = Table("Doc", "Score", "Title", "Snippet")
            for result in results:
                table.add_row(
                    str(result.doc_id),
                    f"{result.score:.4f}",
                    result.title,
                    result.snippet,
                )

    except (OSError, ValueError, SyntaxError) as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(code=1) from exc


@app.command("stats")
def stats_command(
    index_path: Annotated[Path, typer.Argument(help="Saved index path")],
) -> None:
    """Display statistics for a saved index."""
    try:
        with open_index(index_path) as loaded:
            stats = compute_index_stats(loaded)
            console.print(f"Documents: {stats.num_docs}")
            console.print(f"Unique terms: {stats.unique_terms}")
            console.print(f"Total tokens: {stats.total_tokens}")
            console.print(f"Average document length: {stats.avg_doc_length:.2f}")
            console.print(f"Hapax terms: {stats.hapax_count} ({stats.hapax_ratio:.2%})")
            for term, df, tf in stats.top_terms:
                console.print(f"{term}: df={df}, tf={tf}")
    except (OSError, ValueError) as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(code=1) from exc


if __name__ == "__main__":
    app()
