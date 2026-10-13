"""
RAGnarok Developer CLI Admin Deck — Top-Level Commands.

Built with Typer and Rich. Provides concise operational commands with shorthands:
  rag query "<text>"  (q)   — Full retrieval and streaming synthesis
  rag chat            (c)   — Interactive multi-turn developer shell
  rag ingest <path>   (in)  — Document ingestion with progress bars
  rag reindex         (rx)  — Rebuild BM25 and TurboVec indexes from PostgreSQL
  rag stats           (st)  — Multi-panel system status dashboard
  rag doctor          (doc) — Environment, database, and hardware diagnostics
  rag test <target>   (t)   — Modular component probes and test suites
"""

from __future__ import annotations

import os
import re
import time
from pathlib import Path
from typing import Any

import structlog
import typer
from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)

from ragnarok.cli.probes import test_app
from ragnarok.core.diagnostics import (
    check_accelerator,
    check_configuration,
    check_critical_imports,
    check_database,
    check_filesystem_paths,
    check_python_version,
)

logger = structlog.get_logger(__name__)

app = typer.Typer(
    name="rag",
    help="RAGnarok Developer CLI Admin Deck — Unified command interface for retrieval, "
    "ingestion, diagnostics, and component probing.",
    no_args_is_help=True,
    rich_markup_mode="rich",
)
app.add_typer(test_app, name="test", rich_help_panel="Component Probes")
app.add_typer(test_app, name="t", hidden=True)
console = Console()


# ---------------------------------------------------------------------------
# rag query "<text>" (alias: q)
# ---------------------------------------------------------------------------
@app.command(name="query", rich_help_panel="Pipeline Operations")
def query_command(
    text: str = typer.Argument(..., help="Query text for retrieval and synthesis."),
    strategy: str = typer.Option(
        "hybrid", "--strategy", "-s", help="Retrieval strategy: lexical, dense, hybrid."
    ),
    no_rerank: bool = typer.Option(False, "--no-rerank", help="Disable Cross-Encoder reranking."),
    budget: int = typer.Option(0, "--budget", "-b", help="Override context token budget (0=default)."),
    show_waterfall: bool = typer.Option(False, "--waterfall", "-w", help="Show latency waterfall."),
    show_provenance: bool = typer.Option(
        False, "--provenance", "-p", help="Show retrieval provenance table."
    ),
) -> None:
    """
    Execute full retrieval and live streaming synthesis for a query.

    Alias: `rag q`
    """
    console.rule("[bold cyan]RAGnarok Query Pipeline[/bold cyan]")
    console.print(f"[dim]Query    : {text}[/dim]")
    console.print(f"[dim]Strategy : {strategy} | Rerank: {'off' if no_rerank else 'on'}[/dim]")

    try:
        from ragnarok.core.config import load_config
        from ragnarok.generation.service import GenerationService
        from ragnarok.retrieval.pipeline import retrieve_and_assemble_context

        config = load_config()
        if budget > 0:
            config.context_management.max_tokens = budget

        start_time = time.perf_counter()

        # 1. Retrieve candidates and expand parent context
        context_payload, candidates, stage_latencies = retrieve_and_assemble_context(
            query=text,
            config=config,
            strategy=strategy,
            rerank=not no_rerank,
            budget=budget,
        )

        service = GenerationService.from_config(config=config, console=console)

        # 2. Execute grounded streaming generation
        response = service.generate_sync(
            query=text,
            context=context_payload,
            render_live=True,
        )
        total_elapsed = (time.perf_counter() - start_time) * 1000.0

        # Render formatted output
        console.print(service.format_terminal_output(response))

        # Optional provenance display
        if show_provenance:
            from ragnarok.cli.visuals import render_provenance_table
            render_provenance_table(candidates, console=console)

        # Optional waterfall display
        if show_waterfall:
            from ragnarok.cli.visuals import render_latency_waterfall

            render_latency_waterfall(
                {
                    "Retrieval": stage_latencies["retrieval_ms"],
                    "Reranking": stage_latencies["rerank_ms"],
                    "Context Expansion": stage_latencies["context_ms"],
                    "Generation TTFT": response.latency_metrics.ttft_ms,
                    "Stream Synthesis": (
                        response.latency_metrics.total_duration_ms
                        - response.latency_metrics.ttft_ms
                    ),
                    "CLI Overhead": max(
                        0,
                        total_elapsed
                        - stage_latencies["retrieval_ms"]
                        - stage_latencies["rerank_ms"]
                        - stage_latencies["context_ms"]
                        - response.latency_metrics.total_duration_ms,
                    ),
                },
                console=console,
            )

    except KeyboardInterrupt:
        console.print("\n[dim]Query interrupted.[/dim]")
    except Exception as e:
        console.print(f"[bold red]Query failed: {e}[/bold red]")
        logger.error("Query command failed", error=str(e))
        raise typer.Exit(code=1) from e


# Register 'q' as shorthand alias for 'query'
@app.command(name="q", hidden=True)
def query_alias(
    text: str = typer.Argument(..., help="Query text."),
    strategy: str = typer.Option("hybrid", "--strategy", "-s"),
    no_rerank: bool = typer.Option(False, "--no-rerank"),
    budget: int = typer.Option(0, "--budget", "-b"),
    show_waterfall: bool = typer.Option(False, "--waterfall", "-w"),
    show_provenance: bool = typer.Option(False, "--provenance", "-p"),
) -> None:
    """Shorthand for `rag query`."""
    query_command(text, strategy, no_rerank, budget, show_waterfall, show_provenance)


# ---------------------------------------------------------------------------
# rag chat (alias: c)
# ---------------------------------------------------------------------------
@app.command(name="chat", rich_help_panel="Pipeline Operations")
def chat_command() -> None:
    """
    Launch interactive multi-turn developer REPL shell with live streaming.

    Alias: `rag c`
    """
    from ragnarok.cli.repl import run_repl

    run_repl(console=console)


@app.command(name="c", hidden=True)
def chat_alias() -> None:
    """Shorthand for `rag chat`."""
    chat_command()


# ---------------------------------------------------------------------------
# rag ingest <path> (alias: in)
# ---------------------------------------------------------------------------
@app.command(name="ingest", rich_help_panel="Pipeline Operations")
def ingest_command(
    path: Path = typer.Argument(..., help="File or directory path to ingest."),
    batch_size: int = typer.Option(0, "--batch-size", "-n", help="Override batch size (0=default)."),
) -> None:
    """
    Ingest a single document or directory of documents with deduplication and progress bars.

    Alias: `rag in`
    """
    console.rule("[bold cyan]RAGnarok Document Ingestion[/bold cyan]")

    if not path.exists():
        console.print(f"[bold red]Path not found: {path}[/bold red]")
        raise typer.Exit(code=1)

    try:
        from ragnarok.core.config import load_config

        config = load_config()
        if batch_size > 0:
            config.ingestion.batch_size = batch_size

        # Collect files to ingest (filtering out internal UUID vault files and metadata artifacts)
        uuid_pattern = re.compile(
            r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\."
        )

        def _is_user_source_file(f: Path) -> bool:
            if f.name.startswith(".") or "Zone.Identifier" in f.name:
                return False
            return not bool(uuid_pattern.match(f.name))

        files: list[Path] = []
        if path.is_file():
            if _is_user_source_file(path):
                files.append(path)
        elif path.is_dir():
            for ext in config.ingestion.supported_formats:
                for candidate in path.rglob(f"*{ext}"):
                    if _is_user_source_file(candidate):
                        files.append(candidate)
        else:
            console.print(f"[bold red]Invalid path: {path}[/bold red]")
            raise typer.Exit(code=1)

        if not files:
            console.print(f"[yellow]No supported user files found in: {path}[/yellow]")
            console.print(f"[dim]Supported formats: {', '.join(config.ingestion.supported_formats)}[/dim]")
            raise typer.Exit(code=0)

        console.print(f"[dim]Found {len(files)} files to ingest[/dim]")

        # Process with progress bar
        succeeded = 0
        failed = 0
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("Ingesting documents...", total=len(files))

            from ragnarok.ingestion.pipeline import IngestionPipeline
            from ragnarok.storage.database import DatabasePool
            from ragnarok.storage.repository import RelationalRepository

            db_url = os.getenv("DATABASE_URL", config.secrets.database_url)
            pool = DatabasePool(db_url)
            repo = RelationalRepository(pool)
            pipeline = IngestionPipeline(repository=repo, config=config)

            newly_chunked = False
            for file_path in files:
                try:
                    progress.update(
                        task, description=f"Processing: {file_path.name}"
                    )

                    doc, is_duplicate = pipeline.process_file(file_path)

                    if not is_duplicate:
                        succeeded += 1
                        newly_chunked = True
                    else:
                        console.print(
                            f"  [yellow]Duplicate skipped: {file_path.name}[/yellow]"
                        )
                        succeeded += 1

                except Exception as e:
                    failed += 1
                    console.print(
                        f"  [red]Failed: {file_path.name} — {e}[/red]"
                    )
                finally:
                    progress.advance(task)

            # Stage 5: Build search indexes if any files were chunked or indexes are empty
            bm25_dir = Path("data/indexes/bm25")
            bm25_has_files = bm25_dir.exists() and any(
                f.name != ".gitkeep" for f in bm25_dir.iterdir()
            )
            if newly_chunked or not bm25_has_files:
                try:
                    progress.update(task, description="Building BM25 and TurboVec search indexes...")
                    indexed_count = pipeline.reindex()
                    console.print(f"[dim]Indexed {indexed_count} child chunks into BM25s and TurboVec[/dim]")
                except Exception as e:
                    console.print(f"  [yellow]Indexing warning: {e}[/yellow]")

        # Summary
        console.print()
        console.print(
            f"[bold green]✓ Ingestion complete:[/bold green] "
            f"{succeeded} succeeded, {failed} failed out of {len(files)} files"
        )

    except SystemExit:
        raise
    except Exception as e:
        console.print(f"[bold red]Ingestion failed: {e}[/bold red]")
        logger.error("Ingestion command failed", error=str(e))
        raise typer.Exit(code=1) from e


@app.command(name="in", hidden=True)
def ingest_alias(
    path: Path = typer.Argument(..., help="File or directory path."),
    batch_size: int = typer.Option(0, "--batch-size", "-n"),
) -> None:
    """Shorthand for `rag ingest`."""
    ingest_command(path, batch_size)


# ---------------------------------------------------------------------------
# rag reindex (alias: rx)
# ---------------------------------------------------------------------------
@app.command(name="reindex", rich_help_panel="Pipeline Operations")
def reindex_command() -> None:
    """
    Rebuild BM25 and TurboVec indexes directly from PostgreSQL chunks.

    Alias: `rag rx`
    """
    console.rule("[bold cyan]RAGnarok Index Rebuild[/bold cyan]")
    console.print("[dim]Rebuilding search indexes from PostgreSQL child chunks...[/dim]")

    try:
        from ragnarok.core.config import load_config
        from ragnarok.indexing.reindexer import ReindexService
        from ragnarok.storage.database import DatabasePool
        from ragnarok.storage.repository import RelationalRepository

        config = load_config()
        db_url = os.getenv("DATABASE_URL", config.secrets.database_url)
        pool = DatabasePool(db_url)
        repo = RelationalRepository(pool)
        reindexer = ReindexService(repository=repo, config=config)

        start = time.perf_counter()
        count = reindexer.reindex_all()
        elapsed = (time.perf_counter() - start) * 1000.0

        console.print(
            f"\n[bold green]✓ Indexes rebuilt successfully ({count} chunks)[/bold green] in {elapsed:.0f} ms"
        )

    except Exception as e:
        console.print(f"[bold red]Reindex failed: {e}[/bold red]")
        logger.error("Reindex command failed", error=str(e))
        raise typer.Exit(code=1) from e


@app.command(name="rx", hidden=True)
def reindex_alias() -> None:
    """Shorthand for `rag reindex`."""
    reindex_command()


# ---------------------------------------------------------------------------
# rag stats (alias: st)
# ---------------------------------------------------------------------------
@app.command(name="stats", rich_help_panel="Diagnostics")
def stats_command() -> None:
    """
    Display a Rich dashboard of database records, index sizes, and storage metrics.

    Alias: `rag st`
    """
    console.rule("[bold cyan]RAGnarok System Status[/bold cyan]")

    try:
        from ragnarok.cli.visuals import render_stats_dashboard
        from ragnarok.core.config import load_config

        config = load_config()

        # Query database stats
        db_stats: dict[str, int] = {
            "documents": 0,
            "parent_chunks": 0,
            "chunks": 0,
            "jobs_completed": 0,
            "jobs_failed": 0,
        }

        try:
            import psycopg
            from psycopg import sql

            db_url = os.getenv(
                "DATABASE_URL",
                config.secrets.database_url,
            )
            with psycopg.connect(db_url) as conn, conn.cursor() as cur:
                for table_name in ["documents", "parent_chunks", "chunks"]:
                    cur.execute(
                        sql.SQL("SELECT COUNT(*) FROM {};").format(sql.Identifier(table_name))
                    )
                    row = cur.fetchone()
                    if row:
                        db_stats[table_name] = row[0]

                cur.execute(
                    "SELECT COUNT(*) FROM processing_jobs WHERE stage = 'completed';"
                )
                row = cur.fetchone()
                if row:
                    db_stats["jobs_completed"] = row[0]

                cur.execute(
                    "SELECT COUNT(*) FROM processing_jobs WHERE stage = 'failed';"
                )
                row = cur.fetchone()
                if row:
                    db_stats["jobs_failed"] = row[0]
        except Exception as db_err:
            console.print(f"[yellow]Database stats unavailable: {db_err}[/yellow]")

        # Compute index file sizes
        def dir_size_mb(dir_path: Path) -> float:
            total = 0
            if dir_path.exists():
                for f in dir_path.rglob("*"):
                    if f.is_file():
                        total += f.stat().st_size
            return total / (1024 * 1024)

        bm25_path = Path("data/indexes/bm25")
        dense_path = Path("data/indexes/dense")

        index_stats: dict[str, Any] = {
            "bm25_terms": db_stats.get("chunks", 0),
            "bm25_size_mb": dir_size_mb(bm25_path),
            "dense_vectors": db_stats.get("chunks", 0),
            "dense_dim": config.dense_retrieval.vector_dimension,
            "dense_size_mb": dir_size_mb(dense_path),
            "db_size_mb": 0.0,
        }

        # Attempt to get DB size
        try:
            import psycopg

            db_url = os.getenv("DATABASE_URL", config.secrets.database_url)
            with psycopg.connect(db_url) as conn, conn.cursor() as cur:
                cur.execute("SELECT pg_database_size(current_database());")
                row = cur.fetchone()
                if row:
                    index_stats["db_size_mb"] = row[0] / (1024 * 1024)
        except Exception:
            pass

        render_stats_dashboard(
            db_stats=db_stats,
            index_stats=index_stats,
            console=console,
        )

    except Exception as e:
        console.print(f"[bold red]Stats command failed: {e}[/bold red]")
        logger.error("Stats command failed", error=str(e))
        raise typer.Exit(code=1) from e


@app.command(name="st", hidden=True)
def stats_alias() -> None:
    """Shorthand for `rag stats`."""
    stats_command()


# ---------------------------------------------------------------------------
# rag doctor (alias: doc)
# ---------------------------------------------------------------------------
@app.command(name="doctor", rich_help_panel="Diagnostics")
def doctor_command() -> None:
    """
    Run comprehensive environment, database connectivity, and hardware diagnostics.

    Alias: `rag doc`
    """
    console.rule("[bold cyan]RAGnarok Environment Diagnostic Check[/bold cyan]")

    results = [
        check_python_version(),
        check_critical_imports(),
        check_configuration(),
        check_filesystem_paths(),
        check_database(),
    ]
    check_accelerator()

    console.rule()
    if all(results):
        console.print("[bold green]Setup status: READY TO DEVELOP[/bold green]")
        raise typer.Exit(code=0)
    else:
        console.print("[bold red]Setup status: INCOMPLETE — Resolve failures above[/bold red]")
        raise typer.Exit(code=1)


@app.command(name="doc", hidden=True)
def doctor_alias() -> None:
    """Shorthand for `rag doctor`."""
    doctor_command()



if __name__ == "__main__":
    app()

