"""
Visual Metrics Deck for Terminal Dashboards.

Provides Rich-formatted renderers for:
1. Latency waterfall breakdown chart (Unicode bar display)
2. Context token budget allocation gauge
3. Retrieval candidate provenance table
4. System storage & index status dashboard
"""

from __future__ import annotations

from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text


def render_latency_waterfall(
    stage_timings: dict[str, float],
    console: Console | None = None,
) -> None:
    """
    Renders a Unicode bar chart showing execution time breakdown across pipeline stages.

    Parameters:
        stage_timings: Mapping of stage names to elapsed milliseconds.
        console: Rich Console instance for output rendering.
    """
    target = console if console is not None else Console()
    total_ms = sum(stage_timings.values())

    # Determine max label width for alignment
    max_label = max(len(name) for name in stage_timings) if stage_timings else 20
    bar_width = 20

    target.print()
    target.rule("[bold cyan]Pipeline Stage Latency Breakdown[/bold cyan]")

    for stage_name, elapsed_ms in stage_timings.items():
        pct = (elapsed_ms / total_ms * 100.0) if total_ms > 0 else 0.0
        filled = int(pct / 100.0 * bar_width)
        bar = "█" * filled + " " * (bar_width - filled)

        label = f"{stage_name:<{max_label}}"
        target.print(
            f"  {label} : [{bar}] {elapsed_ms:>7.0f} ms ({pct:>5.1f}%)"
        )

    target.print(f"  {'Total Latency':<{max_label}} : [bold]{total_ms:,.0f} ms[/bold]")
    target.print()


def render_budget_gauge(
    source_tokens: list[dict[str, Any]],
    max_tokens: int,
    console: Console | None = None,
) -> None:
    """
    Renders a Unicode gauge showing context token consumption against generation budget.

    Parameters:
        source_tokens: List of dicts with keys 'label', 'parent_index', 'tokens'.
        max_tokens: Configured context token budget from config.toml.
        console: Rich Console instance for output rendering.
    """
    target = console if console is not None else Console()
    bar_width = 20

    target.print()
    target.rule(f"[bold cyan]Context Budget Allocation (max_tokens = {max_tokens})[/bold cyan]")

    total_used = 0
    omitted_count = 0

    for source in source_tokens:
        label = source.get("label", "Source")
        parent_idx = source.get("parent_index", 0)
        tokens = source.get("tokens", 0)
        total_used += tokens
        pct = (tokens / max_tokens * 100.0) if max_tokens > 0 else 0.0
        filled = int(pct / 100.0 * bar_width)
        bar = "█" * filled + " " * (bar_width - filled)

        target.print(
            f"  [{label}] (Parent {parent_idx}): [{bar}] {tokens:>5,} tokens ({pct:>5.1f}%)"
        )

    # Unused budget
    unused = max(0, max_tokens - total_used)
    unused_pct = (unused / max_tokens * 100.0) if max_tokens > 0 else 0.0
    unused_filled = int(unused_pct / 100.0 * bar_width)
    unused_bar = "█" * unused_filled + " " * (bar_width - unused_filled)
    target.print(
        f"  {'Unused Budget':<20}: [{unused_bar}] {unused:>5,} tokens ({unused_pct:>5.1f}%)"
    )

    utilization = (total_used / max_tokens * 100.0) if max_tokens > 0 else 0.0
    target.print(
        f"  Total Utilized       : {total_used:,} / {max_tokens:,} tokens "
        f"({utilization:.1f}%) | Omitted Parents: {omitted_count}"
    )
    target.print()


def render_provenance_table(
    candidates: list[dict[str, Any]] | list[Any],
    console: Console | None = None,
) -> None:
    """
    Renders a multi-column table showing retrieval candidate provenance:
    final rank, chunk/parent IDs, BM25/Dense ranks+scores, RRF score, rerank score.

    Parameters:
        candidates: List of provenance dicts or ChildResultCandidate model instances.
        console: Rich Console instance for output rendering.
    """
    target = console if console is not None else Console()

    table = Table(
        title="Candidate Retrieval Provenance Table",
        show_header=True,
        header_style="bold magenta",
        border_style="bright_black",
        expand=True,
    )
    table.add_column("Final", justify="right", style="bold cyan", width=6)
    table.add_column("Child ID", justify="left", width=14)
    table.add_column("Parent ID", justify="left", width=14)
    table.add_column("BM25 Rank/Score", justify="center", width=16)
    table.add_column("Dense Rank/Score", justify="center", width=16)
    table.add_column("RRF Score", justify="center", width=10)
    table.add_column("Rerank Score", justify="center", width=12)

    for c in candidates:
        if isinstance(c, dict):
            cand_dict: dict[str, Any] = c
        elif hasattr(c, "model_dump"):
            cand_dict = c.model_dump()
        else:
            cand_dict = vars(c)
        chunk_id = str(cand_dict.get("chunk_id", ""))[:10] + "..."
        parent_id = str(cand_dict.get("parent_chunk_id", ""))[:10] + "..."

        bm25_rank = cand_dict.get("bm25_rank")
        bm25_score = cand_dict.get("bm25_score")
        bm25_str = (
            f"#{bm25_rank} / {bm25_score:.2f}" if bm25_rank and bm25_score is not None else "—"
        )

        dense_rank = cand_dict.get("dense_rank")
        dense_score = cand_dict.get("dense_score")
        dense_str = (
            f"#{dense_rank} / {dense_score:.2f}" if dense_rank and dense_score is not None else "—"
        )

        rrf_score = cand_dict.get("fusion_score")
        rrf_str = f"{rrf_score:.4f}" if rrf_score is not None else "—"

        rerank_score = cand_dict.get("rerank_score")
        rerank_str = f"{rerank_score:.3f}" if rerank_score is not None else "—"

        final_rank = cand_dict.get("final_rank", getattr(c, "final_rank", "?"))
        table.add_row(
            f"#{final_rank}",
            chunk_id,
            parent_id,
            bm25_str,
            dense_str,
            rrf_str,
            rerank_str,
        )

    target.print()
    target.print(table)
    target.print()


def render_stats_dashboard(
    db_stats: dict[str, int],
    index_stats: dict[str, Any],
    console: Console | None = None,
) -> None:
    """
    Renders multi-panel system storage and index status dashboard.

    Parameters:
        db_stats: Dict with keys like 'documents', 'parent_chunks', 'chunks', 'jobs_completed', 'jobs_failed'.
        index_stats: Dict with keys like 'bm25_terms', 'bm25_size_mb', 'dense_vectors', 'dense_dim', 'dense_size_mb', 'db_size_mb'.
        console: Rich Console instance for output rendering.
    """
    target = console if console is not None else Console()

    # PostgreSQL stats panel
    pg_text = Text()
    pg_text.append("PostgreSQL Relational Store:\n", style="bold white")
    pg_text.append(f"• Total Documents Stored     : {db_stats.get('documents', 0)}\n")
    pg_text.append(f"• Total Parent Chunk Contexts : {db_stats.get('parent_chunks', 0)}\n")
    pg_text.append(f"• Total Child Retrieval Chunks: {db_stats.get('chunks', 0)}\n")
    pg_text.append(
        f"• Ingestion Jobs Completed   : {db_stats.get('jobs_completed', 0)} "
        f"({db_stats.get('jobs_failed', 0)} Failed)\n"
    )
    pg_panel = Panel(pg_text, title="[bold cyan]Relational Store[/bold cyan]", border_style="cyan")

    # Index stats panel
    idx_text = Text()
    idx_text.append("Search Index Artifacts:\n", style="bold white")
    idx_text.append(
        f"• BM25s Lexical Index    : {index_stats.get('bm25_terms', 0)} terms | "
        f"Size on disk: {index_stats.get('bm25_size_mb', 0.0):.1f} MB (data/indexes/bm25/)\n"
    )
    idx_text.append(
        f"• TurboVec Dense Index   : {index_stats.get('dense_vectors', 0)} vectors "
        f"(dim={index_stats.get('dense_dim', 0)}) | "
        f"Size on disk: {index_stats.get('dense_size_mb', 0.0):.1f} MB (data/indexes/dense/)\n"
    )
    idx_text.append(
        f"• Relational Database    : {index_stats.get('db_size_mb', 0.0):.1f} MB "
        f"(PostgreSQL)\n"
    )
    idx_panel = Panel(
        idx_text, title="[bold cyan]Search Indexes[/bold cyan]", border_style="cyan"
    )

    target.print()
    target.rule("[bold cyan]RAGnarok System Storage & Index Status[/bold cyan]")
    target.print(pg_panel)
    target.print(idx_panel)
    target.print()
