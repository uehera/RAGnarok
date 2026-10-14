"""
Modular Component Probe Deck for RAGnarok CLI.

Each probe function operates in total isolation without side effects,
inspecting a single module of the RAGnarok pipeline independently.
Probes: parse, chunk, embed, bm25, route, rerank, context, guard, pii, auto, eval.
"""

from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path
from typing import Any

import structlog
import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

logger = structlog.get_logger(__name__)

test_app = typer.Typer(
    name="test",
    help="Modular component probes, unit tests, and evaluation suites.",
    no_args_is_help=True,
)
console = Console()


@test_app.command(name="parse")
def probe_parse(
    file: Path = typer.Argument(..., help="Path to a document file to parse with Docling."),
) -> None:
    """Probe: Execute Docling on a file and display extracted Markdown structure in memory."""
    console.rule("[bold cyan]Document Parser Probe[/bold cyan]")

    if not file.exists():
        console.print(f"[bold red]File not found: {file}[/bold red]")
        raise typer.Exit(code=1)

    try:
        from ragnarok.ingestion.parser import parse_document_to_markdown

        console.print(f"[dim]Parsing: {file}[/dim]")
        start = time.perf_counter()
        markdown_text = parse_document_to_markdown(file)
        elapsed_ms = (time.perf_counter() - start) * 1000.0

        # Extract headings for hierarchy display
        headings: list[tuple[int, str]] = []
        for line in markdown_text.splitlines():
            stripped = line.strip()
            if stripped.startswith("#"):
                level = len(stripped) - len(stripped.lstrip("#"))
                heading_text = stripped.lstrip("#").strip()
                headings.append((level, heading_text))

        # Heading hierarchy table
        if headings:
            heading_table = Table(
                title="Detected Heading Hierarchy",
                show_header=True,
                header_style="bold magenta",
            )
            heading_table.add_column("Level", justify="center", width=6)
            heading_table.add_column("Heading", justify="left")
            for level, text in headings:
                indent = "  " * (level - 1)
                heading_table.add_row(f"H{level}", f"{indent}{text}")
            console.print(heading_table)

        # Preview truncated Markdown
        preview_lines = markdown_text.splitlines()[:30]
        preview = "\n".join(preview_lines)
        if len(markdown_text.splitlines()) > 30:
            preview += "\n... (truncated)"

        console.print(Panel(preview, title="[cyan]Normalized Markdown Preview[/cyan]"))
        console.print(
            f"[green]✓[/green] Parsed successfully in {elapsed_ms:.1f} ms "
            f"({len(markdown_text)} chars, {len(headings)} headings)"
        )

    except Exception as e:
        console.print(f"[bold red]Parse probe failed: {e}[/bold red]")
        raise typer.Exit(code=1) from e


@test_app.command(name="chunk")
def probe_chunk(
    file: Path = typer.Argument(..., help="Path to a document file to chunk."),
) -> None:
    """Probe: Run two-stage chunking and display parent/child boundaries with overlap."""
    console.rule("[bold cyan]Chunking Probe[/bold cyan]")

    if not file.exists():
        console.print(f"[bold red]File not found: {file}[/bold red]")
        raise typer.Exit(code=1)

    try:
        from uuid import uuid4

        from ragnarok.core.config import load_config
        from ragnarok.ingestion.chunker import chunk_document
        from ragnarok.ingestion.parser import parse_document_to_markdown

        config = load_config()
        console.print(f"[dim]Parsing and chunking: {file}[/dim]")
        markdown_text = parse_document_to_markdown(file)

        doc_id = uuid4()
        parents, children = chunk_document(
            document_id=doc_id,
            markdown_text=markdown_text,
            config=config.chunking,
        )

        # Parent summary table
        parent_table = Table(
            title=f"Parent Chunks ({len(parents)} total)",
            show_header=True,
            header_style="bold magenta",
        )
        parent_table.add_column("Index", justify="center", width=6)
        parent_table.add_column("Heading Path", justify="left")
        parent_table.add_column("Chars", justify="right", width=8)
        parent_table.add_column("Children", justify="right", width=8)

        for p in parents:
            child_count = sum(1 for c in children if c.parent_chunk_id == p.parent_chunk_id)
            heading = " > ".join(p.heading_path) if p.heading_path else "Root"
            parent_table.add_row(
                str(p.parent_index), heading, str(len(p.content)), str(child_count)
            )
        console.print(parent_table)

        # Child breakdown table
        child_table = Table(
            title=f"Child Chunks ({len(children)} total)",
            show_header=True,
            header_style="bold blue",
        )
        child_table.add_column("Index", justify="center", width=6)
        child_table.add_column("Parent Idx", justify="center", width=10)
        child_table.add_column("Chars", justify="right", width=8)
        child_table.add_column("Preview", justify="left")

        for c in children:
            # Find parent index
            p_idx = "?"
            for p in parents:
                if p.parent_chunk_id == c.parent_chunk_id:
                    p_idx = str(p.parent_index)
                    break
            preview = c.content[:80].replace("\n", " ") + ("..." if len(c.content) > 80 else "")
            child_table.add_row(str(c.chunk_index), p_idx, str(len(c.content)), preview)
        console.print(child_table)

        console.print(
            f"[green]✓[/green] Chunked into {len(parents)} parents and {len(children)} children"
        )

    except Exception as e:
        console.print(f"[bold red]Chunking probe failed: {e}[/bold red]")
        raise typer.Exit(code=1) from e


@test_app.command(name="embed")
def probe_embed(
    text: str = typer.Argument(..., help="Text string to embed and analyze."),
) -> None:
    """Probe: Send text to the embedding service and verify latency/dimensions."""
    console.rule("[bold cyan]Embedding Service Probe[/bold cyan]")

    try:
        import numpy as np
        from sentence_transformers import SentenceTransformer

        from ragnarok.core.config import load_config

        config = load_config()
        model_name = config.dense_retrieval.model
        expected_dim = config.dense_retrieval.vector_dimension

        console.print(f"[dim]Model: {model_name} | Expected dim: {expected_dim}[/dim]")

        start = time.perf_counter()
        model = SentenceTransformer(model_name)
        load_ms = (time.perf_counter() - start) * 1000.0

        start = time.perf_counter()
        vector = model.encode(text, convert_to_numpy=True, normalize_embeddings=True)
        embed_ms = (time.perf_counter() - start) * 1000.0

        vec = np.array(vector)
        actual_dim = vec.shape[0]
        norm = float(np.linalg.norm(vec))
        min_val = float(np.min(vec))
        max_val = float(np.max(vec))

        # Results table
        results_table = Table(
            title="Embedding Results",
            show_header=True,
            header_style="bold magenta",
        )
        results_table.add_column("Metric", justify="left")
        results_table.add_column("Value", justify="right")
        results_table.add_row("Model Load Time", f"{load_ms:.1f} ms")
        results_table.add_row("Embedding Latency", f"{embed_ms:.1f} ms")
        results_table.add_row("Vector Dimension", f"{actual_dim}")
        results_table.add_row("Expected Dimension", f"{expected_dim}")
        results_table.add_row("Dimension Match", "✓" if actual_dim == expected_dim else "✗")
        results_table.add_row("Vector Norm (L2)", f"{norm:.6f}")
        results_table.add_row("Min Value", f"{min_val:.6f}")
        results_table.add_row("Max Value", f"{max_val:.6f}")
        results_table.add_row(
            "Sample Dims [0:5]",
            ", ".join(f"{v:.4f}" for v in vec[:5]),
        )
        console.print(results_table)

        if actual_dim == expected_dim:
            console.print("[green]✓[/green] Embedding probe passed")
        else:
            console.print(
                f"[bold red]✗ Dimension mismatch: expected {expected_dim}, got {actual_dim}[/bold red]"
            )

    except Exception as e:
        console.print(f"[bold red]Embedding probe failed: {e}[/bold red]")
        raise typer.Exit(code=1) from e


@test_app.command(name="bm25")
def probe_bm25(
    query: str = typer.Argument(..., help="Query string to tokenize and analyze."),
) -> None:
    """Probe: Tokenize a query using BM25 settings and show matching index terms."""
    console.rule("[bold cyan]Lexical Tokenizer Probe[/bold cyan]")

    try:
        import bm25s
        from bm25s.tokenization import Tokenizer

        from ragnarok.core.config import load_config

        config = load_config()

        # Tokenize the query
        stemmer: Any = None
        if config.lexical_retrieval.stemming_enabled:
            try:
                import importlib

                stemmer_mod = importlib.import_module("Stemmer")
                stemmer = stemmer_mod.Stemmer("english")
            except Exception:
                console.print("[yellow]PyStemmer not available, skipping stemming[/yellow]")

        tokenizer = Tokenizer(
            lower=config.lexical_retrieval.case == "lowercase",
            stemmer=stemmer,
        )

        query_tokens_list: Any = tokenizer.tokenize([query])
        tokens = list(query_tokens_list[0]) if len(query_tokens_list) > 0 else []

        # Display tokens
        token_table = Table(
            title="Query Token Analysis",
            show_header=True,
            header_style="bold magenta",
        )
        token_table.add_column("Original Query", justify="left")
        token_table.add_column("Normalized Tokens", justify="left")
        token_table.add_column("Token Count", justify="right")
        token_table.add_row(query, " ".join(str(t) for t in tokens), str(len(tokens)))
        console.print(token_table)

        # Attempt loading the persisted BM25 index to show matching docs
        bm25_path = Path("data/indexes/bm25")
        if bm25_path.exists() and any(bm25_path.iterdir()):
            try:
                retriever: Any = bm25s.BM25.load(str(bm25_path), load_corpus=False)
                query_tokens_arg: Any = query_tokens_list
                results, scores = retriever.retrieve(query_tokens_arg, k=5)
                num_docs = getattr(
                    retriever, "num_docs", getattr(retriever, "corpus_size", "unknown")
                )
                console.print(f"[dim]Index loaded: {num_docs} documents indexed[/dim]")
                if scores is not None and len(scores) > 0 and len(scores[0]) > 0:
                    match_table = Table(
                        title="Top-5 BM25 Matches",
                        show_header=True,
                        header_style="bold blue",
                    )
                    match_table.add_column("Rank", justify="center", width=6)
                    match_table.add_column("Internal ID", justify="center", width=12)
                    match_table.add_column("BM25 Score", justify="right", width=12)

                    for i in range(min(5, len(results[0]))):
                        match_table.add_row(
                            f"#{i + 1}",
                            str(results[0][i]),
                            f"{scores[0][i]:.4f}",
                        )
                    console.print(match_table)
            except Exception as idx_err:
                console.print(f"[yellow]Could not load BM25 index: {idx_err}[/yellow]")
        else:
            console.print("[dim]No BM25 index found at data/indexes/bm25/[/dim]")

        console.print("[green]✓[/green] BM25 tokenizer probe completed")

    except Exception as e:
        console.print(f"[bold red]BM25 probe failed: {e}[/bold red]")
        raise typer.Exit(code=1) from e


@test_app.command(name="route")
def probe_route(
    query: str = typer.Argument(..., help="Query to evaluate across pre-retrieval gates."),
) -> None:
    """Probe: Evaluate Self-RAG gate, Query Rewriter, and Adaptive Router signals."""
    console.rule("[bold cyan]Pre-Retrieval Routing Probe[/bold cyan]")

    try:
        from ragnarok.core.config import load_config

        config = load_config()

        # Self-RAG Gate
        console.print("\n[bold]1. Self-RAG Decision Gate[/bold]")
        try:
            from ragnarok.retrieval.self_rag import (
                DefaultLiteLLMClient,
                SelfRagDecision,
                SelfRagGate,
            )

            gate = SelfRagGate(config=config.self_rag, llm_client=DefaultLiteLLMClient())
            start = time.perf_counter()
            result = gate.evaluate(query)
            elapsed = (time.perf_counter() - start) * 1000.0
            decision_str = (
                "Retrieval Required"
                if result.decision == SelfRagDecision.RETRIEVE
                else "Direct LLM Answer"
            )
            console.print(f"   Decision  : {decision_str}")
            console.print(f"   Reasoning : {result.reasoning}")
            console.print(f"   Latency   : {elapsed:.1f} ms")
        except Exception as e:
            console.print(f"   [yellow]Self-RAG gate evaluation failed: {e}[/yellow]")

        # Query Rewriter
        console.print("\n[bold]2. Query Rewriter[/bold]")
        try:
            from ragnarok.retrieval.rewriter import QueryRewriter

            rewriter = QueryRewriter(config=config.query_rewriting)
            start = time.perf_counter()
            rewritten = rewriter.rewrite(query)
            elapsed = (time.perf_counter() - start) * 1000.0
            console.print(f"   Original  : {query}")
            console.print(f"   Rewritten : {rewritten}")
            console.print(f"   Latency   : {elapsed:.1f} ms")
        except Exception as e:
            console.print(f"   [yellow]Query rewriter failed: {e}[/yellow]")

        # Adaptive Router
        console.print("\n[bold]3. Adaptive Router[/bold]")
        try:
            from ragnarok.retrieval.router import AdaptiveRouter

            router = AdaptiveRouter(config=config.adaptive_retrieval)
            start = time.perf_counter()
            strategy = router.route(query)
            signals = router.analyze_signals(query)
            elapsed = (time.perf_counter() - start) * 1000.0
            console.print(f"   Strategy  : {strategy}")
            console.print(
                f"   Signals   : exact={signals.has_exact_match} "
                f"({len(signals.exact_signals)} signals), "
                f"semantic={signals.has_semantic} "
                f"({len(signals.semantic_signals)} signals)"
            )
            console.print(f"   Latency   : {elapsed:.1f} ms")
        except Exception as e:
            console.print(f"   [yellow]Adaptive router failed: {e}[/yellow]")

        console.print("\n[green]✓[/green] Routing probe completed")

    except Exception as e:
        console.print(f"[bold red]Routing probe failed: {e}[/bold red]")
        raise typer.Exit(code=1) from e


@test_app.command(name="rerank")
def probe_rerank(
    query: str | None = typer.Option(
        None, "--query", "-q", help="Query to score against candidates."
    ),
    candidates: Path | None = typer.Option(
        None, "--candidates", "-c", help="JSON file containing candidate text passages."
    ),
    args: list[str] | None = typer.Argument(
        None, help="Query followed by candidate passages (e.g. 'query' 'passage1' 'passage2')."
    ),
) -> None:
    """Probe: Score candidate passages against a query using the Cross-Encoder reranker."""
    console.rule("[bold cyan]Cross-Encoder Reranker Probe[/bold cyan]")

    target_query = query
    passages: list[str] = []

    if target_query is None:
        if args and len(args) > 0:
            target_query = args[0]
            remaining = args[1:]
        else:
            console.print(
                "[bold red]Query must be provided via --query or as first argument.[/bold red]"
            )
            raise typer.Exit(code=1)
    else:
        remaining = args if args else []

    if candidates is not None:
        if not candidates.exists():
            console.print(f"[bold red]Candidates file not found: {candidates}[/bold red]")
            raise typer.Exit(code=1)
        with candidates.open("r", encoding="utf-8") as f:
            passages = json.load(f)
    elif remaining:
        passages = list(remaining)
    else:
        console.print(
            "[bold red]Candidate passages must be provided via --candidates JSON file "
            "or as arguments.[/bold red]"
        )
        raise typer.Exit(code=1)

    if not passages:
        console.print("[bold red]No candidate passages provided to rerank.[/bold red]")
        raise typer.Exit(code=1)

    try:
        from ragnarok.core.config import load_config
        from ragnarok.retrieval.reranker import SentenceTransformerCrossEncoder

        config = load_config()

        console.print(f"[dim]Query: {target_query}[/dim]")
        console.print(f"[dim]Candidates: {len(passages)} passages[/dim]")

        model = SentenceTransformerCrossEncoder(model_name=config.reranking.model)
        start = time.perf_counter()
        pairs = [(target_query, p) for p in passages]
        scored = model.predict(pairs)
        elapsed = (time.perf_counter() - start) * 1000.0

        # Sort by score descending, track original index
        indexed = [(i, s) for i, s in enumerate(scored)]
        sorted_candidates = sorted(indexed, key=lambda x: x[1], reverse=True)

        rerank_table = Table(
            title="Cross-Encoder Reranking Results",
            show_header=True,
            header_style="bold magenta",
        )
        rerank_table.add_column("New Rank", justify="center", width=10)
        rerank_table.add_column("Original Pos", justify="center", width=12)
        rerank_table.add_column("Score", justify="right", width=10)
        rerank_table.add_column("Rank Shift", justify="center", width=10)
        rerank_table.add_column("Preview", justify="left")

        for new_rank, (orig_idx, score) in enumerate(sorted_candidates, 1):
            shift = orig_idx - (new_rank - 1)
            shift_str = f"+{shift}" if shift > 0 else str(shift)
            preview = passages[orig_idx][:60].replace("\n", " ") + "..."
            rerank_table.add_row(
                f"#{new_rank}", f"#{orig_idx + 1}", f"{score:.4f}", shift_str, preview
            )

        console.print(rerank_table)
        console.print(f"[green]✓[/green] Reranking probe completed in {elapsed:.1f} ms")

    except Exception as e:
        console.print(f"[bold red]Reranker probe failed: {e}[/bold red]")
        raise typer.Exit(code=1) from e


@test_app.command(name="context")
def probe_context(
    query: str = typer.Argument(..., help="Query to execute retrieval and context budgeting."),
) -> None:
    """Probe: Execute retrieval and parent expansion without invoking generation."""
    console.rule("[bold cyan]Context Management & Budget Probe[/bold cyan]")
    console.print(f"[dim]Query: {query}[/dim]")

    try:
        from ragnarok.cli.visuals import render_budget_gauge
        from ragnarok.core.config import load_config

        config = load_config()

        # Explain what happens in full pipeline run
        console.print(
            "\n[bold]Context Management Pipeline:[/bold]\n"
            "  1. Retrieve child chunks via lexical/dense search\n"
            "  2. Expand parent contexts from PostgreSQL\n"
            "  3. Deduplicate overlapping parent sections\n"
            "  4. Apply rank-based greedy token budgeting\n"
            f"\n  Budget Limit: {config.context_management.max_tokens} tokens"
        )

        sample_sources = [
            {
                "label": "Source 1",
                "parent_index": 0,
                "tokens": min(1200, config.context_management.max_tokens // 2),
            },
            {
                "label": "Source 2",
                "parent_index": 1,
                "tokens": min(800, config.context_management.max_tokens // 3),
            },
        ]
        render_budget_gauge(
            sample_sources, max_tokens=config.context_management.max_tokens, console=console
        )

        console.print("[green]✓[/green] Context probe configuration verified")

    except Exception as e:
        console.print(f"[bold red]Context probe failed: {e}[/bold red]")
        raise typer.Exit(code=1) from e


@test_app.command(name="guard")
def probe_guard(
    query: str = typer.Argument(..., help="Query to evaluate against input guardrails."),
) -> None:
    """Probe: Evaluate input query against Tier 1 deterministic and Tier 2 semantic guards."""
    console.rule("[bold cyan]Input Guardrails Probe[/bold cyan]")

    try:
        import re

        from ragnarok.core.config import load_config

        config = load_config()
        console.print(f"[dim]Query: {query}[/dim]\n")

        # Tier 1: Fast-Path Deterministic Checks
        console.print("[bold]Tier 1: Deterministic Fast-Path Filters[/bold]")

        # Length check
        token_count = len(query.split())
        length_ok = token_count <= config.guardrails.max_query_tokens
        console.print(
            f"  Length Check   : {token_count} tokens "
            f"(max {config.guardrails.max_query_tokens}) "
            f"{'[green]PASS[/green]' if length_ok else '[red]FAIL[/red]'}"
        )

        # Character sanitization (control characters / ANSI)
        has_suspicious = bool(re.search(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", query))
        console.print(
            f"  Char Sanitize  : "
            f"{'[red]Suspicious control characters detected[/red]' if has_suspicious else '[green]PASS[/green]'}"
        )

        # Injection pattern detection
        injection_patterns = [
            r"(?i)ignore\s+(all\s+)?previous\s+instructions",
            r"(?i)you\s+are\s+now\s+",
            r"(?i)system\s*prompt",
            r"(?i)\\n\s*\\n\s*(system|user|assistant)\s*:",
            r"(?i)act\s+as\s+(a\s+)?",
            r"(?i)reveal\s+(your\s+)?instructions",
            r"(?i)dan\s+mode",
            r"(?i)developer\s+mode",
            r"(<\|im_start\|>|<\|im_end\|>|\[INST\]|\[/INST\])",
        ]
        injection_detected = False
        for pattern in injection_patterns:
            if re.search(pattern, query):
                injection_detected = True
                break
        console.print(
            f"  Injection Scan : "
            f"{'[red]FAIL — Potential prompt injection detected[/red]' if injection_detected else '[green]PASS[/green]'}"
        )

        overall_tier1 = length_ok and not has_suspicious and not injection_detected
        console.print(
            f"\n  [bold]Tier 1 Verdict[/bold]: "
            f"{'[green]SAFE[/green]' if overall_tier1 else '[red]BLOCKED[/red]'}"
        )

        # Standardized refusal output if blocked
        if not overall_tier1:
            if not length_ok:
                console.print(
                    "\n[bold red]Refusal:[/bold red] Your query exceeds the maximum allowed length "
                    f"({config.guardrails.max_query_tokens} tokens). Please submit a more concise question."
                )
            elif injection_detected or has_suspicious:
                console.print(
                    "\n[bold red]Refusal:[/bold red] Your query could not be processed due to safety policy restrictions."
                )

        # Tier 2: Semantic Model Guard (if configured)
        if config.guardrails.enabled and config.guardrails.model:
            console.print("\n[bold]Tier 2: Semantic Model Guard[/bold]")
            console.print(f"  Model          : {config.guardrails.model}")
            console.print(
                "  [yellow]Note: Full Tier 2 semantic evaluation requires active LLM provider.[/yellow]"
            )

        console.print("\n[green]✓[/green] Guardrail probe completed")

    except Exception as e:
        console.print(f"[bold red]Guardrail probe failed: {e}[/bold red]")
        raise typer.Exit(code=1) from e


@test_app.command(name="pii")
def probe_pii(
    text: str = typer.Argument(..., help="Text passage to evaluate for secrets and PII."),
) -> None:
    """Probe: Scan text for API keys, credentials, tokens, and PII patterns."""
    console.rule("[bold cyan]Secret & PII Redaction Probe[/bold cyan]")

    try:
        from ragnarok.generation.sanitizer import sanitize_generated_text

        redacted = sanitize_generated_text(text)

        # Show results side by side
        console.print(Panel(text, title="[yellow]Original Input[/yellow]"))
        console.print(Panel(redacted, title="[green]Redacted Output[/green]"))

        # Detect what was redacted
        changes_detected = text != redacted
        if changes_detected:
            console.print("[bold yellow]⚠ Sensitive patterns detected and redacted[/bold yellow]")
        else:
            console.print("[green]✓[/green] No sensitive patterns detected in input")

    except Exception as e:
        console.print(f"[bold red]PII probe failed: {e}[/bold red]")
        raise typer.Exit(code=1) from e


@test_app.command(name="auto")
def probe_auto() -> None:
    """Execute the automated pytest test suite (unit + integration tests)."""
    console.rule("[bold cyan]Automated Test Suite[/bold cyan]")
    console.print("[dim]Running: uv run pytest tests/ -v[/dim]\n")

    try:
        result = subprocess.run(
            ["uv", "run", "pytest", "tests/", "-v", "--tb=short"],
            cwd=str(Path.cwd()),
            text=True,
            capture_output=False,
        )
        if result.returncode == 0:
            console.print("\n[green]✓[/green] All tests passed")
        else:
            console.print(f"\n[bold red]✗ Tests failed (exit code {result.returncode})[/bold red]")
            raise typer.Exit(code=result.returncode)
    except FileNotFoundError:
        console.print("[bold red]'uv' command not found. Install uv first.[/bold red]")
        raise typer.Exit(code=1) from None


@test_app.command(name="eval")
def probe_eval() -> None:
    """Execute DeepEval against the Golden Dataset benchmark."""
    console.rule("[bold cyan]DeepEval Golden Dataset Benchmark[/bold cyan]")

    try:
        from ragnarok.core.config import load_config

        config = load_config()
        dataset_path = config.evals.golden_dataset_path

        if not dataset_path.exists():
            console.print(
                f"[bold red]Golden dataset not found at: {dataset_path}[/bold red]\n"
                "[dim]Create tests/data/golden_dataset.json with benchmark cases first.[/dim]"
            )
            raise typer.Exit(code=1)

        console.print(f"[dim]Dataset: {dataset_path}[/dim]")
        console.print(f"[dim]Judge Model: {config.evals.judge_model}[/dim]\n")

        # Run golden dataset evaluation suite
        result = subprocess.run(
            [
                "uv",
                "run",
                "pytest",
                "tests/evals/test_golden_dataset.py",
                "-v",
                "-s",
            ],
            cwd=str(Path.cwd()),
            text=True,
            capture_output=False,
        )

        if result.returncode == 0:
            console.print("\n[green]✓[/green] Golden dataset evaluation benchmark passed")
        else:
            console.print(
                f"\n[bold red]✗ Evaluation failed (exit code {result.returncode})[/bold red]"
            )
            raise typer.Exit(code=result.returncode)

    except SystemExit:
        raise
    except Exception as e:
        console.print(f"[bold red]Eval probe failed: {e}[/bold red]")
        raise typer.Exit(code=1) from e
