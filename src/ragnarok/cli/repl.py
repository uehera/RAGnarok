"""
RAGnarok Interactive REPL Shell — Conversational Terminal Client.

Provides a focused, minimal conversational interface with multi-turn memory,
streaming generation, verified inline citations, and runtime retrieval controls.
Commands: natural queries, /sources, /strategy, /rerank, /clear, /exit.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any

import structlog
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from ragnarok.context.contract import AssembledContextPayload
from ragnarok.generation.contract import GenerationResponse
from ragnarok.retrieval.contract import ChildResultCandidate

logger = structlog.get_logger(__name__)


def configure_chat_logging(quiet: bool = True) -> None:
    """Configures logging verbosity for the chat shell."""
    if quiet:
        structlog.configure(
            wrapper_class=structlog.make_filtering_bound_logger(logging.CRITICAL),
        )
        logging.getLogger().setLevel(logging.CRITICAL)
        for name in (
            "httpx",
            "httpcore",
            "litellm",
            "LiteLLM",
            "fastembed",
            "flashrank",
            "urllib3",
            "aiohttp",
        ):
            log = logging.getLogger(name)
            log.setLevel(logging.CRITICAL)
            log.propagate = False
            log.handlers = []
        try:
            import litellm

            litellm.suppress_debug_info = True
            litellm.set_verbose = False
        except Exception:
            pass
    else:
        structlog.configure(
            wrapper_class=structlog.make_filtering_bound_logger(logging.INFO),
        )
        logging.getLogger().setLevel(logging.INFO)


class ReplSession:
    """
    Focused interactive chat shell managing multi-turn conversation memory,
    retrieval overrides, and verified citation formatting.
    """

    def __init__(self, console: Console | None = None) -> None:
        self.console = console if console is not None else Console()
        self.strategy: str = "hybrid"
        self.rerank_enabled: bool = True
        self.history: list[dict[str, str]] = []
        self.query_count: int = 0
        self.last_query: str | None = None
        self.last_response: GenerationResponse | None = None
        self.last_candidates: list[ChildResultCandidate] = []
        self.last_context_payload: AssembledContextPayload | None = None
        self.last_stage_latencies: dict[str, float] = {}

    def print_banner(self) -> None:
        """Renders the minimal REPL session welcome banner."""
        grid = Table.grid(padding=(0, 2))
        grid.add_column(style="bold cyan", width=16)
        grid.add_column(style="white")

        grid.add_row("Environment", "[bold white]RAGnarok Interactive Chat Shell[/bold white]")
        grid.add_row(
            "Mode",
            "Grounded Conversational Q&A with Multi-Turn Memory & Citations",
        )
        grid.add_row(
            "Commands",
            "/sources | /strategy <type> | /rerank <on|off> | /clear | /exit",
        )

        panel = Panel(
            grid,
            title="[bold cyan]RAGnarok Chat[/bold cyan]",
            subtitle="[dim]Ask any question or type /help for chat controls[/dim]",
            border_style="cyan",
            padding=(1, 2),
        )
        self.console.print(panel)

    def print_help(self) -> None:
        """Displays the minimal chat command reference."""
        help_table = Table(
            title="RAGnarok Chat Commands",
            show_header=True,
            header_style="bold magenta",
            border_style="bright_black",
        )
        help_table.add_column("Command", justify="left", width=28)
        help_table.add_column("Category", justify="center", width=12)
        help_table.add_column("Description", justify="left")

        help_table.add_row(
            "/sources",
            "Sources",
            "List all context blocks retrieved for the last answer",
        )
        help_table.add_row(
            "/strategy <type>",
            "Override",
            "Switch retrieval strategy (lexical | dense | hybrid)",
        )
        help_table.add_row(
            "/rerank <on|off>",
            "Override",
            "Toggle Cross-Encoder reranking",
        )
        help_table.add_row(
            "/clear",
            "Session",
            "Clear conversation memory and reset turn history",
        )
        help_table.add_row(
            "/exit, :q, /quit",
            "Session",
            "Exit the interactive chat shell",
        )

        self.console.print(help_table)
        self.console.print("[dim]Tip: For ingestion, indexing, and diagnostic tools, use [bold yellow]rag dev[/bold yellow].[/dim]\n")

    def handle_slash_command(self, command: str) -> bool:
        """
        Processes a chat slash command.

        Returns:
            True if the REPL should continue, False if it should exit.
        """
        parts = command.strip().split()
        if not parts:
            return True
        cmd = parts[0].lower()
        args = parts[1:]

        if cmd in ("/exit", "/quit", ":q"):
            self.console.print("[dim]Ending session...[/dim]")
            return False

        elif cmd == "/help":
            self.print_help()

        elif cmd in ("/sources", "/source"):
            self._display_sources_list()

        elif cmd == "/strategy":
            if not args or args[0] not in ("lexical", "dense", "hybrid"):
                self.console.print("[yellow]Usage: /strategy <lexical|dense|hybrid>[/yellow]")
            else:
                self.strategy = args[0]
                self.console.print(f"[green]✓[/green] Strategy set to: {self.strategy}")

        elif cmd == "/rerank":
            if not args or args[0] not in ("on", "off"):
                self.console.print("[yellow]Usage: /rerank <on|off>[/yellow]")
            else:
                self.rerank_enabled = args[0] == "on"
                state = "enabled" if self.rerank_enabled else "disabled"
                self.console.print(f"[green]✓[/green] Reranking {state}")

        elif cmd == "/clear":
            self.history.clear()
            self.query_count = 0
            self.last_query = None
            self.last_response = None
            self.last_candidates.clear()
            self.last_context_payload = None
            self.last_stage_latencies.clear()
            self.console.print("[green]✓[/green] Conversation history and turn memory cleared")

        else:
            self.console.print(
                f"[yellow]Unknown command: {cmd}. Available in chat: /sources, /strategy, /rerank, /clear, /exit.\n"
                f"For developer tools (ingest, reindex, doctor, stats), use 'rag dev'.[/yellow]"
            )

        return True

    def _display_sources_list(self) -> None:
        """Displays a summary table of all context blocks from the last query."""
        if not self.last_context_payload or not self.last_context_payload.context_blocks:
            self.console.print("[yellow]No source blocks available. Run a query first.[/yellow]")
            return

        table = Table(
            title=f"Retrieved Source Blocks (Query: '{self.last_query or ''}')",
            show_header=True,
            header_style="bold cyan",
            border_style="bright_black",
        )
        table.add_column("Index", justify="center", width=8, style="bold yellow")
        table.add_column("Document", justify="left", style="white", width=28)
        table.add_column("Section / Breadcrumbs", justify="left", style="dim")
        table.add_column("Tokens", justify="right", width=10)
        table.add_column("Best Child", justify="center", width=12)

        from ragnarok.context.formatter import format_heading_path

        for block in self.last_context_payload.context_blocks:
            table.add_row(
                f"[{block.source_index}]",
                block.filename,
                format_heading_path(block.heading_path),
                f"{block.token_count:,}",
                f"Rank #{block.best_child_rank}",
            )

        self.console.print(table)

    def _derive_retrieval_query(self, query: str) -> str:
        """
        Derives an augmented retrieval query for conversational follow-ups
        to preserve context across multi-turn sessions.
        """
        if not self.history:
            return query

        # Find the most recent user turn
        prev_user_queries = [
            turn["content"] for turn in self.history if turn.get("role") == "user"
        ]
        if not prev_user_queries:
            return query

        last_user_query = prev_user_queries[-1].strip()

        # Follow-up indicators
        words = query.strip().lower().split()
        follow_up_tokens = {
            "it", "they", "them", "this", "that", "these", "those",
            "second", "third", "first", "compare", "more", "difference",
            "summarize", "why", "how", "details", "expand", "patterns",
            "approach", "both", "another", "instead",
        }
        is_follow_up = len(words) <= 5 or any(w in follow_up_tokens for w in words)

        if is_follow_up and last_user_query and last_user_query != query:
            return f"{query} ({last_user_query})"

        return query

    def execute_query(self, query: str) -> None:
        """
        Executes a full retrieval and generation pipeline for a single query turn,
        rendering the streamed answer with verified citations.
        """
        self.query_count += 1
        self.last_query = query

        try:
            from ragnarok.core.config import load_config
            from ragnarok.generation.service import GenerationService
            from ragnarok.retrieval.pipeline import retrieve_and_assemble_context

            config = load_config()

            self.console.print(
                f"\n[bold cyan]RAGnarok[/bold cyan] [dim](strategy={self.strategy}, "
                f"rerank={'on' if self.rerank_enabled else 'off'})[/dim]\n"
            )

            # 0. Input Guardrails Check
            from ragnarok.guardrails import GuardrailsService

            guard_service = GuardrailsService.from_config(config=config)
            guard_decision = guard_service.validate_input(query)
            if not guard_decision.is_safe or (config.guardrails.strict_mode and not guard_decision.is_topical):
                self.console.print(f"\n[bold red]Refusal:[/bold red] {guard_decision.fallback_message}\n")
                return

            # Derive contextualized query for retrieval
            retrieval_query = self._derive_retrieval_query(query)

            # 1. Retrieve candidates and expand parent context
            context_payload, candidates, stage_latencies = retrieve_and_assemble_context(
                query=retrieval_query,
                config=config,
                strategy=self.strategy,
                rerank=self.rerank_enabled,
                budget=config.context_management.max_tokens,
            )

            # Cache last retrieval artifacts for on-demand inspection
            self.last_candidates = candidates
            self.last_context_payload = context_payload
            self.last_stage_latencies = stage_latencies

            service = GenerationService.from_config(config=config, console=self.console)

            # 2. Stream answer with assembled context and multi-turn history
            response = service.generate_sync(
                query=query,
                context=context_payload,
                render_live=True,
                history=list(self.history),
            )

            self.last_response = response

            # Post-Hoc Output Guard Warnings
            if response.secrets_redacted:
                self.console.print(
                    f"\n[bold yellow]🛡️ Output Guard:[/bold yellow] [dim]Redacted sensitive tokens: {', '.join(response.secrets_redacted)}[/dim]"
                )
            if not response.is_grounded:
                self.console.print(
                    "\n[bold red]⚠️ Grounding Warning:[/bold red] [red]The generated answer contains assertions not grounded in the retrieved documentation.[/red]"
                )

            # Append to multi-turn conversation memory
            self.history.append({"role": "user", "content": query})
            if response.is_grounded:
                self.history.append({"role": "assistant", "content": response.answer})
            else:
                self.history.append(
                    {"role": "assistant", "content": "[Ungrounded response flagged by Output Guard]"}
                )

            # Render footnotes
            if response.citations:
                from ragnarok.generation.citations import format_footnotes_table

                self.console.print(format_footnotes_table(response.citations))

            # Clean, minimal 1-line footer
            self.console.print(
                f"\n[dim]Model: {response.model_used} | "
                f"TTFT: {response.latency_metrics.ttft_ms:.0f}ms | "
                f"Total: {response.latency_metrics.total_duration_ms:.0f}ms | "
                f"Sources: {context_payload.selected_parent_count} "
                f"(type [bold]/sources[/bold] to inspect)[/dim]"
            )

        except KeyboardInterrupt:
            self.console.print("\n[dim]Query interrupted.[/dim]")
        except Exception as e:
            error_str = str(e)
            if "Connection" in error_str or "resolution" in error_str or "api.groq.com" in error_str:
                self.console.print(
                    "\n[bold yellow]Provider Notice:[/bold yellow] [red]Unable to reach generation API endpoint (offline or network restriction).[/red]\n"
                    f"[dim]Retrieved {len(self.last_candidates)} source candidates successfully. Type [bold cyan]/sources[/bold cyan] to inspect source list.[/dim]"
                )
            else:
                self.console.print(f"\n[bold red]Error: {e}[/bold red]")
            logger.error("REPL query execution failed", error=error_str)


def run_repl(console: Console | None = None) -> None:
    """
    Main REPL loop. Reads user input, dispatches slash commands or queries,
    and renders streaming results with readline/prompt_toolkit support.
    """
    configure_chat_logging(quiet=True)

    target_console = console if console is not None else Console()
    session = ReplSession(console=target_console)
    session.print_banner()

    prompt_session: Any = None
    if sys.stdin.isatty():
        try:
            from prompt_toolkit import PromptSession
            from prompt_toolkit.completion import WordCompleter
            from prompt_toolkit.history import FileHistory

            history_file = Path.home() / ".ragnarok_history"
            completer = WordCompleter(
                ["/sources", "/strategy", "/rerank", "/clear", "/exit", "/quit", "/help"],
                ignore_case=True,
            )
            prompt_session = PromptSession(
                history=FileHistory(str(history_file)),
                completer=completer,
            )
        except Exception:
            prompt_session = None

    while True:
        try:
            if prompt_session is not None:
                user_input = prompt_session.prompt("chat > ").strip()
            else:
                user_input = input("chat > ").strip()

            if not user_input:
                continue

            if user_input.startswith("/"):
                should_continue = session.handle_slash_command(user_input)
                if not should_continue:
                    break
            else:
                session.execute_query(user_input)

        except (KeyboardInterrupt, EOFError):
            target_console.print("\n[dim]Ending session...[/dim]")
            break
