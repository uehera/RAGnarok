"""
Interactive Developer REPL Shell for RAGnarok.

Provides multi-turn querying, live streamed token synthesis, inline citation
rendering, and runtime parameter overrides via in-session slash commands.
Launched via `rag chat` (alias `rag c`).
"""

from __future__ import annotations

import structlog
from rich.console import Console
from rich.panel import Panel

logger = structlog.get_logger(__name__)


class ReplSession:
    """
    Stateful REPL session encapsulating runtime overrides, conversation history,
    and terminal rendering state.
    """

    def __init__(self, console: Console | None = None) -> None:
        self.console = console if console is not None else Console()
        self.strategy: str = "hybrid"
        self.rerank_enabled: bool = True
        self.budget_tokens: int = 4000
        self.show_waterfall: bool = False
        self.show_provenance: bool = False
        self.history: list[dict[str, str]] = []
        self.session_token_count: int = 0
        self.query_count: int = 0

    def print_banner(self) -> None:
        """Renders the REPL session welcome banner."""
        from rich.table import Table

        grid = Table.grid(padding=(0, 2))
        grid.add_column(style="bold cyan", width=18)
        grid.add_column(style="white")

        grid.add_row("Environment", "[bold white]RAGnarok Interactive Chat Shell[/bold white]")
        grid.add_row("Mode", "Grounded Conversational Q&A with Live Token Streaming & Footnotes")
        grid.add_row("Session Controls", "/help (extensions & options) | /clear | /exit (quit)")
        grid.add_row(
            "Live Extensions",
            "/waterfall (timing waterfall) | /provenance (rank tracking) | /strategy | /rerank | /budget",
        )
        grid.add_row(
            "Developer Tools",
            "[dim]Exit and run [bold yellow]rag -dev[/bold yellow] for ingestion, reindexing, diagnostics & test probes[/dim]",
        )

        panel = Panel(
            grid,
            title="[bold cyan]RAGnarok Chat Shell[/bold cyan]",
            subtitle="[dim]Type your inquiry to retrieve and stream answer with verified citations[/dim]",
            border_style="cyan",
            padding=(1, 2),
        )
        self.console.print(panel)

    def print_help(self) -> None:
        """Displays the slash command reference."""
        from rich.table import Table

        help_table = Table(
            title="Chat Shell Extensions & Slash Commands",
            show_header=True,
            header_style="bold magenta",
            border_style="bright_black",
        )
        help_table.add_column("Command", justify="left", width=32)
        help_table.add_column("Type", justify="center", width=14)
        help_table.add_column("Description", justify="left")

        help_table.add_row(
            "/waterfall",
            "Extension",
            "Toggle pipeline latency waterfall chart after every response",
        )
        help_table.add_row(
            "/provenance",
            "Extension",
            "Toggle candidate score and rank provenance table after every response",
        )
        help_table.add_row(
            "/strategy <lexical|dense|hybrid>",
            "Override",
            "Dynamically change retrieval strategy for subsequent queries",
        )
        help_table.add_row(
            "/rerank <on|off>",
            "Override",
            "Dynamically enable or disable the Cross-Encoder reranker",
        )
        help_table.add_row(
            "/budget <tokens>",
            "Override",
            "Adjust context token budget dynamically (e.g. /budget 3500)",
        )
        help_table.add_row(
            "/stats", "Inspection", "Display current session statistics and active overrides"
        )
        help_table.add_row("/clear", "Session", "Clear the terminal conversation history buffer")
        help_table.add_row("/exit, /quit", "Session", "Exit the interactive chat shell")
        help_table.add_row("/help", "Session", "Display this reference table")

        self.console.print(help_table)
        self.console.print(
            "\n[dim]Note: For ingestion, index management, and component probes, run [bold yellow]rag -dev[/bold yellow].[/dim]"
        )

    def handle_slash_command(self, command: str) -> bool:
        """
        Processes a slash command and applies runtime overrides.

        Returns:
            True if the REPL should continue, False if it should exit.
        """
        parts = command.strip().split()
        cmd = parts[0].lower()

        if cmd in ("/exit", "/quit"):
            self.console.print("[dim]Ending session...[/dim]")
            return False

        elif cmd == "/help":
            self.print_help()

        elif cmd == "/strategy":
            if len(parts) < 2 or parts[1] not in ("lexical", "dense", "hybrid"):
                self.console.print("[yellow]Usage: /strategy <lexical|dense|hybrid>[/yellow]")
            else:
                self.strategy = parts[1]
                self.console.print(f"[green]✓[/green] Strategy set to: {self.strategy}")

        elif cmd == "/rerank":
            if len(parts) < 2 or parts[1] not in ("on", "off"):
                self.console.print("[yellow]Usage: /rerank <on|off>[/yellow]")
            else:
                self.rerank_enabled = parts[1] == "on"
                state = "enabled" if self.rerank_enabled else "disabled"
                self.console.print(f"[green]✓[/green] Reranking {state}")

        elif cmd == "/budget":
            if len(parts) < 2:
                self.console.print("[yellow]Usage: /budget <tokens>[/yellow]")
            else:
                try:
                    self.budget_tokens = int(parts[1])
                    self.console.print(
                        f"[green]✓[/green] Token budget set to: {self.budget_tokens}"
                    )
                except ValueError:
                    self.console.print("[yellow]Invalid token count[/yellow]")

        elif cmd == "/waterfall":
            self.show_waterfall = not self.show_waterfall
            state = "on" if self.show_waterfall else "off"
            self.console.print(f"[green]✓[/green] Waterfall display: {state}")

        elif cmd == "/provenance":
            self.show_provenance = not self.show_provenance
            state = "on" if self.show_provenance else "off"
            self.console.print(f"[green]✓[/green] Provenance display: {state}")

        elif cmd == "/stats":
            self.console.print(
                f"\n[bold]Session Statistics:[/bold]\n"
                f"  Queries Executed : {self.query_count}\n"
                f"  History Entries  : {len(self.history)}\n"
                f"  Strategy         : {self.strategy}\n"
                f"  Reranking        : {'on' if self.rerank_enabled else 'off'}\n"
                f"  Token Budget     : {self.budget_tokens}\n"
                f"  Waterfall        : {'on' if self.show_waterfall else 'off'}\n"
                f"  Provenance       : {'on' if self.show_provenance else 'off'}\n"
            )

        elif cmd == "/clear":
            self.history.clear()
            self.query_count = 0
            self.console.print("[green]✓[/green] Conversation history cleared")

        else:
            self.console.print(
                f"[yellow]Unknown command: {cmd}. Type /help for available commands.[/yellow]"
            )

        return True

    def execute_query(self, query: str) -> None:
        """
        Executes a full retrieval and generation pipeline for a single query turn,
        rendering the streamed answer and optional metrics.
        """
        # Guard against dev workspace commands entered directly in chat shell
        first_token = query.strip().split()[0].lower() if query.strip() else ""
        if first_token in (
            "ingest",
            "in",
            "reindex",
            "rx",
            "doctor",
            "doc",
            "test",
            "parse",
            "chunk",
            "embed",
            "bm25",
            "route",
            "rerank",
            "auto",
            "eval",
        ):
            self.console.print(
                f"\n[bold yellow]Notice:[/bold yellow] [dim]'{first_token}' is a developer workspace command.[/dim]\n"
                "[cyan]The chat shell is dedicated to conversational Q&A. To execute document ingestion, index rebuilds, or component probes, exit and run:[/cyan]\n"
                "  [bold green]rag -dev[/bold green]  [dim](or 'rag --dev')[/dim]\n"
            )
            return

        self.query_count += 1
        self.history.append({"role": "user", "content": query})

        try:
            from ragnarok.core.config import load_config
            from ragnarok.generation.service import GenerationService
            from ragnarok.retrieval.pipeline import retrieve_and_assemble_context

            config = load_config()

            self.console.print(
                f"\n[bold cyan]RAGnarok[/bold cyan] [dim](strategy={self.strategy}, "
                f"rerank={'on' if self.rerank_enabled else 'off'}, "
                f"budget={self.budget_tokens})[/dim]\n"
            )

            # 1. Retrieve candidates and expand parent context
            context_payload, candidates, stage_latencies = retrieve_and_assemble_context(
                query=query,
                config=config,
                strategy=self.strategy,
                rerank=self.rerank_enabled,
                budget=self.budget_tokens,
            )

            service = GenerationService.from_config(config=config, console=self.console)

            # 2. Stream answer with assembled context
            response = service.generate_sync(
                query=query,
                context=context_payload,
                render_live=True,
            )

            # Store answer in history
            self.history.append({"role": "assistant", "content": response.answer})

            # Render footnotes
            if response.citations:
                from ragnarok.generation.citations import format_footnotes_table

                self.console.print(format_footnotes_table(response.citations))

            # Render optional provenance table
            if self.show_provenance:
                from ragnarok.cli.visuals import render_provenance_table

                render_provenance_table(candidates, console=self.console)

            # Render optional metrics
            if self.show_waterfall:
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
                    },
                    console=self.console,
                )

            # Summary line
            self.console.print(
                f"\n[dim]Model: {response.model_used} | "
                f"TTFT: {response.latency_metrics.ttft_ms:.0f}ms | "
                f"Total: {response.latency_metrics.total_duration_ms:.0f}ms | "
                f"Fallback: {'Yes' if response.fallback_triggered else 'No'}[/dim]"
            )

        except KeyboardInterrupt:
            self.console.print("\n[dim]Query interrupted.[/dim]")
        except Exception as e:
            self.console.print(f"\n[bold red]Error: {e}[/bold red]")
            logger.error("REPL query execution failed", error=str(e))


def run_repl(console: Console | None = None) -> None:
    """
    Main REPL loop. Reads user input, dispatches slash commands or queries,
    and renders streaming results until the user exits.
    """
    session = ReplSession(console=console)
    session.print_banner()

    try:
        while True:
            try:
                user_input = input("\n🔱 rag> ").strip()
            except EOFError:
                session.console.print("\n[dim]End of input. Exiting...[/dim]")
                break

            if not user_input:
                continue

            if user_input.startswith("/"):
                should_continue = session.handle_slash_command(user_input)
                if not should_continue:
                    break
            else:
                session.execute_query(user_input)

    except KeyboardInterrupt:
        session.console.print("\n[dim]Session interrupted. Goodbye![/dim]")

    session.console.print(
        f"\n[bold cyan]Session ended.[/bold cyan] "
        f"Queries: {session.query_count} | Turns: {len(session.history)}"
    )
