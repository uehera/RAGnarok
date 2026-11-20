"""
RAGnarok Developer Workspace (Dev Deck) — Interactive Shell.

Provides a dedicated interactive environment for document ingestion, index rebuilds,
diagnostics, modular component probes, and continuous evaluation benchmarking.
Launched via `rag -dev`, `rag --dev`, or `rag dev`.
"""

from __future__ import annotations

import contextlib
import shlex
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import structlog
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

if TYPE_CHECKING:
    from ragnarok.context.contract import AssembledContextPayload
    from ragnarok.retrieval.contract import ChildResultCandidate

logger = structlog.get_logger(__name__)


class DevShellSession:
    """
    Stateful interactive shell for developer operations and component probes.
    """

    def __init__(self, console: Console | None = None) -> None:
        self.console = console if console is not None else Console()
        self.command_count: int = 0
        self.last_query: str | None = None
        self.last_context_payload: AssembledContextPayload | None = None
        self.last_candidates: list[ChildResultCandidate] = []

    def print_banner(self) -> None:
        """Renders the developer workspace welcome banner with command overview."""
        grid = Table.grid(padding=(0, 2))
        grid.add_column(style="bold yellow", width=18)
        grid.add_column(style="white")

        grid.add_row(
            "Workspace", "[bold white]RAGnarok Developer Workspace (Dev Deck)[/bold white]"
        )
        grid.add_row("Core Operations", "ingest <path> (in) | reindex (rx)")
        grid.add_row("Diagnostics", "doctor (doc) | stats (st)")
        grid.add_row("Inspection", "/source <N> (inspect source text)")
        grid.add_row(
            "Retrieval Query",
            'query "<text>" (q) [--waterfall, --provenance, --strategy, --budget]',
        )
        grid.add_row(
            "Component Probes",
            "parse, chunk, embed, bm25, route, rerank, context, guard, pii, auto, eval",
        )
        grid.add_row("Workspace Controls", "help (/help) | clear (/clear) | exit (quit)")
        grid.add_row(
            "Chat Shell",
            "[dim]Exit and run [bold cyan]rag -c[/bold cyan] for conversational chat[/dim]",
        )

        panel = Panel(
            grid,
            title="[bold yellow]RAGnarok Local Developer Workspace[/bold yellow]",
            subtitle="[dim]Execute dev, query, inspection, or probe commands directly[/dim]",
            border_style="yellow",
            padding=(1, 2),
        )
        self.console.print(panel)

    def print_help(self) -> None:
        """Displays full developer workspace command reference."""
        table = Table(
            title="Developer Workspace Commands",
            show_header=True,
            header_style="bold yellow",
            border_style="bright_black",
        )
        table.add_column("Command", justify="left", width=28)
        table.add_column("Category", justify="center", width=16)
        table.add_column("Description", justify="left")

        table.add_row(
            "ingest <path> (in)",
            "Operations",
            "Ingest & chunk documents into vault with auto-indexing",
        )
        table.add_row(
            "reindex (rx)", "Operations", "Rebuild BM25 and TurboVec indexes from DuckDB"
        )
        table.add_row(
            "doctor (doc)", "Diagnostics", "Run 6-point environment, DB, and accelerator checks"
        )
        table.add_row(
            "stats (st)", "Diagnostics", "Display database record counts and index disk sizes"
        )
        table.add_row(
            "/source <N>, /s <N>", "Inspection", "Inspect full text and metadata for source block [N]"
        )
        table.add_row(
            'query "<text>" (q)',
            "Retrieval",
            "Test full retrieval & generation [--waterfall, --provenance, --strategy, --budget]",
        )
        table.add_row(
            "add <file> [dest] (a)",
            "Staging",
            "Copy external file into docs/ or custom dir for ingestion",
        )
        table.add_row("parse <file>", "Probe", "Inspect layout parser Markdown extraction in memory")
        table.add_row(
            "chunk <file>", "Probe", "Inspect two-stage parent/child boundaries & overlap"
        )
        table.add_row(
            "embed <text>", "Probe", "Measure embedding latency and vector dimensions (384)"
        )
        table.add_row(
            "bm25 <query>", "Probe", "Inspect lexical tokenization and top-5 BM25 matches"
        )
        table.add_row(
            "route <query>", "Probe", "Test Self-RAG gate, Query Rewriter, Adaptive Router"
        )
        table.add_row(
            "rerank <q> <p1> <p2>", "Probe", "Score candidate passages with Cross-Encoder"
        )
        table.add_row(
            "context <query>", "Probe", "Test parent context expansion & token budget gauge"
        )
        table.add_row(
            "guard <query>", "Probe", "Test input prompt injection & length filter guards"
        )
        table.add_row("pii <text>", "Probe", "Test secret, email, credential & IP redaction")
        table.add_row("auto", "Test", "Run automated pytest test suite (unit + integration)")
        table.add_row("eval", "Benchmark", "Run DeepEval 20-case Golden Dataset benchmark suite")
        table.add_row("clear, /clear", "Control", "Clear terminal screen")
        table.add_row("help, /help", "Control", "Display developer command reference")
        table.add_row("exit, quit", "Control", "Exit developer workspace")

        self.console.print(table)

    def _display_single_source(self, index: int) -> None:
        """Displays the full untruncated text of a specific parent context block."""
        if self.last_context_payload and hasattr(self.last_context_payload, "context_blocks"):
            for b in self.last_context_payload.context_blocks:
                if b.source_index == index:
                    from ragnarok.context.formatter import format_heading_path

                    section_str = format_heading_path(b.heading_path)
                    panel = Panel(
                        b.content,
                        title=f"[bold cyan]Source [{b.source_index}]: {b.filename}[/bold cyan]",
                        subtitle=f"[dim]Section: {section_str} | Tokens: {b.token_count} | Parent UUID: {b.parent_chunk_id}[/dim]",
                        border_style="cyan",
                        padding=(1, 2),
                    )
                    self.console.print(panel)
                    return
            max_idx = len(self.last_context_payload.context_blocks)
            self.console.print(
                f"[yellow]Source [{index}] not found in last query context. Available sources: 1 to {max_idx}.[/yellow]"
            )
            return

        # Fallback: inspect parent chunk directly from DuckDB
        try:
            from ragnarok.core.config import load_config
            from ragnarok.storage.database import DatabasePool

            config = load_config()
            with DatabasePool(db_path=config.database.db_path) as pool, pool.connection() as conn, conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT p.parent_index, d.filename, p.heading_path, p.content, p.parent_chunk_id
                    FROM parent_chunks p
                    JOIN documents d ON p.document_id = d.document_id
                    WHERE p.parent_index = ?
                    LIMIT 1;
                    """,
                    (index,),
                )
                row = cur.fetchone()
                if row:
                    p_idx, fname, h_path, content, p_id = row
                    from ragnarok.context.formatter import format_heading_path

                    section_str = format_heading_path(h_path or [])
                    panel = Panel(
                        content,
                        title=f"[bold cyan]Parent Chunk [{p_idx}]: {fname}[/bold cyan]",
                        subtitle=f"[dim]Section: {section_str} | UUID: {p_id}[/dim]",
                        border_style="cyan",
                        padding=(1, 2),
                    )
                    self.console.print(panel)
                    return
                else:
                    self.console.print(
                        f"[yellow]No parent chunk found with index [{index}]. Run a query first or ingest documents.[/yellow]"
                    )
        except Exception as e:
            self.console.print(f"[yellow]Unable to inspect source [{index}]: {e}[/yellow]")

    def dispatch(self, raw_line: str) -> bool:
        """
        Parses and executes a command string within the developer workspace.

        Returns:
            True to continue REPL, False to exit.
        """
        line = raw_line.strip()
        if not line:
            return True

        try:
            tokens = shlex.split(line)
        except ValueError as e:
            self.console.print(f"[bold red]Syntax error in command: {e}[/bold red]")
            return True

        if not tokens:
            return True

        cmd = tokens[0].lower()
        args = tokens[1:]

        # Exit
        if cmd in ("exit", "quit", "/exit", "/quit", ":q"):
            self.console.print("[dim]Exiting developer workspace...[/dim]")
            return False

        self.command_count += 1

        # Help
        if cmd in ("help", "/help", "?"):
            self.print_help()
            return True

        # Clear
        if cmd in ("clear", "/clear", "cls"):
            self.console.clear()
            self.print_banner()
            return True

        # Chat shell transition
        if cmd in ("chat", "c", "/chat"):
            self.console.print("[dim]Transitioning to interactive chat shell...[/dim]")
            if sys.stdin.isatty():
                from ragnarok.cli.repl import run_repl

                run_repl(console=self.console)
                self.print_banner()
            return True

        # Strip optional "rag" prefix so `rag add ...` or `rag ingest ...` works seamlessly
        if cmd == "rag" and args:
            cmd = args[0].lower()
            args = args[1:]

        # Strip optional "test" prefix so both `test parse` and `parse` work
        if cmd in ("test", "t") and args:
            cmd = args[0].lower()
            args = args[1:]

        try:
            # Stage / Add external file
            if cmd in ("add", "a"):
                from ragnarok.cli.main import add_command

                if not args:
                    self.console.print(
                        "[bold red]Usage: add <file_path> [destination] [--force][/bold red]"
                    )
                    return True

                force = False
                non_flag_args: list[str] = []
                for a in args:
                    if a in ("--force", "-f"):
                        force = True
                    else:
                        non_flag_args.append(a)

                if not non_flag_args:
                    self.console.print(
                        "[bold red]Usage: add <file_path> [destination] [--force][/bold red]"
                    )
                    return True

                src_arg = Path(non_flag_args[0])
                dest_arg = Path(non_flag_args[1]) if len(non_flag_args) > 1 else Path("docs")

                with contextlib.suppress(SystemExit):
                    add_command(file_path=src_arg, destination=dest_arg, force=force)

            # Ingestion
            elif cmd in ("ingest", "in"):
                from ragnarok.cli.main import ingest_command

                path = Path(args[0]) if args else Path("docs")
                batch_size = 0
                if "--batch-size" in args:
                    idx = args.index("--batch-size")
                    if idx + 1 < len(args):
                        batch_size = int(args[idx + 1])
                elif "-n" in args:
                    idx = args.index("-n")
                    if idx + 1 < len(args):
                        batch_size = int(args[idx + 1])

                ingest_command(path=path, batch_size=batch_size)

            # Reindex
            elif cmd in ("reindex", "rx"):
                from ragnarok.cli.main import reindex_command

                reindex_command()

            # Doctor
            elif cmd in ("doctor", "doc"):
                from ragnarok.cli.main import doctor_command

                with contextlib.suppress(SystemExit):
                    doctor_command()

            # Stats
            elif cmd in ("stats", "st"):
                from ragnarok.cli.main import stats_command

                stats_command()

            # Source Inspection
            elif cmd in ("/source", "source", "/s", "s"):
                if not args:
                    self.console.print("[yellow]Usage: /source <N> (inspect source block text)[/yellow]")
                    return True
                try:
                    idx = int(args[0])
                    self._display_single_source(idx)
                except ValueError:
                    self.console.print("[yellow]Source index must be an integer (e.g. /source 1)[/yellow]")

            # Query
            elif cmd in ("query", "q"):
                from ragnarok.cli.main import query_command

                if not args:
                    self.console.print(
                        "[bold red]Usage: query <text> [--waterfall] [--provenance] [--strategy <name>] [--budget <n>][/bold red]"
                    )
                    return True

                query_text = args[0]
                show_w = "--waterfall" in args or "-w" in args
                show_p = "--provenance" in args or "-p" in args
                no_rerank = "--no-rerank" in args
                strategy = "hybrid"
                budget = 0

                for flag in ("--strategy", "-s"):
                    if flag in args:
                        idx = args.index(flag)
                        if idx + 1 < len(args):
                            strategy = args[idx + 1]

                for flag in ("--budget", "-b"):
                    if flag in args:
                        idx = args.index(flag)
                        if idx + 1 < len(args):
                            budget = int(args[idx + 1])

                self.last_query = query_text
                with contextlib.suppress(SystemExit):
                    result = query_command(
                        text=query_text,
                        strategy=strategy,
                        no_rerank=no_rerank,
                        budget=budget,
                        show_waterfall=show_w,
                        show_provenance=show_p,
                    )
                    if result:
                        self.last_context_payload, self.last_candidates = result

            # Probes: Parse
            elif cmd == "parse":
                from ragnarok.cli.probes import probe_parse

                if not args:
                    self.console.print("[bold red]Usage: parse <file_path>[/bold red]")
                    return True
                probe_parse(file=Path(args[0]))

            # Probes: Chunk
            elif cmd == "chunk":
                from ragnarok.cli.probes import probe_chunk

                if not args:
                    self.console.print("[bold red]Usage: chunk <file_path>[/bold red]")
                    return True
                probe_chunk(file=Path(args[0]))

            # Probes: Embed
            elif cmd == "embed":
                from ragnarok.cli.probes import probe_embed

                if not args:
                    self.console.print('[bold red]Usage: embed "<text>"[/bold red]')
                    return True
                probe_embed(text=" ".join(args))

            # Probes: BM25
            elif cmd == "bm25":
                from ragnarok.cli.probes import probe_bm25

                if not args:
                    self.console.print('[bold red]Usage: bm25 "<query>"[/bold red]')
                    return True
                probe_bm25(query=" ".join(args))

            # Probes: Route
            elif cmd == "route":
                from ragnarok.cli.probes import probe_route

                if not args:
                    self.console.print('[bold red]Usage: route "<query>"[/bold red]')
                    return True
                probe_route(query=" ".join(args))

            # Probes: Rerank
            elif cmd == "rerank":
                from ragnarok.cli.probes import probe_rerank

                if not args:
                    self.console.print(
                        '[bold red]Usage: rerank "<query>" "<passage1>" "<passage2>"...[/bold red]'
                    )
                    return True
                probe_rerank(query=None, candidates=None, args=args)

            # Probes: Context
            elif cmd == "context":
                from ragnarok.cli.probes import probe_context

                if not args:
                    self.console.print('[bold red]Usage: context "<query>"[/bold red]')
                    return True
                probe_context(query=" ".join(args))

            # Probes: Guard
            elif cmd == "guard":
                from ragnarok.cli.probes import probe_guard

                if not args:
                    self.console.print('[bold red]Usage: guard "<query>"[/bold red]')
                    return True
                probe_guard(query=" ".join(args))

            # Probes: PII
            elif cmd == "pii":
                from ragnarok.cli.probes import probe_pii

                if not args:
                    self.console.print('[bold red]Usage: pii "<text>"[/bold red]')
                    return True
                probe_pii(text=" ".join(args))

            # Probes: Auto (Pytest)
            elif cmd == "auto":
                from ragnarok.cli.probes import probe_auto

                with contextlib.suppress(SystemExit):
                    probe_auto()

            # Probes: Eval (DeepEval)
            elif cmd == "eval":
                from ragnarok.cli.probes import probe_eval

                with contextlib.suppress(SystemExit):
                    probe_eval()

            else:
                self.console.print(
                    f"[yellow]Unknown command: '{cmd}'. Type 'help' for available workspace commands.[/yellow]"
                )

        except SystemExit:
            pass
        except Exception as e:
            self.console.print(f"[bold red]Command '{cmd}' failed: {e}[/bold red]")
            logger.error("Dev shell command failed", command=cmd, error=str(e))

        return True


def run_dev_shell(console: Console | None = None) -> None:
    """
    Main Developer Workspace REPL loop.
    Reads commands, executes isolated operations, and provides feedback until exit.
    """
    session = DevShellSession(console=console)
    session.print_banner()

    try:
        while True:
            try:
                user_input = input("\n🔱 rag-dev> ").strip()
            except EOFError:
                session.console.print("\n[dim]End of input. Exiting developer workspace...[/dim]")
                break

            if not user_input:
                continue

            should_continue = session.dispatch(user_input)
            if not should_continue:
                break

    except KeyboardInterrupt:
        session.console.print("\n[dim]Developer workspace interrupted. Goodbye![/dim]")

    session.console.print(
        f"\n[bold yellow]Developer workspace closed.[/bold yellow] "
        f"Commands executed: {session.command_count}"
    )
