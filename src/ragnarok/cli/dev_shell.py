"""
RAGnarok Developer Workspace (Dev Deck) — Interactive Shell.

Provides a dedicated interactive environment for document ingestion, index rebuilds,
diagnostics, modular component probes, and continuous evaluation benchmarking.
Launched via `rag -dev`, `rag --dev`, or `rag dev`.
"""

from __future__ import annotations

import contextlib
import shlex
from pathlib import Path

import structlog
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

logger = structlog.get_logger(__name__)


class DevShellSession:
    """
    Stateful interactive shell for developer operations and component probes.
    """

    def __init__(self, console: Console | None = None) -> None:
        self.console = console if console is not None else Console()
        self.command_count: int = 0

    def print_banner(self) -> None:
        """Renders the developer workspace welcome banner with command overview."""
        grid = Table.grid(padding=(0, 2))
        grid.add_column(style="bold yellow", width=18)
        grid.add_column(style="white")

        grid.add_row(
            "Workspace", "[bold white]RAGnarok Developer Workspace (Dev Deck)[/bold white]"
        )
        grid.add_row("Stage & Ingest", "add <file> [dest] (a) | ingest <path> (in) | reindex (rx)")
        grid.add_row("Diagnostics", "doctor (doc) | stats (st)")
        grid.add_row(
            "Component Probes",
            "parse, chunk, embed, bm25, route, rerank, context, guard, pii",
        )
        grid.add_row(
            "Test & Benchmarks", "auto (pytest suite) | eval (DeepEval 20-case golden dataset)"
        )
        grid.add_row(
            "Retrieval Query",
            'query "<text>" (q) [--waterfall, --provenance, --strategy, --budget]',
        )
        grid.add_row("Workspace Controls", "help (/help) | clear (/clear) | exit (quit)")
        grid.add_row(
            "Chat Shell",
            "[dim]Exit and run [bold cyan]rag -c[/bold cyan] for conversational chat[/dim]",
        )

        panel = Panel(
            grid,
            title="[bold yellow]RAGnarok Local Developer Workspace[/bold yellow]",
            subtitle="[dim]Execute any dev, probe, or test command directly without typing 'rag'[/dim]",
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
            "add <file> [dest] (a)",
            "Staging",
            "Copy external file into docs/ or custom dir for ingestion",
        )
        table.add_row(
            "ingest <path> (in)",
            "Ingestion",
            "Ingest & chunk documents into vault with auto-indexing",
        )
        table.add_row(
            "reindex (rx)", "Indexing", "Rebuild BM25 and TurboVec indexes from PostgreSQL"
        )
        table.add_row(
            "doctor (doc)", "Diagnostics", "Run 6-point environment, DB, and accelerator checks"
        )
        table.add_row(
            "stats (st)", "Diagnostics", "Display database record counts and index disk sizes"
        )
        table.add_row(
            "query <text> (q)", "Retrieval", "Test full retrieval & generation (-w, -p, -s, -b)"
        )
        table.add_row("parse <file>", "Probe", "Inspect Docling Markdown extraction in memory")
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
        table.add_row("exit, quit", "Control", "Exit developer workspace")

        self.console.print(table)

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

        # Help
        if cmd in ("help", "/help", "?"):
            self.print_help()
            return True

        # Clear
        if cmd in ("clear", "/clear", "cls"):
            self.console.clear()
            self.print_banner()
            return True

        # Chat shell warning
        if cmd in ("chat", "c", "/chat"):
            self.console.print(
                "[bold cyan]Notice:[/bold cyan] [dim]'chat' runs in the dedicated chat shell.[/dim]\n"
                "[yellow]To open the conversational chat shell with real-time streaming, exit and run:[/yellow] "
                "[bold green]rag -c[/bold green]\n"
            )
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

            # Query
            elif cmd in ("query", "q"):
                from ragnarok.cli.main import query_command

                if not args:
                    self.console.print(
                        "[bold red]Usage: query <text> [--waterfall] [--provenance] [--strategy <name>][/bold red]"
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

                query_command(
                    text=query_text,
                    strategy=strategy,
                    no_rerank=no_rerank,
                    budget=budget,
                    show_waterfall=show_w,
                    show_provenance=show_p,
                )

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
