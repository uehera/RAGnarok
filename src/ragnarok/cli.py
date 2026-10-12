"""
RAGnarok CLI Admin Deck.

Command-line interface built with Typer and Rich.
Provides administrative commands, diagnostics (doctor), and pipeline probes.
"""

from __future__ import annotations

import typer
from rich.console import Console

from ragnarok.core.diagnostics import (
    check_accelerator,
    check_configuration,
    check_critical_imports,
    check_database,
    check_filesystem_paths,
    check_python_version,
)

app = typer.Typer(
    name="ragnarok",
    help="RAGnarok CLI Admin Deck & Diagnostic Suite",
    no_args_is_help=True,
)
console = Console()

@app.command(name="doctor")
def doctor() -> None:
    """
    Run comprehensive environment, dependency, configuration, and database diagnostics.
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
        console.print(
            "[bold red]Setup status: INCOMPLETE — Resolve failures above[/bold red]"
        )
        raise typer.Exit(code=1)

if __name__ == "__main__":
    app()
