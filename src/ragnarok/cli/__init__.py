"""
RAGnarok CLI Admin Deck Package.

Exports the Typer application instance and modular subcommands:
- Top-level operational commands (query, ingest, reindex, stats, doctor)
- Component probe deck (test parse, test chunk, test embed, etc.)
- Visual metrics deck (waterfall, budget gauge, provenance table)
- Interactive developer REPL shell (chat)
"""

from __future__ import annotations

from ragnarok.cli.main import app

__all__ = ["app"]
