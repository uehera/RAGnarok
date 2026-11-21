"""
RAGnarok FastAPI Web Application Subsystem.

Provides REST and SSE endpoints alongside classic modern HTML/CSS interface.
"""

from __future__ import annotations

from ragnarok.api.app import create_app

__all__ = ["create_app"]
