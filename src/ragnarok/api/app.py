"""
FastAPI Application Factory for RAGnarok.

Serves RESTful APIs, SSE event streams, and static single-page interface.
"""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from ragnarok.api.routes import router
from ragnarok.core.config import load_config


def create_app() -> FastAPI:
    """Creates and configures the FastAPI application instance."""
    config = load_config()
    app = FastAPI(
        title="RAGnarok API",
        description="Security-Aware Context Screening & Evidence Selection for RAG Systems",
        version="0.1.0",
        docs_url="/docs",
        redoc_url="/redoc",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.middleware("http")
    async def add_no_cache_headers(request, call_next):
        response = await call_next(request)
        if request.url.path == "/" or request.url.path.startswith("/static/"):
            response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate, max-age=0"
            response.headers["Pragma"] = "no-cache"
            response.headers["Expires"] = "0"
        return response

    app.include_router(router)

    static_dir = Path(__file__).parent / "static"
    if static_dir.exists():
        app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

    @app.get("/", include_in_schema=False)
    def index():
        index_file = static_dir / "index.html"
        if index_file.exists():
            return FileResponse(index_file)
        return {
            "status": "online",
            "message": "RAGnarok API is active. Open /docs for Swagger interactive documentation.",
        }

    return app


# Root application instance for uvicorn
app = create_app()
