"""
RAGnarok Diagnostic Functions.

Procedural diagnostic functions checking runtime environment, package imports,
configuration validity, filesystem permissions, database connectivity, and accelerators.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path


def check_python_version() -> bool:
    print("Checking Python version...", end=" ")
    v = sys.version_info
    if v.major == 3 and v.minor == 12:
        print(f"[OK] Python {v.major}.{v.minor}.{v.micro}")
        return True
    print(f"[FAIL] Required Python 3.12, found {v.major}.{v.minor}.{v.micro}")
    return False


def check_critical_imports() -> bool:
    print("Checking critical dependencies...", end=" ")
    packages = [
        "docling",
        "psycopg",
        "bm25s",
        "turbovec",
        "sentence_transformers",
        "litellm",
        "pydantic",
        "pydantic_settings",
        "structlog",
        "typer",
        "deepeval",
        "langchain_text_splitters",
    ]
    failed: list[str] = []
    for pkg in packages:
        try:
            __import__(pkg)
        except ImportError:
            failed.append(pkg)

    if not failed:
        print(f"[OK] {len(packages)} packages imported successfully")
        return True
    print(f"[FAIL] Missing packages: {', '.join(failed)}")
    return False


def check_configuration() -> bool:
    print("Checking configuration subsystem...", end=" ")
    try:
        from ragnarok.core.config import load_config

        config = load_config()
        print(f"[OK] Configuration loaded cleanly (App: {config.general.app_name})")
        return True
    except Exception as e:
        print(f"[FAIL] Failed to load configuration: {e}")
        return False


def check_filesystem_paths() -> bool:
    print("Checking runtime filesystem directories...", end=" ")
    required_paths = [
        Path("data/staging"),
        Path("data/raw"),
        Path("data/parsed"),
        Path("data/indexes/bm25"),
        Path("data/indexes/dense"),
    ]
    for p in required_paths:
        p.mkdir(parents=True, exist_ok=True)
        test_file = p / ".write_test"
        try:
            test_file.write_text("ok", encoding="utf-8")
            test_file.unlink()
        except Exception as e:
            print(f"[FAIL] Cannot write to {p}: {e}")
            return False

    print(f"[OK] {len(required_paths)} directories verified and writable")
    return True


def check_database() -> bool:
    print("Checking PostgreSQL connectivity...", end=" ")
    try:
        import psycopg

        db_url = os.getenv("DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/ragnarok")
        with psycopg.connect(db_url) as conn, conn.cursor() as cur:
            cur.execute("SELECT 1;")
            res = cur.fetchone()
            if res and res[0] == 1:
                cur.execute(
                    """
                    SELECT table_name FROM information_schema.tables
                    WHERE table_schema = 'public'
                    AND table_name IN ('documents', 'parent_chunks', 'chunks', 'processing_jobs');
                    """
                )
                tables = [row[0] for row in cur.fetchall()]
                if len(tables) == 4:
                    print(f"[OK] Connected. Found all 4 tables: {sorted(tables)}")
                    return True
                else:
                    print(f"[WARN] Connected. Found {len(tables)}/4 tables: {sorted(tables)}")
                    return False
    except Exception as e:
        print(f"[FAIL] Could not connect to database: {e}")
        return False
    return False


def check_accelerator() -> None:
    print("Detecting hardware acceleration...", end=" ")
    try:
        import torch

        if torch.cuda.is_available():
            device_name = torch.cuda.get_device_name(0)
            print(f"[OK] CUDA available ({device_name})")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            print("[OK] Apple MPS available")
        else:
            print("[NOTE] Running on CPU (standard fallback)")
    except Exception:
        print("[NOTE] Torch not loaded; compute defaults to CPU")
