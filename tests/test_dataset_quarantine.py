"""
End-to-End Dataset Ingestion & Quarantine Verification Test Suite.

Tests the full ingestion pipeline on benchmark datasets:
- Clean technical documents from docs/dataset/
- Adversarial / injection documents from docs/false/

Verifies:
1. Quarantine Mode:
   - Clean documents are parsed, chunked, and persisted to DuckDB.
   - Poisoned documents are quarantined with JSON forensic sidecars in data/quarantine/.
   - Poisoned documents produce exactly 0 chunks in DuckDB and search indexes.
2. Sanitize Mode:
   - Poisoned documents are safely defanged into [DEFANGED INJECTION DIRECTIVE: <TAG>].
3. Retrieval Invariants:
   - Legitimate queries retrieve clean parent-child chunks with high BM25 scores.
   - Malicious attack directives are absent from search indexes.
"""

from __future__ import annotations

import json
from pathlib import Path
from uuid import UUID

import pytest

from ragnarok.core.config import AppConfig
from ragnarok.core.models import ChildChunk
from ragnarok.guardrails.document_guard import DocumentSecurityGuard
from ragnarok.indexing.lexical import LexicalIndex
from ragnarok.ingestion.chunker import TwoStageChunker
from ragnarok.ingestion.pipeline import IngestionPipeline
from ragnarok.retrieval.lexical_search import LexicalSearcher
from ragnarok.storage.database import DatabasePool
from ragnarok.storage.repository import RelationalRepository


CLEAN_DATASET_DIR = Path("docs/dataset")
FALSE_DATASET_DIR = Path("docs/false")

CLEAN_FILES = [
    "clean_01_api_gateway.md",
    "clean_02_database_indexing.md",
    "clean_03_raft_consensus.md",
    "clean_04_tls_security.md",
    "clean_05_event_driven_architecture.md",
]
POISONED_FILES = [
    "poisoned_01_hidden_html_override.md",
    "poisoned_02_developer_mode_hijack.md",
    "poisoned_03_chatml_delimiter_escape.md",
    "poisoned_04_system_prompt_override.md",
    "poisoned_05_role_spoofing.md",
]


@pytest.fixture
def test_env(tmp_path: Path):
    """Sets up an isolated database and filesystem environment for ingestion testing."""
    db_path = tmp_path / "test_quarantine.duckdb"
    pool = DatabasePool(db_path=str(db_path))
    pool.init_schema(Path("scripts/init_schema.sql"))
    repo = RelationalRepository(pool)

    staging_dir = tmp_path / "staging"
    raw_dir = tmp_path / "raw"
    parsed_dir = tmp_path / "parsed"
    quarantine_dir = tmp_path / "quarantine"

    for d in [staging_dir, raw_dir, parsed_dir, quarantine_dir]:
        d.mkdir(parents=True, exist_ok=True)

    config = AppConfig()
    config.ingestion.staging_directory = staging_dir
    config.ingestion.raw_directory = raw_dir
    config.ingestion.parsed_directory = parsed_dir
    config.guardrails.enabled = True
    config.guardrails.enable_document_guard = True
    config.guardrails.document_injection_action = "quarantine"
    config.guardrails.quarantine_directory = quarantine_dir

    try:
        yield {
            "pool": pool,
            "repo": repo,
            "config": config,
            "quarantine_dir": quarantine_dir,
        }
    finally:
        pool.close()


def test_mixed_dataset_ingestion_quarantine_mode(test_env: dict):
    """
    Ingests the mixed dataset with document_injection_action = 'quarantine'.
    Asserts:
    - 5 clean files generate parent and child chunks in DuckDB.
    - 5 poisoned files are isolated to quarantine_dir with JSON metadata sidecars.
    - 0 chunks exist in DuckDB for any of the 5 poisoned files.
    """
    repo: RelationalRepository = test_env["repo"]
    config: AppConfig = test_env["config"]
    quarantine_dir: Path = test_env["quarantine_dir"]

    guard = DocumentSecurityGuard(config=config.guardrails)
    chunker = TwoStageChunker(config=config.chunking, document_guard=guard)
    pipeline = IngestionPipeline(repository=repo, config=config, chunker=chunker)

    # 1. Ingest clean documents from docs/dataset/
    clean_docs = []
    for clean_name in CLEAN_FILES:
        path = CLEAN_DATASET_DIR / clean_name
        doc, is_dup = pipeline.process_file(path)
        assert doc is not None
        assert is_dup is False
        clean_docs.append(doc)

        # Assert chunks exist in DuckDB
        chunks = repo.query_chunk_ids_by_metadata(document_id=doc.document_id)
        assert len(chunks) > 0, f"Expected chunks for clean file {clean_name}"

    # 2. Ingest poisoned documents from docs/false/
    poisoned_docs = []
    for poisoned_name in POISONED_FILES:
        path = FALSE_DATASET_DIR / poisoned_name
        doc, is_dup = pipeline.process_file(path)
        assert doc is not None
        poisoned_docs.append(doc)

        # Zero contamination: Assert EXACTLY 0 chunks in DuckDB
        chunks = repo.query_chunk_ids_by_metadata(document_id=doc.document_id)
        assert len(chunks) == 0, f"Poisoned file {poisoned_name} must produce 0 chunks in DuckDB!"

    # 3. Assert quarantine directory artifacts
    quarantine_md_files = list(quarantine_dir.glob("*.md"))
    quarantine_json_files = list(quarantine_dir.glob("*.json"))

    assert len(quarantine_md_files) == 5, f"Expected 5 quarantined markdown files, found {len(quarantine_md_files)}"
    assert len(quarantine_json_files) == 5, f"Expected 5 forensic JSON reports, found {len(quarantine_json_files)}"

    # Check forensic metadata contents
    tags_found = set()
    for jf in quarantine_json_files:
        data = json.loads(jf.read_text(encoding="utf-8"))
        assert data["status"] == "QUARANTINED"
        assert data["action_taken"] == "quarantine"
        assert len(data["sha256"]) == 64
        assert len(data["threat_excerpts"]) > 0
        for tag in data["injections_detected"]:
            tags_found.add(tag)

    # Verify that the primary attack vectors were flagged
    assert "HIDDEN_HTML_INJECTION" in tags_found
    assert "PERSONA_HIJACKING" in tags_found
    assert "SPECIAL_TOKEN_INJECTION" in tags_found
    assert "IGNORE_INSTRUCTIONS" in tags_found
    assert "ROLE_HEADER_INJECTION" in tags_found


def test_mixed_dataset_ingestion_sanitize_mode(tmp_path: Path):
    """
    Ingests the 5 poisoned documents with document_injection_action = 'sanitize'.
    Asserts:
    - Directives are replaced with [DEFANGED INJECTION DIRECTIVE: <TAG>].
    - Chunks are persisted to DuckDB in defanged form.
    """
    db_path = tmp_path / "test_sanitize.duckdb"
    pool = DatabasePool(db_path=str(db_path))
    pool.init_schema(Path("scripts/init_schema.sql"))
    repo = RelationalRepository(pool)

    staging_dir = tmp_path / "staging"
    raw_dir = tmp_path / "raw"
    parsed_dir = tmp_path / "parsed"

    for d in [staging_dir, raw_dir, parsed_dir]:
        d.mkdir(parents=True, exist_ok=True)

    config = AppConfig()
    config.ingestion.staging_directory = staging_dir
    config.ingestion.raw_directory = raw_dir
    config.ingestion.parsed_directory = parsed_dir
    config.guardrails.enabled = True
    config.guardrails.enable_document_guard = True
    config.guardrails.document_injection_action = "sanitize"

    guard = DocumentSecurityGuard(config=config.guardrails)
    chunker = TwoStageChunker(config=config.chunking, document_guard=guard)
    pipeline = IngestionPipeline(repository=repo, config=config, chunker=chunker)

    try:
        for poisoned_name in POISONED_FILES:
            path = FALSE_DATASET_DIR / poisoned_name
            doc, _ = pipeline.process_file(path)
            assert doc is not None

            # Parent chunks must exist and contain defanged markers
            with pool.connection() as conn, conn.cursor() as cur:
                cur.execute("SELECT content FROM parent_chunks WHERE document_id = ?;", (doc.document_id,))
                rows = cur.fetchall()
                assert len(rows) > 0
                all_content = " ".join(r[0] for r in rows)
                assert "[DEFANGED INJECTION DIRECTIVE:" in all_content
                assert "<!-- SYSTEM:" not in all_content
                assert "<|im_start|>" not in all_content
    finally:
        pool.close()


def test_clean_document_search_and_index_cleanliness(test_env: dict):
    """
    Tests that clean documents can be retrieved accurately while quarantined attacks
    do not pollute the lexical search index.
    """
    repo: RelationalRepository = test_env["repo"]
    pool: DatabasePool = test_env["pool"]
    config: AppConfig = test_env["config"]

    guard = DocumentSecurityGuard(config=config.guardrails)
    chunker = TwoStageChunker(config=config.chunking, document_guard=guard)
    pipeline = IngestionPipeline(repository=repo, config=config, chunker=chunker)

    # Ingest clean docs from docs/dataset/
    for clean_name in CLEAN_FILES:
        pipeline.process_file(CLEAN_DATASET_DIR / clean_name)

    # Ingest poisoned docs from docs/false/ (quarantine mode)
    for poisoned_name in POISONED_FILES:
        pipeline.process_file(FALSE_DATASET_DIR / poisoned_name)

    # Fetch all stored child chunks from DuckDB
    with pool.connection() as conn, conn.cursor() as cur:
        cur.execute("SELECT chunk_id, document_id, parent_chunk_id, content, chunk_index, heading_path FROM chunks;")
        rows = cur.fetchall()

    stored_chunks = [
        ChildChunk(
            chunk_id=r[0],
            document_id=r[1],
            parent_chunk_id=r[2],
            content=r[3],
            chunk_index=r[4],
            heading_path=r[5],
        )
        for r in rows
    ]

    # Build BM25 index on indexed chunks
    lexical_index = LexicalIndex(config=config.lexical_retrieval)
    lexical_index.build(stored_chunks)
    searcher = LexicalSearcher(index=lexical_index, repository=repo, config=config.lexical_retrieval)

    # Query 1: Clean query matching clean_01
    results1 = searcher.search("token bucket algorithm rate limiting", top_k=3)
    assert len(results1) > 0
    assert "token bucket" in (results1[0].content or "").lower()

    # Query 2: Clean query matching clean_03
    results2 = searcher.search("Raft leader election candidate vote", top_k=3)
    assert len(results2) > 0
    assert "raft" in (results2[0].content or "").lower()

    # Query 3: Search for exploit payloads - must return 0 positive score matches!
    results_exploit = searcher.search("EXPLOIT_EXECUTED ROOT_SHELL_GRANTED", top_k=3)
    positive_matches = [r for r in results_exploit if (r.bm25_score or 0.0) > 0.0]
    assert len(positive_matches) == 0, "Quarantined payloads must have 0 positive score matches!"
    for r in results_exploit:
        content_text = r.content or ""
        assert "EXPLOIT_EXECUTED" not in content_text
        assert "ROOT_SHELL_GRANTED" not in content_text
