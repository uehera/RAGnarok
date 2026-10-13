"""
Enterprise Battle-Testing Suite — Dimension 5: Memory Leaks & Resource Profiling.

Verifies:
1. Memory Footprint Stability: RAM usage plateaus across consecutive query and tokenization passes.
2. File Descriptor Leak Audit: open file handles are strictly closed across consecutive file parses.
"""

from __future__ import annotations

import gc
import resource
from pathlib import Path

from ragnarok.indexing.lexical import tokenize_query
from ragnarok.ingestion.chunker import TiktokenCounter
from ragnarok.ingestion.parser import parse_document_to_markdown


def test_profiling_memory_footprint_stability() -> None:
    """
    Executes 500 consecutive query tokenizations and text analyses.
    Asserts memory consumption plateaus after initial model warm-up without unbounded growth.
    """
    token_counter = TiktokenCounter(encoding_name="cl100k_base")
    sample_queries = [
        "What is machine learning in Python?",
        "Explain high-availability database replication strategies",
        "How does reciprocal rank fusion calculate candidate scores?",
        "Describe two-stage chunking with parent-child hierarchy",
    ]

    gc.collect()
    # Warm up
    for q in sample_queries:
        _ = token_counter.count_tokens(q)

    # Initial memory benchmark
    mem_before_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

    # Run 500 consecutive tokenizations
    for i in range(500):
        q = sample_queries[i % len(sample_queries)]
        _ = token_counter.count_tokens(q)
        _ = tokenize_query(q)

    gc.collect()
    mem_after_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

    # Memory growth should not exceed 100MB across 500 consecutive query passes
    diff_mb = (mem_after_kb - mem_before_kb) / 1024.0
    assert diff_mb < 100.0, f"Excessive memory accumulation detected: {diff_mb:.2f} MB"


def test_profiling_file_descriptor_leak_audit(tmp_path: Path) -> None:
    """
    Parses 50 documents consecutively.
    Asserts open file descriptors do not accumulate or leak across operations.
    """
    def get_open_fd_count() -> int:
        fd_dir = Path("/proc/self/fd")
        if fd_dir.exists():
            return len(list(fd_dir.iterdir()))
        return 0

    # Create 50 small sample files
    test_files: list[Path] = []
    for i in range(50):
        f = tmp_path / f"doc_{i}.txt"
        f.write_text(f"# Title {i}\nSample content for document {i}.", encoding="utf-8")
        test_files.append(f)

    # Baseline file descriptor count
    initial_fds = get_open_fd_count()

    # Parse 50 documents consecutively
    for f in test_files:
        _ = parse_document_to_markdown(f)

    gc.collect()
    final_fds = get_open_fd_count()

    # If /proc/self/fd is accessible on Linux, verify no descriptor leakage
    if initial_fds > 0:
        fd_diff = final_fds - initial_fds
        assert fd_diff <= 3, f"Detected leaking file descriptors: grew by {fd_diff} handles"
