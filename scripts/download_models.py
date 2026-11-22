#!/usr/bin/env python3
"""
RAGnarok Model Pre-Warmer & Downloader.

Pre-downloads and verifies the quantized ONNX models required by RAGnarok:
1. BAAI/bge-small-en-v1.5 (FastEmbed 384-d dense embedding model, ~65 MB)
2. ms-marco-MiniLM-L-12-v2 (FlashRank cross-encoder reranking model, ~33 MB)

Total size: ~98 MB.
Ensures zero cold-start latency and offline readiness.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path


def download_and_verify_models(repo_root: Path) -> None:
    models_dir = repo_root / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    fastembed_cache = models_dir / "fastembed_cache"
    fastembed_cache.mkdir(parents=True, exist_ok=True)

    print("════════════════════════════════════════════════════════════════════")
    print("🚀 Provisioning RAGnarok Local ONNX Neural Models (~98 MB Total)")
    print("════════════════════════════════════════════════════════════════════")

    # 1. FastEmbed Dense Embedder
    print("\n[1/2] Checking FastEmbed Dense Embedding Model (BAAI/bge-small-en-v1.5)...")
    t0 = time.perf_counter()
    try:
        from fastembed import TextEmbedding

        embedder = TextEmbedding(
            model_name="BAAI/bge-small-en-v1.5",
            cache_dir=str(fastembed_cache),
        )
        test_vec = list(embedder.embed(["model pre-warm test vector"]))
        dim = len(test_vec[0])
        print(f"  ✔ BGE-Small ONNX loaded successfully (Dimension: {dim}-d, {time.perf_counter() - t0:.2f}s)")
    except Exception as exc:
        print(f"  ❌ Failed to load FastEmbed model: {exc}", file=sys.stderr)
        raise

    # 2. FlashRank Cross-Encoder Reranker
    print("\n[2/2] Checking FlashRank Cross-Encoder Model (ms-marco-MiniLM-L-12-v2)...")
    t1 = time.perf_counter()
    try:
        from flashrank import Ranker, RerankRequest

        ranker = Ranker(
            model_name="ms-marco-MiniLM-L-12-v2",
            cache_dir=str(models_dir),
        )
        req = RerankRequest(query="test", passages=[{"id": 1, "text": "test passage"}])
        ranker.rerank(req)
        print(f"  ✔ FlashRank ONNX loaded successfully ({time.perf_counter() - t1:.2f}s)")
    except Exception as exc:
        print(f"  ❌ Failed to load FlashRank model: {exc}", file=sys.stderr)
        raise

    print("\n════════════════════════════════════════════════════════════════════")
    print("✅ All ONNX models verified in models/ directory. Zero cold-start latency!")
    print("════════════════════════════════════════════════════════════════════")


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    download_and_verify_models(root)
