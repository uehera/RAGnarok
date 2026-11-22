#!/usr/bin/env python3
"""
RAGnarok Benchmark Dataset Downloader.

Downloads the 5 formal peer-reviewed benchmark datasets into data/benchmarks/:
1. BEIR SciFact (300 test queries, 5,183 scientific abstracts)
2. BEIR NFCorpus (323 test queries, 3,633 medical abstracts)
3. RGB Benchmark (Chen et al., ACL 2024, 100 noise/rejection queries)
4. Microsoft BIPIA (Indirect Prompt Injection test suite)
5. MultiHop-RAG (Multi-document reasoning corpus & queries)

Total download size: ~30 MB (fast, uses Python standard library).
"""

from __future__ import annotations

import argparse
import os
import sys
import urllib.request
from pathlib import Path


DATASET_SOURCES: dict[str, dict[str, str]] = {
    "scifact": {
        "corpus.parquet": "https://huggingface.co/datasets/BeIR/scifact/resolve/main/corpus/corpus-00000-of-00001.parquet",
        "queries.parquet": "https://huggingface.co/datasets/BeIR/scifact/resolve/main/queries/queries-00000-of-00001.parquet",
        "test_qrels.tsv": "https://huggingface.co/datasets/BeIR/scifact-qrels/resolve/main/test.tsv",
    },
    "nfcorpus": {
        "corpus.parquet": "https://huggingface.co/datasets/BeIR/nfcorpus/resolve/main/corpus/corpus-00000-of-00001.parquet",
        "queries.parquet": "https://huggingface.co/datasets/BeIR/nfcorpus/resolve/main/queries/queries-00000-of-00001.parquet",
        "test_qrels.tsv": "https://huggingface.co/datasets/BeIR/nfcorpus-qrels/resolve/main/test.tsv",
    },
    "rgb": {
        "en.json": "https://raw.githubusercontent.com/chen700564/RGB/master/data/en.json",
    },
    "bipia": {
        "text_attack_test.json": "https://raw.githubusercontent.com/microsoft/BIPIA/main/benchmark/text_attack_test.json",
        "code_attack_test.json": "https://raw.githubusercontent.com/microsoft/BIPIA/main/benchmark/code_attack_test.json",
    },
    "multihop_rag": {
        "corpus.json": "https://huggingface.co/datasets/yixuantt/MultiHopRAG/resolve/main/corpus.json",
        "MultiHopRAG.json": "https://huggingface.co/datasets/yixuantt/MultiHopRAG/resolve/main/MultiHopRAG.json",
    },
}


def download_file(url: str, dest_path: Path, force: bool = False) -> None:
    """Downloads a remote file with progress reporting."""
    if dest_path.exists() and not force:
        size_kb = dest_path.stat().st_size / 1024
        print(f"  [EXISTS] {dest_path.name} ({size_kb:.1f} KB) - skipping")
        return

    dest_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"  [DOWNLOADING] {dest_path.name} from {url}...")

    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"})
    with urllib.request.urlopen(req, timeout=30) as resp, open(dest_path, "wb") as out_file:
        chunk_size = 64 * 1024
        downloaded = 0
        while True:
            chunk = resp.read(chunk_size)
            if not chunk:
                break
            out_file.write(chunk)
            downloaded += len(chunk)

    print(f"  [SAVED] {dest_path.name} ({downloaded / 1024:.1f} KB)")


def download_all_benchmarks(base_dir: Path, force: bool = False, dataset_filter: str | None = None) -> None:
    """Downloads all benchmark datasets into base_dir/data/benchmarks."""
    benchmarks_dir = base_dir / "data" / "benchmarks"
    print(f"Target benchmarks directory: {benchmarks_dir}")

    for dataset_name, files in DATASET_SOURCES.items():
        if dataset_filter and dataset_filter.lower() != dataset_name.lower():
            continue

        print(f"\n📦 Processing dataset: {dataset_name.upper()}")
        target_dir = benchmarks_dir / dataset_name
        for filename, url in files.items():
            dest = target_dir / filename
            try:
                download_file(url, dest, force=force)
            except Exception as exc:
                print(f"  ❌ Error downloading {filename}: {exc}", file=sys.stderr)

    print("\n✅ Benchmark dataset provisioning complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download RAGnarok benchmark datasets.")
    parser.add_argument("--force", action="store_true", help="Force re-download even if files exist.")
    parser.add_argument("--dataset", type=str, default=None, help="Download only a specific dataset (scifact, nfcorpus, rgb, bipia, multihop_rag).")
    args = parser.parse_args()

    # Determine project root (where data/ resides)
    repo_root = Path(__file__).resolve().parents[1]
    download_all_benchmarks(repo_root, force=args.force, dataset_filter=args.dataset)
