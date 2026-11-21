"""
BEIR SciFact Formal Information Retrieval Benchmark Engine.

Evaluates RAGnarok's retrieval pipeline against the official BEIR SciFact benchmark
(300 test queries, 5,183 scientific abstracts) across four ablation strategies:
1. Dense Retrieval Only (TurboVec Cosine Search, 384-d BGE-small)
2. Lexical Retrieval Only (BM25s Core Inverted Index)
3. Hybrid Fusion (Reciprocal Rank Fusion, k=60)
4. Hybrid + Cross-Encoder Reranking (FlashRank ONNX ms-marco-MiniLM-L-12-v2)

Computes official Information Retrieval metrics:
- Recall@5 & Recall@10
- MRR@10 (Mean Reciprocal Rank)
- nDCG@10 (Normalized Discounted Cumulative Gain)
- Latency (mean latency per query in milliseconds)
"""

from __future__ import annotations

import csv
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import structlog
from rich.console import Console
from rich.table import Table

from ragnarok.indexing.dense import FastEmbedEmbedder
from ragnarok.indexing.lexical import tokenize_corpus
from ragnarok.retrieval.reranker import FlashRankCrossEncoder
import bm25s

logger = structlog.get_logger(__name__)


@dataclass
class IRStrategyMetrics:
    """Standard Information Retrieval metrics for a specific retrieval strategy."""

    strategy_name: str
    recall_at_5: float
    recall_at_10: float
    mrr_at_10: float
    ndcg_at_10: float
    mean_latency_ms: float
    total_queries: int


def compute_ndcg_at_k(ranked_ids: list[str], ground_truth: dict[str, int], k: int = 10) -> float:
    """Computes Normalized Discounted Cumulative Gain (nDCG@K)."""
    dcg = 0.0
    for idx, doc_id in enumerate(ranked_ids[:k]):
        rel = ground_truth.get(doc_id, 0)
        if rel > 0:
            dcg += (2**rel - 1) / math.log2(idx + 2)

    ideal_rels = sorted(ground_truth.values(), reverse=True)[:k]
    idcg = 0.0
    for idx, rel in enumerate(ideal_rels):
        if rel > 0:
            idcg += (2**rel - 1) / math.log2(idx + 2)

    return dcg / idcg if idcg > 0 else 0.0


def compute_mrr_at_k(ranked_ids: list[str], relevant_set: set[str], k: int = 10) -> float:
    """Computes Mean Reciprocal Rank (MRR@K)."""
    for idx, doc_id in enumerate(ranked_ids[:k]):
        if doc_id in relevant_set:
            return 1.0 / (idx + 1)
    return 0.0


def compute_recall_at_k(ranked_ids: list[str], relevant_set: set[str], k: int) -> float:
    """Computes Recall@K."""
    if not relevant_set:
        return 0.0
    hits = sum(1 for doc_id in ranked_ids[:k] if doc_id in relevant_set)
    return hits / len(relevant_set)


class SciFactBenchmarkRunner:
    """High-performance harness for benchmarking SciFact across 4 retrieval strategies."""

    def __init__(self, data_dir: Path | None = None) -> None:
        self.data_dir = data_dir or Path(__file__).resolve().parents[3] / "data" / "benchmarks" / "scifact"
        self.corpus_path = self.data_dir / "corpus.parquet"
        self.queries_path = self.data_dir / "queries.parquet"
        self.qrels_path = self.data_dir / "test_qrels.tsv"

        self.doc_ids: list[str] = []
        self.corpus_texts: list[str] = []
        self.doc_id_to_idx: dict[str, int] = {}

        self.queries: dict[str, str] = {}
        self.qrels: dict[str, dict[str, int]] = {}

        self.bm25_retriever: Any = None
        self.dense_embeddings: np.ndarray | None = None
        self.embedder = FastEmbedEmbedder()
        self.reranker = FlashRankCrossEncoder()

    def load_dataset(self) -> None:
        """Loads corpus, test queries, and ground truth relevance judgements."""
        conn = duckdb.connect()

        logger.info("Loading SciFact corpus from parquet...")
        rows = conn.execute(
            f"SELECT _id, title, text FROM read_parquet('{self.corpus_path}')"
        ).fetchall()
        for idx, (doc_id, title, text) in enumerate(rows):
            doc_id_str = str(doc_id)
            full_text = f"{title} {text}".strip()
            self.doc_ids.append(doc_id_str)
            self.corpus_texts.append(full_text)
            self.doc_id_to_idx[doc_id_str] = idx

        logger.info("Loading SciFact test qrels...")
        with self.qrels_path.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                qid = row["query-id"]
                cid = row["corpus-id"]
                score = int(row["score"])
                if qid not in self.qrels:
                    self.qrels[qid] = {}
                self.qrels[qid][cid] = score

        test_qids = set(self.qrels.keys())
        logger.info("Loading SciFact test queries...", total_test_qids=len(test_qids))
        q_rows = conn.execute(
            f"SELECT _id, text FROM read_parquet('{self.queries_path}')"
        ).fetchall()
        for qid, qtext in q_rows:
            qid_str = str(qid)
            if qid_str in test_qids:
                self.queries[qid_str] = str(qtext)

        logger.info(
            "SciFact dataset loaded successfully",
            corpus_size=len(self.doc_ids),
            test_queries=len(self.queries),
        )

    def build_indexes(self) -> None:
        """Loads or builds BM25s index and dense embeddings."""
        logger.info("Tokenizing corpus and indexing BM25s...")
        corpus_tokens = tokenize_corpus(self.corpus_texts)
        self.bm25_retriever = bm25s.BM25()
        self.bm25_retriever.index(corpus_tokens)

        embed_cache_file = self.data_dir / "corpus_embeddings.npy"
        if embed_cache_file.exists():
            logger.info("Loading precomputed dense embeddings from cache...", path=str(embed_cache_file))
            self.dense_embeddings = np.load(embed_cache_file)
        else:
            logger.info("Computing dense vectors with FastEmbed BGE-small...")
            t0 = time.perf_counter()
            vectors = []
            batch_size = 128
            for i in range(0, len(self.corpus_texts), batch_size):
                batch = self.corpus_texts[i : i + batch_size]
                batch_vecs = self.embedder.embed_texts(batch)
                vectors.append(batch_vecs)
            self.dense_embeddings = np.vstack(vectors)
            norms = np.linalg.norm(self.dense_embeddings, axis=1, keepdims=True)
            norms[norms == 0.0] = 1.0
            self.dense_embeddings = self.dense_embeddings / norms
            np.save(embed_cache_file, self.dense_embeddings)
            logger.info("Dense embeddings computed and cached", duration_s=round(time.perf_counter() - t0, 2))

        # Pre-warm Cross-Encoder model session
        logger.info("Pre-warming FlashRank Cross-Encoder ONNX session...")
        self.reranker._get_model()

    def evaluate(self, limit_queries: int = 0) -> dict[str, IRStrategyMetrics]:
        """Runs the 4 retrieval strategies and returns comparative metrics."""
        q_items = list(self.queries.items())
        if limit_queries > 0:
            q_items = q_items[:limit_queries]

        strategies = ["dense_only", "bm25_only", "hybrid_rrf", "hybrid_rerank"]
        results: dict[str, list[dict[str, float]]] = {s: [] for s in strategies}
        timings: dict[str, list[float]] = {s: [] for s in strategies}

        logger.info("Running vectorized evaluation across strategies...", num_queries=len(q_items))

        # Vectorized Query Embedding
        query_texts = [qtext for _, qtext in q_items]
        t_embed_start = time.perf_counter()
        q_vecs = self.embedder.embed_texts(query_texts)
        q_norms = np.linalg.norm(q_vecs, axis=1, keepdims=True)
        q_norms[q_norms == 0.0] = 1.0
        q_vecs = q_vecs / q_norms
        embed_time_per_query = ((time.perf_counter() - t_embed_start) / len(q_items)) * 1000.0

        # Vectorized Dense Search (Dot-Product across all queries in one matrix multiply)
        t_dense_start = time.perf_counter()
        all_sims = np.dot(q_vecs, self.dense_embeddings.T)  # (N_queries, N_corpus)
        dense_search_time_per_query = ((time.perf_counter() - t_dense_start) / len(q_items)) * 1000.0

        # Vectorized BM25 Search
        t_bm25_start = time.perf_counter()
        all_q_tokens = tokenize_corpus(query_texts)
        all_bm25_top_indices, _ = self.bm25_retriever.retrieve(all_q_tokens, k=50)
        bm25_search_time_per_query = ((time.perf_counter() - t_bm25_start) / len(q_items)) * 1000.0

        k_rrf = 60

        for q_idx, (qid, qtext) in enumerate(q_items):
            ground_truth = self.qrels.get(qid, {})
            relevant_set = {cid for cid, s in ground_truth.items() if s > 0}
            if not relevant_set:
                continue

            # 1. Dense Only Top-50
            sims = all_sims[q_idx]
            dense_top_indices = np.argpartition(-sims, 50)[:50]
            dense_top_indices = dense_top_indices[np.argsort(-sims[dense_top_indices])]
            dense_ranked_ids = [self.doc_ids[idx] for idx in dense_top_indices]
            timings["dense_only"].append(embed_time_per_query + dense_search_time_per_query)

            # 2. BM25 Only Top-50
            bm25_ranked_ids = [self.doc_ids[idx] for idx in all_bm25_top_indices[q_idx]]
            timings["bm25_only"].append(bm25_search_time_per_query)

            # 3. Hybrid RRF Top-50
            t_rrf_start = time.perf_counter()
            rrf_scores: dict[str, float] = {}
            for rank, doc_id in enumerate(bm25_ranked_ids, start=1):
                rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + 1.0 / (k_rrf + rank)
            for rank, doc_id in enumerate(dense_ranked_ids, start=1):
                rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + 1.0 / (k_rrf + rank)
            hybrid_ranked_ids = sorted(rrf_scores.keys(), key=lambda d: rrf_scores[d], reverse=True)[:50]
            rrf_overhead = (time.perf_counter() - t_rrf_start) * 1000.0
            timings["hybrid_rrf"].append(embed_time_per_query + dense_search_time_per_query + bm25_search_time_per_query + rrf_overhead)

            # 4. Hybrid + Cross-Encoder Rerank (Top-10 reranked, consistent with production RAGnarok)
            t_rerank_start = time.perf_counter()
            candidates_to_rerank = hybrid_ranked_ids[:10]
            pairs = [(qtext, self.corpus_texts[self.doc_id_to_idx[did]]) for did in candidates_to_rerank]
            rerank_scores = self.reranker.predict(pairs)
            scored = sorted(zip(candidates_to_rerank, rerank_scores), key=lambda x: x[1], reverse=True)
            reranked_ids = [c[0] for c in scored]
            reranked_ids.extend([c for c in hybrid_ranked_ids if c not in set(candidates_to_rerank)])
            rerank_overhead = (time.perf_counter() - t_rerank_start) * 1000.0
            timings["hybrid_rerank"].append(timings["hybrid_rrf"][-1] + rerank_overhead)

            for strat, ranked in [
                ("dense_only", dense_ranked_ids),
                ("bm25_only", bm25_ranked_ids),
                ("hybrid_rrf", hybrid_ranked_ids),
                ("hybrid_rerank", reranked_ids),
            ]:
                results[strat].append({
                    "recall@5": compute_recall_at_k(ranked, relevant_set, k=5),
                    "recall@10": compute_recall_at_k(ranked, relevant_set, k=10),
                    "mrr@10": compute_mrr_at_k(ranked, relevant_set, k=10),
                    "ndcg@10": compute_ndcg_at_k(ranked, ground_truth, k=10),
                })

        final_metrics: dict[str, IRStrategyMetrics] = {}
        for strat in strategies:
            n = len(results[strat])
            final_metrics[strat] = IRStrategyMetrics(
                strategy_name=strat,
                recall_at_5=float(np.mean([m["recall@5"] for m in results[strat]])),
                recall_at_10=float(np.mean([m["recall@10"] for m in results[strat]])),
                mrr_at_10=float(np.mean([m["mrr@10"] for m in results[strat]])),
                ndcg_at_10=float(np.mean([m["ndcg@10"] for m in results[strat]])),
                mean_latency_ms=float(np.mean(timings[strat])),
                total_queries=n,
            )

        return final_metrics

    def print_results(self, metrics: dict[str, IRStrategyMetrics], console: Console | None = None) -> None:
        """Renders publication-ready comparison table."""
        c = console or Console()
        table = Table(
            title="[bold cyan]BEIR SciFact Formal IR Benchmark (300 Test Queries)[/bold cyan]",
            show_header=True,
            header_style="bold magenta",
        )
        table.add_column("Retrieval Strategy", justify="left", width=28)
        table.add_column("Recall@5", justify="center", width=12)
        table.add_column("Recall@10", justify="center", width=12)
        table.add_column("MRR@10", justify="center", width=12)
        table.add_column("nDCG@10", justify="center", width=12)
        table.add_column("Latency (ms)", justify="center", width=14)

        display_names = {
            "dense_only": "Dense Vector (TurboVec Cosine)",
            "bm25_only": "BM25s Lexical Index",
            "hybrid_rrf": "Hybrid Fusion (RRF k=60)",
            "hybrid_rerank": "[bold green]Hybrid + Reranker (RAGnarok)[/bold green]",
        }

        for strat, m in metrics.items():
            name = display_names.get(strat, strat)
            table.add_row(
                name,
                f"{m.recall_at_5:.3f}",
                f"{m.recall_at_10:.3f}",
                f"{m.mrr_at_10:.3f}",
                f"{m.ndcg_at_10:.3f}",
                f"{m.mean_latency_ms:.1f}ms",
            )

        c.print()
        c.print(table)
        c.print()

    def export_results(self, metrics: dict[str, IRStrategyMetrics], output_path: Path) -> None:
        """Exports metrics as structured JSON for reports and paper submissions."""
        payload = {
            "benchmark": "BEIR SciFact",
            "total_test_queries": len(self.queries),
            "corpus_size": len(self.doc_ids),
            "strategies": {
                strat: {
                    "recall_at_5": round(m.recall_at_5, 4),
                    "recall_at_10": round(m.recall_at_10, 4),
                    "mrr_at_10": round(m.mrr_at_10, 4),
                    "ndcg_at_10": round(m.ndcg_at_10, 4),
                    "mean_latency_ms": round(m.mean_latency_ms, 2),
                }
                for strat, m in metrics.items()
            },
        }
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        logger.info("Exported SciFact benchmark report", path=str(output_path))


if __name__ == "__main__":
    runner = SciFactBenchmarkRunner()
    runner.load_dataset()
    runner.build_indexes()
    metrics = runner.evaluate()
    runner.print_results(metrics)
    export_file = Path("notes/report/scifact_benchmark_results.json")
    runner.export_results(metrics, export_file)
