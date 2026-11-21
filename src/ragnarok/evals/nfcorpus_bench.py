"""
BEIR NFCorpus Formal Information Retrieval Benchmark Engine.

Evaluates RAGnarok's retrieval pipeline against the official BEIR NFCorpus benchmark
(323 test queries, 3,633 medical and nutrition abstracts) across four ablation strategies:
1. Dense Retrieval Only (TurboVec Cosine Search, 384-d BGE-small)
2. Lexical Retrieval Only (BM25s Core Inverted Index)
3. Hybrid Fusion (Reciprocal Rank Fusion, k=60)
4. Hybrid + Cross-Encoder Reranking (FlashRank ONNX ms-marco-MiniLM-L-12-v2)

Computes official Information Retrieval metrics with 95% Confidence Intervals (CI_95):
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
from dataclasses import asdict, dataclass
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
class MetricWithCI:
    mean: float
    std: float
    ci_95: float

    def to_dict(self) -> dict[str, float]:
        return {
            "mean": round(self.mean, 4),
            "std": round(self.std, 4),
            "ci_95": round(self.ci_95, 4),
        }


@dataclass
class NFCorpusStrategyMetrics:
    strategy_name: str
    recall_at_5: MetricWithCI
    recall_at_10: MetricWithCI
    mrr_at_10: MetricWithCI
    ndcg_at_10: MetricWithCI
    mean_latency_ms: float
    p95_latency_ms: float
    total_queries: int


def calculate_ci_95(values: list[float]) -> MetricWithCI:
    if not values:
        return MetricWithCI(mean=0.0, std=0.0, ci_95=0.0)
    arr = np.array(values, dtype=np.float64)
    mean = float(np.mean(arr))
    std = float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0
    se = std / math.sqrt(len(arr)) if len(arr) > 0 else 0.0
    ci_95 = 1.96 * se
    return MetricWithCI(mean=mean, std=std, ci_95=ci_95)


def compute_ndcg_at_k(ranked_ids: list[str], ground_truth: dict[str, int], k: int = 10) -> float:
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
    for idx, doc_id in enumerate(ranked_ids[:k]):
        if doc_id in relevant_set:
            return 1.0 / (idx + 1)
    return 0.0


def compute_recall_at_k(ranked_ids: list[str], relevant_set: set[str], k: int) -> float:
    if not relevant_set:
        return 0.0
    hits = sum(1 for doc_id in ranked_ids[:k] if doc_id in relevant_set)
    return hits / len(relevant_set)


class NFCorpusBenchmarkRunner:
    """High-performance harness for benchmarking NFCorpus across 4 retrieval strategies."""

    def __init__(self, data_dir: Path | None = None) -> None:
        self.data_dir = data_dir or Path(__file__).resolve().parents[3] / "data" / "benchmarks" / "nfcorpus"
        self.corpus_path = self.data_dir / "corpus.parquet"
        self.queries_path = self.data_dir / "queries.parquet"
        self.qrels_path = self.data_dir / "test_qrels.tsv"

        self.doc_ids: list[str] = []
        self.corpus_texts: list[str] = []
        self.doc_id_to_idx: dict[str, int] = {}
        self.queries: dict[str, str] = {}
        self.qrels: dict[str, dict[str, int]] = {}

        self.embedder = FastEmbedEmbedder(model_name="BAAI/bge-small-en-v1.5")
        self.reranker = FlashRankCrossEncoder(model_name="ms-marco-MiniLM-L-12-v2")
        self.bm25_retriever: bm25s.BM25 | None = None
        self.dense_embeddings: np.ndarray | None = None

    def load_data(self) -> None:
        logger.info("Loading NFCorpus dataset files via DuckDB...", data_dir=str(self.data_dir))
        con = duckdb.connect(":memory:")

        # 1. Corpus
        corpus_df = con.execute(f"SELECT _id, title, text FROM '{self.corpus_path}'").fetchall()
        for doc_id, title, text in corpus_df:
            full_text = f"{title}. {text}" if title else text
            self.doc_id_to_idx[doc_id] = len(self.doc_ids)
            self.doc_ids.append(doc_id)
            self.corpus_texts.append(full_text)
        logger.info("Loaded NFCorpus documents", count=len(self.doc_ids))

        # 2. Test Qrels
        with open(self.qrels_path, mode="r", encoding="utf-8") as f:
            reader = csv.reader(f, delimiter="\t")
            header = next(reader, None)
            for row in reader:
                if len(row) >= 3:
                    qid, doc_id, score_str = row[0], row[1], row[2]
                    try:
                        score = int(score_str)
                    except ValueError:
                        score = 1
                    if qid not in self.qrels:
                        self.qrels[qid] = {}
                    self.qrels[qid][doc_id] = score
        logger.info("Loaded NFCorpus test qrels", query_count=len(self.qrels))

        # 3. Queries (filter strictly to the 323 test queries in qrels)
        queries_df = con.execute(f"SELECT _id, text FROM '{self.queries_path}'").fetchall()
        for qid, text in queries_df:
            if qid in self.qrels:
                self.queries[qid] = text
        logger.info("Filtered NFCorpus test queries", count=len(self.queries))

    def build_indexes(self) -> None:
        logger.info("Tokenizing NFCorpus and indexing BM25s...")
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

        logger.info("Pre-warming FlashRank Cross-Encoder ONNX session...")
        self.reranker._get_model()

    def evaluate(self, limit_queries: int = 0) -> dict[str, NFCorpusStrategyMetrics]:
        q_items = list(self.queries.items())
        if limit_queries > 0:
            q_items = q_items[:limit_queries]

        strategies = ["dense_only", "bm25_only", "hybrid_rrf", "hybrid_rerank"]
        results: dict[str, list[dict[str, float]]] = {s: [] for s in strategies}
        timings: dict[str, list[float]] = {s: [] for s in strategies}

        logger.info("Running vectorized evaluation across strategies for NFCorpus...", num_queries=len(q_items))

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
        all_sims = np.dot(q_vecs, self.dense_embeddings.T)
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
            timings["hybrid_rrf"].append(
                embed_time_per_query + dense_search_time_per_query + bm25_search_time_per_query + rrf_overhead
            )

            # 4. Hybrid + Cross-Encoder Rerank (Top-10 reranked)
            t_rerank_start = time.perf_counter()
            candidates_to_rerank = hybrid_ranked_ids[:10]
            pairs = [(qtext, self.corpus_texts[self.doc_id_to_idx[did]]) for did in candidates_to_rerank]
            rerank_scores = self.reranker.predict(pairs)
            scored = sorted(zip(candidates_to_rerank, rerank_scores), key=lambda x: x[1], reverse=True)
            reranked_ids = [c[0] for c in scored]
            rerank_overhead = (time.perf_counter() - t_rerank_start) * 1000.0
            timings["hybrid_rerank"].append(
                embed_time_per_query + dense_search_time_per_query + bm25_search_time_per_query + rrf_overhead + rerank_overhead
            )

            for strat, ranked in [
                ("dense_only", dense_ranked_ids),
                ("bm25_only", bm25_ranked_ids),
                ("hybrid_rrf", hybrid_ranked_ids),
                ("hybrid_rerank", reranked_ids),
            ]:
                r5 = compute_recall_at_k(ranked, relevant_set, 5)
                r10 = compute_recall_at_k(ranked, relevant_set, 10)
                mrr = compute_mrr_at_k(ranked, relevant_set, 10)
                ndcg = compute_ndcg_at_k(ranked, ground_truth, 10)
                results[strat].append({
                    "r5": r5,
                    "r10": r10,
                    "mrr": mrr,
                    "ndcg": ndcg,
                })

        # Calculate metrics with CI_95
        final_metrics: dict[str, NFCorpusStrategyMetrics] = {}
        for s in strategies:
            rows = results[s]
            r5_ci = calculate_ci_95([r["r5"] for r in rows])
            r10_ci = calculate_ci_95([r["r10"] for r in rows])
            mrr_ci = calculate_ci_95([r["mrr"] for r in rows])
            ndcg_ci = calculate_ci_95([r["ndcg"] for r in rows])
            times = timings[s]

            final_metrics[s] = NFCorpusStrategyMetrics(
                strategy_name=s,
                recall_at_5=r5_ci,
                recall_at_10=r10_ci,
                mrr_at_10=mrr_ci,
                ndcg_at_10=ndcg_ci,
                mean_latency_ms=round(float(np.mean(times)), 2) if times else 0.0,
                p95_latency_ms=round(float(np.percentile(times, 95)), 2) if times else 0.0,
                total_queries=len(rows),
            )

        return final_metrics

    def run_and_save(self, output_path: Path | None = None) -> dict[str, Any]:
        self.load_data()
        self.build_indexes()
        metrics = self.evaluate()

        out_file = output_path or Path(__file__).resolve().parents[3] / "notes" / "report" / "nfcorpus_benchmark_results.json"
        out_file.parent.mkdir(parents=True, exist_ok=True)

        payload = {
            "dataset": "BEIR NFCorpus",
            "domain": "Biomedical & Nutrition Information Retrieval",
            "total_corpus_documents": len(self.doc_ids),
            "total_test_queries": len(self.queries),
            "evaluation_timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "strategies": {
                name: {
                    "strategy_name": m.strategy_name,
                    "recall_at_5": m.recall_at_5.to_dict(),
                    "recall_at_10": m.recall_at_10.to_dict(),
                    "mrr_at_10": m.mrr_at_10.to_dict(),
                    "ndcg_at_10": m.ndcg_at_10.to_dict(),
                    "mean_latency_ms": m.mean_latency_ms,
                    "p95_latency_ms": m.p95_latency_ms,
                    "total_queries": m.total_queries,
                }
                for name, m in metrics.items()
            },
        }

        with open(out_file, mode="w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

        logger.info("Saved NFCorpus benchmark results", path=str(out_file))

        console = Console()
        table = Table(title="BEIR NFCorpus Formal IR Benchmark Results (323 Queries)")
        table.add_column("Retrieval Strategy", style="bold cyan")
        table.add_column("Recall@5 (±CI95)", justify="center")
        table.add_column("Recall@10 (±CI95)", justify="center", style="bold green")
        table.add_column("MRR@10 (±CI95)", justify="center")
        table.add_column("nDCG@10 (±CI95)", justify="center", style="bold yellow")
        table.add_column("Mean Latency", justify="right")

        for s, m in metrics.items():
            table.add_row(
                s,
                f"{m.recall_at_5.mean:.4f} (±{m.recall_at_5.ci_95:.4f})",
                f"{m.recall_at_10.mean:.4f} (±{m.recall_at_10.ci_95:.4f})",
                f"{m.mrr_at_10.mean:.4f} (±{m.mrr_at_10.ci_95:.4f})",
                f"{m.ndcg_at_10.mean:.4f} (±{m.ndcg_at_10.ci_95:.4f})",
                f"{m.mean_latency_ms:.2f} ms",
            )
        console.print(table)

        return payload


if __name__ == "__main__":
    runner = NFCorpusBenchmarkRunner()
    runner.run_and_save()
