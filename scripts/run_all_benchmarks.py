#!/usr/bin/env python3
"""
RAGnarok Master Benchmark Suite Runner.

Executes all 5 formal empirical benchmarks sequentially and updates their
corresponding JSON scorecards in notes/report/:
1. SciFact (BEIR Scientific Claim IR & Retrieval Ablations)
2. NFCorpus (BEIR Biomedical Cross-Domain IR)
3. RGB (ACL 2024 Grounding, Noise Robustness & Negative Rejection)
4. BIPIA (Microsoft Research Indirect Prompt Injection Defense)
5. MultiHop-RAG (Multi-Document Retrieval & Context Expansion)

Usage:
    uv run python scripts/run_all_benchmarks.py
    uv run python scripts/run_all_benchmarks.py --suite scifact
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from rich.console import Console
from rich.panel import Panel

console = Console()


def run_scifact() -> None:
    console.print("\n[bold cyan]═══ 1/5: Running BEIR SciFact IR Benchmark ═══[/bold cyan]")
    from ragnarok.evals.scifact_bench import SciFactBenchmarkRunner

    runner = SciFactBenchmarkRunner()
    runner.load_dataset()
    runner.build_indexes()
    metrics = runner.evaluate()
    runner.print_results(metrics)
    export_file = Path("notes/report/scifact_benchmark_results.json")
    runner.export_results(metrics, export_file)
    console.print(f"[bold green]✔ SciFact benchmark saved to {export_file}[/bold green]")


def run_nfcorpus() -> None:
    console.print("\n[bold cyan]═══ 2/5: Running BEIR NFCorpus Biomedical Benchmark ═══[/bold cyan]")
    from ragnarok.evals.nfcorpus_bench import NFCorpusBenchmarkRunner

    runner = NFCorpusBenchmarkRunner()
    runner.load_dataset()
    runner.build_indexes()
    metrics = runner.evaluate()
    runner.print_results(metrics)
    export_file = Path("notes/report/nfcorpus_benchmark_results.json")
    runner.export_results(metrics, export_file)
    console.print(f"[bold green]✔ NFCorpus benchmark saved to {export_file}[/bold green]")


def run_rgb() -> None:
    console.print("\n[bold cyan]═══ 3/5: Running ACL 2024 RGB Grounding Benchmark ═══[/bold cyan]")
    from ragnarok.evals.rgb_bench import RGBBenchmarkRunner
    import json

    runner = RGBBenchmarkRunner()
    runner.load_dataset()
    report = runner.evaluate_slice(n_noise=50, n_negative=50)
    report.print_rich_table()

    export_file = Path("notes/report/rgb_benchmark_results.json")
    export_file.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "benchmark": "RGB (ACL 2024)",
        "total_cases": report.total_cases,
        "noise_robustness_accuracy": round(report.noise_robustness_accuracy, 4),
        "negative_rejection_rate": round(report.negative_rejection_rate, 4),
        "ghost_citation_rate": round(report.ghost_citation_rate, 4),
        "mean_latency_ms": round(report.mean_latency_ms, 2),
        "p95_latency_ms": round(report.p95_latency_ms, 2),
    }
    with export_file.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    console.print(f"[bold green]✔ RGB benchmark saved to {export_file}[/bold green]")


def run_bipia() -> None:
    console.print("\n[bold cyan]═══ 4/5: Running Microsoft BIPIA Security Benchmark ═══[/bold cyan]")
    from ragnarok.evals.bipia_bench import BIPIABenchmarkRunner
    import json

    runner = BIPIABenchmarkRunner()
    report = runner.evaluate(n_attacks=75, n_benign=25)
    report.print_rich_table()

    export_file = Path("notes/report/bipia_benchmark_results.json")
    export_file.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "benchmark": "BIPIA Indirect Prompt Injection",
        "total_attacks_tested": report.total_attacks_tested,
        "total_benign_tested": report.total_benign_tested,
        "attack_detection_rate": round(report.attack_detection_rate, 4),
        "database_contamination_rate": round(report.database_contamination_rate, 4),
        "secret_leakage_rate": round(report.secret_leakage_rate, 4),
        "false_positive_rate": round(report.false_positive_rate, 4),
        "mean_guardrail_latency_ms": round(report.mean_guardrail_latency_ms, 2),
    }
    with export_file.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    console.print(f"[bold green]✔ BIPIA benchmark saved to {export_file}[/bold green]")


def run_multihop() -> None:
    console.print("\n[bold cyan]═══ 5/5: Running MultiHop-RAG Reasoning Benchmark ═══[/bold cyan]")
    from ragnarok.evals.multihop_bench import MultiHopBenchmarkRunner

    runner = MultiHopBenchmarkRunner()
    runner.load_data()
    runner.build_indexes()
    report = runner.evaluate(k_list=[5, 10, 20])
    runner.print_rich_table(report)
    export_file = Path("notes/report/multihop_benchmark_results.json")
    runner.export_results(report, export_file)
    console.print(f"[bold green]✔ MultiHop-RAG benchmark saved to {export_file}[/bold green]")


SUITES = {
    "scifact": run_scifact,
    "nfcorpus": run_nfcorpus,
    "rgb": run_rgb,
    "bipia": run_bipia,
    "multihop": run_multihop,
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Execute RAGnarok benchmark suite.")
    parser.add_argument(
        "--suite",
        choices=["all", "scifact", "nfcorpus", "rgb", "bipia", "multihop"],
        default="all",
        help="Select which benchmark to run (default: all)",
    )
    args = parser.parse_args()

    # Ensure running from repo root
    repo_root = Path(__file__).resolve().parents[1]
    import os
    os.chdir(repo_root)

    t0 = time.perf_counter()
    console.print(
        Panel.fit(
            "[bold white]🚀 RAGnarok Empirical Benchmark Execution Suite[/bold white]\n"
            "[dim]Evaluates retrieval quality, grounding SLA, security gateways, and reasoning.[/dim]",
            border_style="magenta",
        )
    )

    if args.suite == "all":
        for name, fn in SUITES.items():
            fn()
    else:
        SUITES[args.suite]()

    total_time = time.perf_counter() - t0
    console.print(
        Panel.fit(
            f"[bold green]🎉 All benchmarks completed in {total_time:.1f}s![/bold green]\n"
            f"[dim]Updated JSON scorecards written to notes/report/.[/dim]\n"
            f"[bold yellow]Launch 'rag ui' to view the updated metrics in the Web Dashboard.[/bold yellow]",
            border_style="green",
        )
    )


if __name__ == "__main__":
    main()
