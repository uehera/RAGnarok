"""
Unit and Integration Tests for RAGAS Formal Evaluation Runner.

Verifies:
1. Benchmark dataset conversion to Hugging Face / Ragas Dataset format.
2. Formal Quality Gate assertion across the 20 Golden Dataset cases.
3. Accurate computation of Context Precision, Recall, Faithfulness, and Relevancy.
4. Terminal Rich table reporting.
"""

from __future__ import annotations

import pytest
from rich.console import Console

from ragnarok.core.config import load_config
from ragnarok.evals.ragas_bench import RagasBenchmarkRunner


def test_ragas_dataset_conversion() -> None:
    """Verifies that internal golden dataset cases convert to standard Ragas Dataset format."""
    runner = RagasBenchmarkRunner()
    dataset = runner.to_ragas_dataset()

    assert dataset is not None
    assert "question" in dataset.column_names
    assert "contexts" in dataset.column_names
    assert "answer" in dataset.column_names
    assert "ground_truth" in dataset.column_names
    assert len(dataset) >= 20


def test_ragas_benchmark_quality_gates() -> None:
    """
    Verifies that all 20 Golden Dataset benchmark cases satisfy formal RAGAS quality gates.
    """
    config = load_config()
    thresholds = config.evals.thresholds
    runner = RagasBenchmarkRunner(config=config)

    report = runner.run_deterministic_eval()

    assert report.total_cases == 20
    assert report.passed_cases == 20
    assert report.all_passed

    # Verify mean metric thresholds
    assert report.mean_context_precision >= thresholds.context_precision
    assert report.mean_context_recall >= thresholds.context_recall
    assert report.mean_faithfulness >= thresholds.faithfulness
    assert report.mean_answer_relevancy >= thresholds.answer_relevancy

    # Verify individual case results
    for case_res in report.results:
        assert case_res.passed, f"Case {case_res.case_id} failed quality gate"
        assert case_res.context_precision >= thresholds.context_precision
        assert case_res.context_recall >= thresholds.context_recall
        assert case_res.faithfulness >= thresholds.faithfulness
        assert case_res.answer_relevancy >= thresholds.answer_relevancy


def test_ragas_rich_table_rendering() -> None:
    """Verifies that benchmark results render cleanly to console."""
    runner = RagasBenchmarkRunner()
    report = runner.run_deterministic_eval()

    console = Console(record=True)
    report.print_rich_table(console=console)
    output = console.export_text()

    assert "RAGAS Formal Benchmark Quality Matrix" in output
    assert "eval" in output
    assert "PASS" in output
    assert "RAGAS Evaluation Summary" in output
