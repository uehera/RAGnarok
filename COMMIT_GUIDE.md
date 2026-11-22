# 🚀 RAGnarok: Git Staging, Commit & Push Guide

This document provides a clean, production-ready guide to stage, commit, and push all recent changes in **RAGnarok**—including the Web UI single-page application, interactive Architecture Studio (`Arch`), formal evaluation benchmarks, multi-surface guardrails, and the overhauled documentation portal.

---

## 📌 Working Directory

All commands must be executed from the **`RAGnarok/`** repository root:

```bash
cd /home/swayam/Projects/major_project/RAGnarok
```

> **Security Note:** `.env`, `data/`, `models/`, `.venv/`, and `__pycache__/` are already excluded in `.gitignore` and will never be staged or committed.

---

## 📋 Summary of Changes to Commit

| Category | Files / Directories | Key Changes |
| :--- | :--- | :--- |
| **Core & Indexing** | `src/ragnarok/indexing/`, `src/ragnarok/ingestion/`, `src/ragnarok/retrieval/`, `src/ragnarok/context/`, `docs/dataset/` | Bijective `VectorIdMap` (259ms sync), parent-child expansion, sample corpora |
| **Guardrails & Security** | `src/ragnarok/guardrails/` (incl. `context_guard.py`), `src/ragnarok/generation/`, `tests/test_context_screening.py` | In-flight context screening, Shannon entropy gate ($H(X) > 4.8$), stream sanitizer |
| **Web UI & API** | `src/ragnarok/api/` (`app.py`, `routes.py`, `static/`), `tests/test_api.py` | FastAPI backend, SSE token streaming, Evidence Drawer, Architecture Studio modal |
| **Formal Evals** | `src/ragnarok/evals/`, `notes/report/*_benchmark_results.json`, `tests/evals/` | SciFact, NFCorpus, RGB, BIPIA, and MultiHop empirical benchmarks |
| **CLI & Configs** | `src/ragnarok/cli/`, `config.toml`, `pyproject.toml`, `uv.lock`, battle tests | Developer Workspace (`rag dev`), CLI chat, updated dependencies, locked environment |
| **Documentation** | `notes/arc/`, `notes/developer/`, `notes/report/`, `README.md` | Overhauled setup guide, formal algorithms, architecture specs, root README |

---

## 🧭 Choose Your Staging Flow

* **[Option A: Clean Multi-Commit Flow (Recommended)](#option-a-clean-multi-commit-flow-recommended)**: 6 structured, atomic commits following the [Conventional Commits](https://www.conventionalcommits.org/) standard.
* **[Option B: Consolidated Single-Commit Flow](#option-b-consolidated-single-commit-flow)**: 1 single commit staging all verified changes at once.

---

## Option A: Clean Multi-Commit Flow (Recommended)

Run these 6 commits sequentially in your terminal:

### Commit 1: Core Engine, Ingestion, Indexing & Storage
Stages dual indexing, the bijective `VectorIdMap`, contiguous DuckDB window expansion, and test datasets:
```bash
git add src/ragnarok/indexing/ \
        src/ragnarok/ingestion/ \
        src/ragnarok/retrieval/ \
        src/ragnarok/context/ \
        src/ragnarok/core/config.py \
        docs/dataset/

git commit -m "refactor(core): refine dual indexing, parent-child expansion, and incremental sync"
```

---

### Commit 2: Multi-Surface Guardrails & Context Screening
Stages Surface 2.5 In-Flight Context Screening, Shannon entropy analysis, sliding-window stream sanitizer, and security test suites:
```bash
git add src/ragnarok/guardrails/ \
        src/ragnarok/generation/ \
        tests/test_guardrails.py \
        tests/test_generation.py \
        tests/test_context_screening.py

git commit -m "feat(guardrails): add in-flight context guard, Shannon entropy checks, and stream sanitizer"
```

---

### Commit 3: Web UI Application, REST API & Architecture Studio
Stages the single-page application (`index.html`, `style.css`, `app.js`), FastAPI SSE streaming endpoints (`/api/chat/stream`, `/api/query`, `/api/benchmarks`), Evidence Drawer, and Architecture Studio (`Arch`):
```bash
git add src/ragnarok/api/ \
        tests/test_api.py

git commit -m "feat(ui): add single-page web app, SSE streaming chat, Evidence Drawer, and Architecture Studio"
```

---

### Commit 4: Formal Evaluation Benchmarks & Verification Suite
Stages the formal evaluation benchmark runners (`src/ragnarok/evals/`), automated dataset download and execution scripts (`scripts/`), RAGAS test suite, and raw empirical benchmark JSON outputs:
```bash
git add src/ragnarok/evals/ \
        scripts/download_benchmarks.py \
        scripts/run_all_benchmarks.py \
        tests/evals/test_ragas.py \
        notes/report/scifact_benchmark_results.json \
        notes/report/nfcorpus_benchmark_results.json \
        notes/report/rgb_benchmark_results.json \
        notes/report/bipia_benchmark_results.json \
        notes/report/multihop_benchmark_results.json

git commit -m "feat(evals): add formal benchmark suite, dataset downloaders, and test runners"
```

---

### Commit 5: CLI Shells, Dependency Lockfile & Project Configuration
Stages the Developer Workspace (`rag dev`), interactive chat shell (`rag chat`), command probes, `config.toml`, `pyproject.toml`, and `uv.lock`:
```bash
git add src/ragnarok/cli/ \
        config.toml \
        pyproject.toml \
        uv.lock \
        tests/battle/test_adversarial.py \
        tests/test_indexing.py

git commit -m "chore(cli): update dev deck, REPL shells, configurations, and test suites"
```

---

### Commit 6: Complete Documentation Portal & README Overhaul
Stages the overhauled step-by-step setup guide (`setup.md`), architectural blueprints (`notes/arc/`), research reports (`notes/report/`), developer guides, and modern root `README.md`:
```bash
git add notes/arc/ \
        notes/developer/ \
        notes/report/comprehensive_evaluation_report.md \
        notes/report/context_screening_and_evaluation.md \
        notes/report/algorithm.md \
        notes/report/stack.md \
        README.md

git commit -m "docs: overhaul setup guide, architectural blueprints, research reports, and main README"
```

---

## Option B: Consolidated Single-Commit Flow

If you prefer to stage and commit the entire milestone in a single clean commit:

```bash
# 1. Stage all tracked and untracked project files
git add src/ragnarok/ \
        tests/ \
        docs/ \
        notes/ \
        scripts/ \
        config.toml \
        pyproject.toml \
        uv.lock \
        README.md

# 2. Commit everything under a consolidated milestone message
git commit -m "feat: complete Web UI, Architecture Studio, formal evals, and full documentation overhaul"
```

---

## 🚀 Push to Remote

Once your commit(s) are created, push them to GitHub:

```bash
git push origin main
```

*(If you are pushing to an existing tracking branch, plain `git push` works as well).*

---

## 🔍 Post-Push Verification

Confirm that your working tree is clean and synchronized with origin:

```bash
# 1. Verify working directory is clean
git status

# Expected output:
# On branch main
# Your branch is up to date with 'origin/main'.
# nothing to commit, working tree clean

# 2. View recent commit history
git log -n 6 --oneline
```
