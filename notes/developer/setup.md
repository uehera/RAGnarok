# Developer Setup & Complete Environment Initialization

> **Target File:** `notes/developer/setup.md`  
> **Supported Environments:** Linux (Ubuntu 22.04 / 24.04 LTS), WSL 2 (recommended for Windows), and Windows 10/11 Native (Git Bash / PowerShell).  
> **Scope:** Definitive step-by-step guide to installing dependencies, configuring secrets, provisioning local neural ONNX models, seeding search indexes, running benchmark suites, and launching interfaces.

---

## 1. Setup Architecture Overview

```mermaid
flowchart TD
    A["1. System Prerequisites<br>(Python 3.12, Astral uv, Git)"] --> B["2. Clone Repository & Disable CRLF<br>(git config core.autocrlf false)"]
    B --> C["3. Install Virtualenv via UV<br>(uv sync --link-mode=copy)"]
    C --> D["4. Configure Secrets & Parameters<br>(.env & config.toml)"]
    D --> E["5. Provision Neural Models<br>(uv run python scripts/download_models.py)"]
    E --> F["6. Run System Diagnostics<br>(uv run rag doctor)"]
    F --> G["7. Ingest Knowledge Corpus<br>(uv run rag ingest ./docs/dataset/)"]
    G --> H["8. (Optional) Run Benchmark Suite<br>(download_benchmarks.py & run_all_benchmarks.py)"]
    H --> I1["Launch Web UI Dashboard<br>(uv run rag ui -> http://127.0.0.1:8000)"]
    H --> I2["Launch Terminal Chat Shell<br>(uv run rag chat)"]
    H --> I3["Launch Developer Workspace<br>(uv run rag dev)"]
```

---

## 2. Platform Compatibility & Path Matrix

| Platform | Recommended Shell | Virtualenv Python Path | Storage Directory Style | Recommended Working Directory |
| :--- | :--- | :--- | :--- | :--- |
| **WSL 2 (Ubuntu)** | `/bin/bash` or `zsh` | `.venv/bin/python` | Linux ext4 forward slashes (`./data`) | `~/Projects/major_project/RAGnarok` *(native Linux ext4)* |
| **Linux Native** | `/bin/bash` or `zsh` | `.venv/bin/python` | Linux ext4 forward slashes (`./data`) | `~/Projects/RAGnarok` |
| **Windows Native** | Git Bash (`mintty`) or PowerShell | `.venv/Scripts/python.exe` | Standard forward slashes (`./data`) | `C:/Users/<User>/Projects/RAGnarok` |

> [!IMPORTANT]
> **WSL File System Performance Rule:** Never place or run RAGnarok inside a Windows mount path such as `/mnt/c/Users/...` when using WSL 2. Cross-boundary 9P translation creates a **10× to 50× I/O performance bottleneck** on DuckDB table scans and memory-mapped index reads. Always clone and execute inside your native Linux home directory (`~/...`).

---

## 3. Step-by-Step Installation Walkthrough

### Step 1: Install System Prerequisites & `uv` Package Manager

RAGnarok standardizes on **Astral `uv`**, an ultra-fast Rust-based package resolver that installs virtual environments and dependencies deterministically in seconds.

#### Option A: Linux & WSL 2 (Ubuntu / Debian)
```bash
# 1. Update system packages and install build tools
sudo apt update && sudo apt install -y curl git build-essential

# 2. Install standalone uv binary
curl -LsSf https://astral.sh/uv/install.sh | sh

# 3. Source environment into active shell session
source "$HOME/.local/bin/env"

# 4. Verify installation
uv --version
# Expected: uv 0.4.x or higher
```

#### Option B: Windows Native (Git Bash or PowerShell)
In **Git Bash**:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
echo 'export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"' >> ~/.bashrc
uv --version
```
In **PowerShell** (Run as Administrator):
```powershell
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
# Restart your terminal and verify:
uv --version
```

---

### Step 2: Clone Repository & Normalize Git Line Endings

Windows environments often default to converting Unix line feeds (`\n`) to Windows carriage returns (`\r\n`). In RAGnarok, carriage-return characters corrupt cryptographic SHA-256 deduplication and trigger false positives in quarantine hashing (`staging.py`).

```bash
# 1. Clone the repository
git clone https://github.com/uehera/RAGnarok.git
cd RAGnarok

# 2. Strictly disable CRLF conversion for this repository
git config --local core.autocrlf false

# 3. Reset all checked-out files to clean Unix LF line endings
git checkout -- .
```

---

### Step 3: Install Python 3.12 & Synchronize Dependencies

Using `--link-mode=copy` ensures package installation succeeds cleanly across cross-filesystem mounts, virtual drives, and NTFS junctions without hardlink permission failures:

```bash
uv sync --link-mode=copy
```

**Expected Output:**
```text
Resolved 68 packages in 420ms
Prepared 68 packages in 1.85s
Installed 68 packages in 2.10s
 + bm25s==0.1.10
 + duckdb==2.0.0
 + fastembed==0.9.1
 + fastapi==0.115.0
 + litellm==1.40.0
 + pymupdf==2.1.0
 + turbovec==0.1.0
 + uvicorn==0.30.6
```

---

### Step 4: Configure Environment Secrets (`.env`) & Engine Parameters (`config.toml`)

#### 1. Configure `.env`
Copy the example environment configuration template:
```bash
cp .env.example .env
```

Open `.env` in your text editor and configure your preferred inference provider. RAGnarok natively supports Groq, OpenAI, Anthropic, and local Ollama:

```ini
# ==============================================================================
# RAGnarok Runtime Secrets
# ==============================================================================

# Primary Cloud Inference Provider (Groq delivers sub-250ms TTFT)
GROQ_API_KEY="gsk_your_groq_api_key_here"

# Alternative Cloud Provider: OpenAI
# OPENAI_API_KEY="sk-proj-your_openai_api_key_here"

# Alternative Local Provider: Ollama (fully offline, zero external API keys)
# OLLAMA_API_BASE="http://localhost:11434"

# Storage & Runtime Environment Overrides
RAGNAROK_DATA_DIR="./data"
RAGNAROK_ENV="development"
```

#### 2. Review Engine Settings in `config.toml`
Default operational parameters are predefined in `config.toml`:

```toml
[retrieval]
strategy = "hybrid"         # "hybrid", "dense", or "bm25"
candidate_pool_size = 50    # Top-K candidate pool retrieved before RRF fusion
final_top_k = 10            # Top chunks provided to the generator
rrf_k = 60                  # Reciprocal Rank Fusion rank dampening constant

[reranker]
enabled = false             # Cross-Encoder toggle (disabled by default for sub-40ms SLA)
model_name = "ms-marco-MiniLM-L-12-v2"

[guardrails]
enabled = true
token_limit = 1024          # Rejection threshold for context-stuffing DoS
fast_path_enabled = true    # Sub-millisecond regex & Shannon entropy checks
speculative_window = 35     # Character sliding buffer for real-time secret masking
```

---

### Step 5: Provision Local Neural ONNX Models (`scripts/download_models.py`)

To ensure zero cold-start latency and 100% offline runtime execution, pre-download the quantized ONNX neural weights for dense embeddings and Cross-Encoder reranking:

```bash
uv run python scripts/download_models.py
```

**What this downloads (~98 MB total):**
1. **FastEmbed Dense Embedding Model (`BAAI/bge-small-en-v1.5`, ~67 MB):** Quantized ONNX weights generating 384-dimensional dense vectors cached in `models/fastembed_cache/`.
2. **FlashRank Cross-Encoder Model (`ms-marco-MiniLM-L-12-v2`, ~31 MB):** ONNX reranker weights cached in `models/ms-marco-MiniLM-L-12-v2/`.

**Expected Output:**
```text
============================================================
🔱 RAGnarok Local Model Provisioning
============================================================
[1/2] Pre-warming FastEmbed ONNX model (BAAI/bge-small-en-v1.5)...
  ✓ FastEmbed initialized and verified.
[2/2] Downloading FlashRank Cross-Encoder ONNX (ms-marco-MiniLM-L-12-v2)...
  ✓ FlashRank reranker initialized and verified.
============================================================
✨ All ONNX model weights successfully provisioned in models/!
```

---

### Step 6: Run System Diagnostics (`rag doctor`)

Verify that the local Python environment, DuckDB database permissions, FastEmbed ONNX engine, and storage directories are fully operational:

```bash
uv run rag doctor
# Shorthand alias:
uv run rag doc
```

**Expected Diagnostic Report:**
```text
╭───────────────────── RAGnarok System Diagnostics ─────────────────────╮
│ Component              Status      Details                            │
├───────────────────────────────────────────────────────────────────────┤
│ Operating System       [PASS]      Linux 6.6.x (WSL2) / Windows x86_64│
│ Python Version         [PASS]      Python 3.12.x                      │
│ Configuration          [PASS]      config.toml & .env loaded          │
│ DuckDB Relational      [PASS]      data/ragnarok.duckdb accessible    │
│ Storage Directories    [PASS]      data/indexes, data/quarantine (RW) │
│ FastEmbed ONNX Engine  [PASS]      BAAI/bge-small-en-v1.5 initialized │
│ Input Guardrails       [PASS]      Deterministic Fast-Path active     │
╰───────────────────────────────────────────────────────────────────────╯
[SUCCESS] All systems operational. Ready to ingest documents or query.
```

---

### Step 7: Ingest Documentation & Seed Search Indexes

Seed the DuckDB relational vault and compile the BM25s inverted index and TurboVec dense embeddings:

```bash
uv run rag ingest ./docs/dataset/
# Shorthand alias:
uv run rag in ./docs/dataset/
```

**Expected Ingestion Output:**
```text
RAGnarok Ingestion Pipeline
Scanning: ./docs/dataset/
  ├── Clean Documents:       12
  └── Quarantined Threats:    0
Ingesting into DuckDB and compiling dual indexes...
[SUCCESS] Ingested 12 documents into DuckDB, TurboVec, and BM25s in 3.42s.
```

---

### Step 8: (Optional) Provision Benchmark Datasets & Run Evaluations

If you are setting up RAGnarok on a new machine or a higher-performing workstation/server, you can provision the official benchmark corpora and rerun all formal benchmarks:

```bash
# 1. Download official benchmark datasets (~30 MB total into data/benchmarks/)
python scripts/download_benchmarks.py

# 2. Execute the master benchmark suite across all 5 evaluation layers
uv run python scripts/run_all_benchmarks.py
```

**What this executes:**
* **BEIR SciFact:** 300 test queries against 5,183 scientific abstracts.
* **BEIR NFCorpus:** 323 test queries against 3,633 medical abstracts.
* **ACL 2024 RGB:** 100 test cases for noise robustness and negative refusal.
* **Microsoft BIPIA:** 100 cases of indirect prompt injection defense.
* **MultiHop-RAG:** 200 multi-document reasoning cases.

> [!TIP]
> **Automatic Web UI Metric Updates:** Running `run_all_benchmarks.py` overwrites the JSON scorecards in `notes/report/`. When you open the Web UI at `http://127.0.0.1:8000` and navigate to the **METRICS** tab, the Top KPI cards, Master 5-Layer Table, and Pareto Frontier chart dynamically read these files and update immediately.

---

### Step 9: Launch Application Interfaces

RAGnarok provides multiple execution interfaces tailored for different user workflows:

#### 1. Web Application & Operations Dashboard (Recommended)
Launch the modern 3-column web interface featuring real-time SSE streaming, interactive Evidence Drawer, and Architecture Studio (`Arch`):
```bash
uv run rag ui
# Shorthand alias:
uv run rag serve
```
Open **`http://127.0.0.1:8000`** in your browser.

#### 2. Interactive Terminal Chat Shell
For conversing directly from the command line with multi-turn memory:
```bash
uv run rag chat
```

#### 3. Developer Workspace Deck
For systems engineering, pipeline latency profiling (`--waterfall`), and component inspection:
```bash
uv run rag dev
```

#### 4. Direct Terminal Query (Non-Interactive Headless Mode)
For scripting, CI checks, or quick lookups:
```bash
uv run rag query "How does parent-child context expansion work?"
```

---

## 4. Complete Project Directory Layout

```text
RAGnarok/
├── config.toml                 # Global retrieval, reranker, and guardrail settings
├── .env.example                # Template for LLM API keys and runtime paths
├── pyproject.toml              # Dependencies, build scripts, and CLI entry points
├── uv.lock                     # Deterministic dependency lockfile
├── scripts/
│   ├── download_models.py      # Automated ONNX neural weights pre-warmer (~98 MB)
│   ├── download_benchmarks.py  # Benchmark dataset downloader (~30 MB)
│   └── run_all_benchmarks.py   # Master 5-layer benchmark runner
├── models/                     # Local ONNX weights (gitignored)
│   ├── fastembed_cache/        # BAAI/bge-small-en-v1.5 embeddings (~67 MB)
│   └── ms-marco-MiniLM-L-12-v2/ # FlashRank Cross-Encoder reranker (~31 MB)
├── data/                       # Local runtime storage (gitignored)
│   ├── ragnarok.duckdb         # Relational database for parent chunks & documents
│   ├── indexes/                # TurboVec binary embeddings & BM25s vocabulary
│   ├── quarantine/             # Isolated Trojan documents & forensic audit sidecars
│   └── benchmarks/             # Downloaded SciFact, NFCorpus, RGB, BIPIA, MultiHop
├── docs/dataset/               # Clean sample knowledge documentation
├── src/ragnarok/
│   ├── api/                    # FastAPI backend & static SPA Web UI (index.html, app.js, style.css)
│   ├── cli/                    # Typer subcommands, REPL shells, and component probes
│   ├── core/                   # Pydantic settings and data contracts
│   ├── evals/                  # Benchmark execution engines (SciFact, NFCorpus, RGB, etc.)
│   ├── guardrails/             # Input Gateway, Staging Quarantine, Stream Sanitizer
│   ├── indexing/               # BM25s sparse index & TurboVec dense vector store
│   ├── retrieval/              # Hybrid RRF fusion pipeline & Cross-Encoder reranker
│   └── storage/                # DuckDB relational schema & CRUD operations
└── notes/
    ├── arc/                    # Architectural decision blueprints (00-08)
    ├── developer/              # Operational, testing, and interface handbooks
    └── report/                 # Empirical evaluation scorecards, papers, and algorithms
```

---

## 5. Troubleshooting & FAQ

### 1. `command not found: uv`
* **Cause:** The shell has not updated its `PATH` variable to include Astral `uv` binaries.
* **Fix:** In WSL/Linux execute `source "$HOME/.local/bin/env"`. In Git Bash execute `export PATH="$HOME/.local/bin:$PATH"` or restart your terminal.

### 2. `duckdb.IOException: Could not set lock on file`
* **Cause:** Another terminal session or background server is currently holding an exclusive write lock on `data/ragnarok.duckdb`.
* **Fix:** Terminate duplicate running processes (`kill $(pgrep -f "rag")` in WSL/Linux or end Python processes in Windows Task Manager).

### 3. FastEmbed ONNX Model Download Hangs or Fails
* **Cause:** Network interruption or blocked access to HuggingFace during initial download.
* **Fix:** Run `uv run python scripts/download_models.py`. It performs robust streaming downloads directly into the local `models/` directory with automatic retries.

### 4. Running Benchmarks on Another Machine
* **Question:** How do I run the full benchmark suite on a more powerful machine?
* **Answer:** Follow Steps 1–5 on the target machine, run `python scripts/download_benchmarks.py`, and then run `uv run python scripts/run_all_benchmarks.py`. The resulting JSON scorecards are written to `notes/report/` and immediately reflected in the Web UI dashboard under **METRICS**.
