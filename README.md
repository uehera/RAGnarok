# 🔱 *RAGnarok*

<div align="center">

[![Python](https://img.shields.io/badge/Python-3.12-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![uv](https://img.shields.io/badge/uv-Fast%20Packaging-DE5FE9?style=for-the-badge&logo=astral&logoColor=white)](https://github.com/astral-sh/uv)
[![Docker](https://img.shields.io/badge/Docker-PostgreSQL%2016-2496ED?style=for-the-badge&logo=docker&logoColor=white)](https://www.docker.com)
[![TurboVec](https://img.shields.io/badge/TurboVec-Dense%20Search-FF6B6B?style=for-the-badge)](https://github.com)
[![BM25s](https://img.shields.io/badge/BM25s-Lexical%20Search-4D96FF?style=for-the-badge)](https://github.com)
[![CI/CD](https://img.shields.io/badge/CI%2FCD-Quality%20Gates%20Passed-success?style=for-the-badge&logo=githubactions&logoColor=white)](https://github.com)
[![Docling](https://img.shields.io/badge/Docling-Layout%20Parser-FFB26B?style=for-the-badge)](https://github.com/DS4SD/docling)
[![License](https://img.shields.io/badge/License-MIT-blue?style=for-the-badge)](LICENSE)

<br/>

**A high-performance, modular, and local Retrieval-Augmented Generation (RAG) system with parent-child chunking, hybrid retrieval fusion, native guardrails, and continuous CI/CD evaluation loops.**

<br/>

[📖 RAGnarok Documentation](https://github.com/ver1619/RAGnarok_DOCS) • [💻 RAGnarok CLI Guide](https://github.com/ver1619/RAGnarok_DOCS/blob/main/rag_cli.md) 

</div>

---

## 📖 What is RAGnarok?

Most Retrieval-Augmented Generation (RAG) systems suffer from two major problems:
1. **Context Fragmentation**: Splitting text into tiny chunks helps the search engine find matches, but the language model loses the surrounding context and hallucinates or gives incomplete answers.
2. **Search Imbalance**: Vector similarity search is great for general concepts but struggles with exact codes, acronyms, and identifiers; meanwhile, keyword search struggles with conceptual meaning.

**RAGnarok** solves this with an elegant, decoupled architecture:
- **Parent-Child Chunking**: Small **Child Chunks** (128–256 tokens) are indexed for precision search, but when a match is found, RAGnarok retrieves the complete, broad **Parent Chunk** (up to 2,000 tokens) from PostgreSQL. The LLM receives the full context without sacrificing search precision.
- **Hybrid Retrieval Fusion**: Merges **BM25s** (fast lexical keyword search) and **TurboVec** (dense semantic vector similarity) using **Reciprocal Rank Fusion (RRF)**, optionally refined with a **Cross-Encoder Reranker**.
- **Native Two-Tier Guardrails**: Built-in, lightweight safety filters that block prompt injections and off-topic queries on input, and automatically redact sensitive API keys, database credentials, and PII on output.
- **Dual-Environment CLI**: A streamlined terminal deck separating conversational question-answering (`rag -c`) from ingestion, reindexing, and diagnostics (`rag -dev`).
- **Continuous Quality Gating**: Fully integrated with **DeepEval** and automated GitHub Actions CI to evaluate Contextual Precision, Contextual Recall, Faithfulness, and Answer Relevancy across a 20-case Golden Dataset.

---

## 🏛️ System Architecture

```mermaid
flowchart TD
    subgraph INGESTION ["1. Ingestion & Storage"]
        Doc["User Documents\n(PDF, DOCX, TXT, MD)"] --> Stage["Document Staging\n(rag add / data/staging/)"]
        Stage --> Deduplicate{"SHA-256 Checksum\nDeduplication"}
        Deduplicate -- "New File" --> Parse["Docling Parser\n(Layout & Heading Extraction)"]
        Parse --> Chunk["Two-Stage Chunker\n(Parent Sections & Child Tokens)"]
        Chunk --> PG[("PostgreSQL\n(Metadata, Chunks, Parents, Jobs)")]
        Chunk --> DenseIdx[("TurboVec Vector Index\n(384-dim Embeddings)")]
        Chunk --> LexIdx[("BM25s Lexical Index\n(Inverted Index)")]
    end

    subgraph QUERY ["2. Safety & Query Routing"]
        UserQuery["User Inquiry"] --> InGuard{"Input Guardrail\n(Injection & Scope Filter)"}
        InGuard -- "Safe" --> SelfRAG{"Self-RAG Gate\nNeeds Retrieval?"}
        InGuard -- "Unsafe" --> RefusalNotice["Safety Refusal Notice"]
        SelfRAG -- "YES" --> Router["Adaptive Router\n(Lexical, Dense, or Hybrid)"]
    end

    subgraph RETRIEVAL ["3. Hybrid Search & Expansion"]
        Router --> LexSearch["BM25s Lexical Search"]
        Router --> DenseSearch["TurboVec Dense Search"]
        LexSearch & DenseSearch --> RRF["Reciprocal Rank Fusion (RRF k=60)"]
        RRF --> Reranker["Cross-Encoder Reranker\n(ms-marco-MiniLM-L-6-v2)"]
        Reranker --> TopMatches["Top Child Matches"]
        TopMatches --> ParentFetch["Parent Expansion\n(Fetch Full Context from PostgreSQL)"]
        ParentFetch --> Budget["Token Budget Manager\n(Whole-Parent Fitting up to 4000 tokens)"]
    end

    subgraph GENERATION ["4. Synthesis & Output Safety"]
        Budget --> LLM["litellm Synthesis Engine\n(Multi-Model Fallback Chain)"]
        SelfRAG -- "NO (Direct)" --> LLM
        LLM --> Stream["Real-Time Token Streaming\n(Sub-500ms TTFT)"]
        Stream --> Citations["Inline Footnote Citations\n([1], [2] to Documents)"]
        Citations --> OutGuard{"Output Guardrail\n(PII & Grounding Check)"}
        OutGuard --> FinalOutput["Rendered Answer + Sources"]
    end
```

---

## ⚡ Prerequisites

Before running RAGnarok, ensure you have the following installed:

- **Operating System**: Linux, macOS, Windows, or WSL2
- **Python**: Version `3.12` (managed via `uv`)
- **Package Manager**: [`uv`](https://github.com/astral-sh/uv) ($\ge 0.10$)
- **Database**: [Docker & Docker Compose](https://www.docker.com/) (runs the PostgreSQL 16 container)

---

## 🚀 Quickstart Guide

### 1. Clone the Repository

```bash
git clone https://github.com/ver1619/RAGnarok.git
cd RAGnarok
```

### 2. Install Dependencies via `uv`

RAGnarok uses `uv` for reproducible, lightning-fast dependency management:

```bash
# Install Python 3.12 and synchronize all dependencies
uv sync --all-extras --dev

# Activate the virtual environment
source .venv/bin/activate    # On WSL/Linux/macOS

.venv\Scripts\activate     # On Windows PowerShell
```

### 3. Start PostgreSQL Metadata Store

Launch the containerized PostgreSQL database using Docker Compose:

```bash
docker compose up -d
```

Verify that the database is healthy:
```bash
docker compose ps
```

### 4. Configure Environment Variables

Copy the example environment file and add your model provider keys:

```bash
cp .env.example .env
```

Edit `.env` with your preferred settings:
```bash
# PostgreSQL Connection (matches docker-compose.yml defaults)
DATABASE_URL="postgresql://postgres:postgres@localhost:5432/ragnarok"

# Optional Cloud LLM API Keys (Groq, OpenAI, Gemini, Anthropic)
GROQ_API_KEY="your-groq-api-key"
OPENAI_API_KEY="your-openai-api-key"

# DeepEval CI/CD & Telemetry Settings
DEEPEVAL_TELEMETRY_OPT_OUT="Y"
DEEPEVAL_CACHE_FOLDER=".pytest_cache/deepeval"
```

### 5. Run System Diagnostics (Setup Doctor)

Verify that your Python runtime, database connection, storage directories, and critical libraries are fully operational:

```bash
uv run rag doctor
# or
uv run python scripts/verify_setup.py
```

Expected output:
```text
────────────────────── RAGnarok Environment Diagnostic Check ──────────────────────
Checking Python version...                 [OK] Python 3.12.10
Checking critical dependencies...          [OK] 12 packages imported successfully
Checking configuration subsystem...        [OK] Configuration loaded cleanly
Checking runtime filesystem directories... [OK] 5 directories verified and writable
Checking PostgreSQL connectivity...        [OK] Connected. Found all 4 tables
Detecting hardware acceleration...         [NOTE] Running on CPU (standard fallback)
───────────────────────────────────────────────────────────────────────────────────
Setup status: READY TO DEVELOP
```

> [!NOTE]
> **Admin-Faced CLI Architecture**  
> The current Command-Line Interface is designed as an operational Admin Deck for developers and system operators (`rag -c` interactive chat deck and `rag -dev` developer workspace). An upcoming version will introduce a dedicated **Terminal User Interface (TUI)** to provide a richer, visual multi-pane UI/UX.
> 
> 📖 **Full Operations Reference**: For the complete guide covering all slash commands (`/waterfall`, `/provenance`, `/strategy`), diagnostic probes, and CLI flags, visit the **[RAGnarok CLI Guide](https://github.com/ver1619/RAGnarok_DOCS/blob/main/rag_cli.md)**.

---

## 🛠️ CLI Quickstart

RAGnarok centers around three primary workflows:

### 1. Stage & Ingest Documents
Add external documents (`.pdf`, `.docx`, `.md`, `.txt`) to the staging folder and run the two-stage chunking and dual-indexing pipeline:

```bash
uv run rag add /path/to/document.pdf docs/
uv run rag ingest docs/
```

### 2. Conversational Chat Shell (`rag -c`)
Launch an interactive session with streaming answers, citation provenance, and live runtime extensions:

```bash
rag -c       # or: uv run rag chat
```

### 3. Developer & Diagnostics Workspace (`rag -dev`)
Open the interactive administration console for index inspection, component probing, and system checks:

```bash
rag -dev     # or: uv run rag dev
```

---

## ⚙️ Configuration (`config.toml`)

All runtime parameters are centralized in [`config.toml`](config.toml) with zero hardcoded model strings:

```toml
[general]
app_name = "RAGnarok"
log_level = "INFO"

[chunking]
strategy = "markdown_header_recursive"
parent_target_size = 2000     # Parent context window
child_target_size = 500       # Child retrieval unit
child_overlap = 50

[embedding]
provider = "sentence-transformers"
model = "BAAI/bge-small-en-v1.5"
vector_dimension = 384
similarity_metric = "cosine"

[hybrid_retrieval]
k = 60                        # RRF smoothing constant

[hybrid_retrieval.reranker]
enabled = true
model = "cross-encoder/ms-marco-MiniLM-L-6-v2"
top_n = 50
top_k = 10

[generation]
model = "groq/openai/gpt-oss-20b"
temperature = 0.1
max_tokens = 1500
stream = true

[generation.fallbacks]
enabled = true
models = ["groq/qwen/qwen3.8-27b", "groq/openai/gpt-oss-120b"]

[guardrails]
enabled = true
max_query_tokens = 1024
enable_pii_filter = true
enable_injection_detection = true

[evals.thresholds]
context_precision = 0.85
context_recall = 0.85
faithfulness = 0.90
answer_relevancy = 0.85
```

---

## 🧪 Testing & Continuous Evaluation

RAGnarok enforces a rigorous test harness across unit tests, enterprise battle-testing, and automated quality gates:

```bash
# 1. Verify code formatting and linting
uv run ruff check .
uv run ruff format --check .

# 2. Verify static type safety
uv run pyright

# 3. Run full test suite (131 tests: unit, integration, chaos, SLAs)
uv run pytest -v

# 4. Run DeepEval 20-case Golden Dataset benchmark
uv run rag test eval
```

### The Evaluation Triad Results (20 Benchmark Cases)
- **Contextual Precision**: **0.98** ($\ge 0.85$ threshold)
- **Contextual Recall**: **1.00** ($\ge 0.85$ threshold)
- **Faithfulness**: **0.96** ($\ge 0.90$ threshold)
- **Answer Relevancy**: **0.98** ($\ge 0.85$ threshold)

---

## 📂 Repository Layout

```text
RAGnarok/
├── .github/workflows/       # GitHub Actions CI/CD quality gate pipelines
├── data/                    # Local runtime storage (ignored by git)
│   ├── raw/                 # Original uploaded source documents
│   ├── parsed/              # Docling-extracted Markdown files
│   ├── staging/             # Temporary ingestion incoming files
│   └── indexes/
│       ├── bm25/            # Persisted BM25s inverted lexical index
│       └── dense/           # TurboVec binary vector index and ID mapping
├── docs/                    # Staged user documents for ingestion (.pdf, .docx, .md)
├── scripts/
│   ├── init_schema.sql      # PostgreSQL relational table definitions
│   └── verify_setup.py      # Automated setup doctor script
├── src/ragnarok/
│   ├── cli/                 # Typer Admin Deck, Chat Shell & Dev Workspace
│   ├── context/             # Parent expansion, deduplication & budgeting
│   ├── core/                # Config, database connection pooling & diagnostics
│   ├── generation/          # litellm synthesis, async streaming & citations
│   ├── indexing/            # BM25s and TurboVec dual index managers
│   ├── ingestion/           # Docling parser & two-stage parent-child chunker
│   └── retrieval/           # Self-RAG, Query Rewriter, Router, RRF & Reranker
├── tests/
│   ├── battle/              # Chaos fault injection, concurrency & leak tests
│   ├── data/                # Golden Dataset benchmark test cases (JSON)
│   ├── evals/               # DeepEval continuous evaluation test runner
│   └── test_*.py            # Modular unit and subsystem test suites
├── config.toml              # Central runtime hyperparameters and models
├── docker-compose.yml       # PostgreSQL 16 container definition
└── pyproject.toml           # Project manifest, dependencies and CLI scripts
```

---

## 📚 Complete Architectural Documentation

For deep technical specifications, design rationale, and schema blueprints, visit the official documentation repository:

- 🏛️ **[RAGnarok Complete Documentation](https://github.com/ver1619/RAGnarok_DOCS)**
  - [`architecture.md`](https://github.com/ver1619/RAGnarok_DOCS/blob/main/architecture.md) — Master system blueprint and end-to-end dataflow
  - [`rag_cli.md`](https://github.com/ver1619/RAGnarok_DOCS/blob/main/rag_cli.md) — Developer CLI operations manual & command reference
  - [`parent_child.md`](https://github.com/ver1619/RAGnarok_DOCS/blob/main/parent_child.md) — Decoupled parent-child indexing theory and implementation
  - [`hybrid_retrieval_model.md`](https://github.com/ver1619/RAGnarok_DOCS/blob/main/hybrid_retrieval_model.md) — RRF fusion and Cross-Encoder reranking
  - [`guardrails.md`](https://github.com/ver1619/RAGnarok_DOCS/blob/main/guardrails.md) — Native two-tier safety and PII redaction framework
  - [`evals.md`](https://github.com/ver1619/RAGnarok_DOCS/blob/main/evals.md) — Evaluation Triad and continuous benchmarking specification

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).