# 00 — RAGnarok Master Architecture Blueprint

> **Scope:** End-to-end data lifecycle, pipeline topology, latency budgeting, and component interaction flows.  
> **Order:** Chronological master index and navigation blueprint.

---

## 1. System Topology & Decoupled Architecture

RAGnarok decouples operational workflows into two independent pipelines with zero runtime resource contention:
1. **Knowledge Ingestion & Indexing Pipeline (Write Path):** An asynchronous pipeline that extracts layout-aware text from documents (PDF, Markdown, TXT), screens against Indirect Prompt Injection (IPI) to isolate Trojan payloads to `data/quarantine/`, partitions text into hierarchical parent-child chunks, and updates the local storage triad (DuckDB, BM25s, and TurboVec).
2. **Guarded Query & Synthesis Pipeline (Read Path):** A low-latency synchronous/streaming pipeline that validates user queries in sub-millisecond fast-paths, classifies whether retrieval is needed via Self-RAG, executes parallel hybrid lexical-dense retrieval, expands parent context windows from DuckDB, sanitizes streaming tokens in real time, and verifies citation integrity.

```mermaid
flowchart TD
    subgraph ClientLayer["0. Client Transport & Gateway Layer"]
        CLI["Terminal Interfaces<br>(rag chat / rag dev / rag query)"]
        WEB["Single-Page Web UI<br>(SSE Stream & Operations Dashboard)"]
        CLI & WEB --> API["Client Gateway (FastAPI /routes.py)<br>• SSE Stream: POST /api/chat/stream<br>• Sync API: POST /api/query"]
    end

    subgraph WritePath["1. Ingestion & Indexing Pipeline (Write Path)"]
        direction TB
        RAW["Raw Documents (.pdf, .md, .txt)"] --> S01["01: Ingestion & Quarantine<br>• PyMuPDF Layout Parsing<br>• Surface 2 Document Guard (BIPIA)"]
        S01 -- "Trojan Detected" --> QUA[("data/quarantine/<br>• Isolated File<br>• Forensic JSON Sidecar")]
        S01 -- "Clean Document" --> S02["02: Hierarchical Chunking<br>• Parent Sections (512-1024 tok)<br>• Child Chunks (256 tok, 64 overlap)"]
        S02 --> S03["03: Storage & Dual Indexing"]
        
        subgraph StorageEngines["Storage & Search Triad"]
            S03 --> DDB[("DuckDB Relational Vault<br>• documents table<br>• parent_chunks table")]
            S03 --> BM25[("BM25s Lexical Index<br>• Sparse Inverted Vocabulary<br>• 0.45 ms Exact Term Search")]
            S03 --> TVEC[("TurboVec Dense Index<br>• 384-d FastEmbed ONNX<br>• 259 ms Incremental Sync")]
            S03 --> MDL[("Local ONNX Models (models/)<br>• FastEmbed cache (~67 MB)<br>• FlashRank MiniLM (~31 MB)")]
        end
    end

    subgraph ReadPath["2. Guarded Query & Synthesis Pipeline (Read Path)"]
        direction TB
        API --> S04["04: Input Threat Gateway (Surface 1)<br>• Fast-Path De-Obfuscation & Regex (&lt; 1 ms)<br>• Shannon Entropy Analyzer<br>• Self-RAG Parametric Classifier"]
        S04 -- "Threat Intercepted" --> REF["Immediate Refusal Response<br>(&lt; 0.5 ms Latency)"]
        S04 -- "Parametric Query" --> LLM_DIR["Direct LLM Generator<br>(Bypasses Retrieval Entirely)"]
        S04 -- "Corpus Query" --> S05["05: Hybrid Retrieval & Reranking<br>• Parallel BM25s + TurboVec Search<br>• Reciprocal Rank Fusion (RRF k=60)<br>• Optional Cross-Encoder Reranker"]
        
        S05 --> S06["06: Context Assembly & Expansion<br>• DuckDB Sibling Window Expansion<br>• Token Budgeting (Max Budget)<br>• Perimeter Ordering ('Lost in Middle')"]
        S06 --> S07["07: Generation & Stream Sanitizer<br>• Zero-Speculation Prompt Directive<br>• Speculative Sliding Buffer (W = 35)<br>• Real-Time Secret Redaction"]
        S07 --> S08["08: Grounding & Citation Integrity<br>• 1:1 DuckDB Chunk Binding [1], [2]<br>• Ghost Citation Elimination (0.00% GCR)<br>• 98.0% Negative Rejection Directive"]
        S08 --> OUT["Verified Grounded Answer<br>Interactive Citation Badges & Evidence Drawer"]
    end

    DDB -. "Parent Window Expansion" .-> S06
    BM25 -. "0.45 ms Lexical Matches" .-> S05
    TVEC -. "37.3 ms Dense Vectors" .-> S05
```

---

## 2. Chronological Architectural Directory

The documents in this directory represent the chronological progression of components through the system:

| Order | Component Document | Primary Subsystem | Key Architectural Responsibility |
| :--: | :--- | :--- | :--- |
| **01** | [`01_ingestion_and_quarantine.md`](./01_ingestion_and_quarantine.md) | Ingestion Boundary | Document parsing, SHA-256 deduplication, Surface 2 BIPIA Trojan screening, and quarantine isolation. |
| **02** | [`02_hierarchical_chunking.md`](./02_hierarchical_chunking.md) | Data Transformation | Dual-resolution parent-child segmentation, markdown AST preservation, and foreign key lineage binding. |
| **03** | [`03_storage_and_indexing.md`](./03_storage_and_indexing.md) | Storage Triad | Relational DuckDB vault, BM25s sparse index, TurboVec dense index, and 259ms incremental sync. |
| **04** | [`04_input_guardrail_gateway.md`](./04_input_guardrail_gateway.md) | Pre-Retrieval Gateway | Sub-millisecond de-obfuscation, Shannon entropy analysis, and Self-RAG parametric routing. |
| **05** | [`05_hybrid_retrieval_and_reranking.md`](./05_hybrid_retrieval_and_reranking.md) | Search & Ranking | Parallel lexical/dense fanout, RRF $k=60$ fusion, and Pareto frontier latency analysis. |
| **06** | [`06_context_assembly_and_expansion.md`](./06_context_assembly_and_expansion.md) | Context Synthesis | DuckDB parent window expansion (+4.42% MultiHop recall), token budgeting, and perimeter ordering. |
| **07** | [`07_generation_and_stream_sanitization.md`](./07_generation_and_stream_sanitization.md) | Inference & Redaction | LLM generative streaming and sliding-window ($W=35$) token-boundary secret redaction ($\text{SLR}=0.00\%$). |
| **08** | [`08_grounding_and_citation_integrity.md`](./08_grounding_and_citation_integrity.md) | Verification & Grounding | 1:1 citation binding, 0.00% ghost citation rate, and 98.0% negative refusal directive. |
