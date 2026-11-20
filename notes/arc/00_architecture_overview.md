# 00 — RAGnarok Master Architecture Blueprint

> **Scope:** End-to-end data lifecycle, pipeline topology, latency budgeting, and component interaction flows.  
> **Order:** Chronological master index and navigation blueprint.

---

## 1. System Topology & Dual Pipeline Separation

RAGnarok decouples operational workflows into two independent pipelines with zero runtime resource contention:
1. **Knowledge Ingestion & Indexing Pipeline (Write Path):** An asynchronous, batch-capable pipeline that extracts layout-aware text from documents, performs pre-ingestion security screening to isolate Trojan Horse payloads, partitions text into hierarchical parent-child chunks, and updates relational and dual-index stores.
2. **Guarded Query & Synthesis Pipeline (Read Path):** A low-latency synchronous pipeline that validates user queries in sub-millisecond fast-paths, executes hybrid lexical-dense retrieval, reranks candidate passages, expands parent context from DuckDB, sanitizes streaming tokens against boundary secret leaks, and verifies factual grounding.

```mermaid
flowchart TD
    subgraph WritePath["1. Ingestion & Indexing Pipeline (Write Path)"]
        direction TB
        RAW["Raw Un-indexed Documents<br>(PDF, Markdown, Text)"] --> S01["01: Ingestion & Quarantine<br>• PyMuPDF4LLM Parsing<br>• DocumentSecurityGuard Screening"]
        S01 -- "Malicious / Trojan" --> QUA[("data/quarantine/<br>• Isolated Artifact<br>• Forensic JSON Sidecar")]
        S01 -- "Clean Document" --> S02["02: Hierarchical Chunking<br>• Parent Chunker (~1024 tok)<br>• Child Chunker (~256 tok, 64 overlap)"]
        S02 --> S03["03: Storage & Dual Indexing"]
        
        subgraph StorageEngines["Storage & Search Layer"]
            S03 --> DDB[("DuckDB Relational Vault<br>• documents table<br>• parent_chunks table")]
            S03 --> BM25[("BM25s Lexical Index<br>• Inverted Term Directory<br>• Child Chunk Tokens")]
            S03 --> TVEC[("TurboVec Dense Index<br>• FastEmbed 384-d Vectors<br>• Cosine Search Arrays")]
        end
    end

    subgraph ReadPath["2. Guarded Query & Synthesis Pipeline (Read Path)"]
        direction TB
        QRY["Raw User Input Query"] --> S04["04: Input Guardrail Gateway<br>• Tier 1: Fast-Path De-Obfuscation & Regex (&lt; 1 ms)<br>• Tier 2: Semantic Intent Classifier"]
        S04 -- "Threat Intercepted" --> REF["Immediate Refusal Response<br>(&lt; 0.5 ms Latency)"]
        S04 -- "Sanitized Query" --> S05["05: Hybrid Retrieval & Reranking<br>• Parallel BM25s + TurboVec Search<br>• Reciprocal Rank Fusion (k=60)<br>• Cross-Encoder Reranker (15 to 10)"]
        
        S05 --> S06["06: Context Assembly & Expansion<br>• Child-to-Parent Relational Lookup<br>• Token Budgeting (Max Budget)<br>• Perimeter Ordering ('Lost in Middle')"]
        S06 --> S07["07: Generation & Stream Sanitizer<br>• LLM Generative Synthesis<br>• Speculative Sliding Buffer (W = 35)<br>• Live Token Boundary Redaction"]
        S07 --> S08["08: Grounding & Citation Integrity<br>• Claim-Level Entailment Verification<br>• 1:1 Provenance Footnote Binding<br>• Ghost Citation Elimination"]
        S08 --> OUT["Verified Grounded Answer<br>with Footnotes Table"]
    end

    DDB -. "Parent Chunks" .-> S06
    BM25 -. "Lexical Matches" .-> S05
    TVEC -. "Dense Matches" .-> S05
```

---

## 2. Chronological Architectural Directory

The documents in this directory represent the chronological progression of components through the system. Each document provides structural design rationales, invariants, topologies, and failure handling.

| Order | Component Document | Primary Subsystem | Key Architectural Responsibility |
| :--: | :--- | :--- | :--- |
| **01** | [`01_ingestion_and_quarantine.md`](./01_ingestion_and_quarantine.md) | Ingestion Boundary | Document parsing, Indirect Prompt Injection (IPI) screening, and zero-contamination quarantine. |
| **02** | [`02_hierarchical_chunking.md`](./02_hierarchical_chunking.md) | Data Transformation | Dual-resolution parent-child segmentation, markdown AST preservation, and provenance keys. |
| **03** | [`03_storage_and_indexing.md`](./03_storage_and_indexing.md) | Storage & Dual Indexes | Relational DuckDB parent store, BM25s inverted term index, and TurboVec dense vector store. |
| **04** | [`04_input_guardrail_gateway.md`](./04_input_guardrail_gateway.md) | Pre-Retrieval Gateway | Sub-millisecond recursive de-obfuscation, token bounding, and heuristic regex injection defense. |
| **05** | [`05_hybrid_retrieval_and_reranking.md`](./05_hybrid_retrieval_and_reranking.md) | Search & Ranking | Parallel lexical/dense retrieval, Reciprocal Rank Fusion, and Cross-Encoder distractor pruning. |
| **06** | [`06_context_assembly_and_expansion.md`](./06_context_assembly_and_expansion.md) | Context Synthesis | Relational parent resolution, token budgeting, and perimeter ordering ("Lost in the Middle"). |
| **07** | [`07_generation_and_stream_sanitization.md`](./07_generation_and_stream_sanitization.md) | Inference & Redaction | LLM generative streaming and sliding-window ($W=35$) token-boundary secret redaction. |
| **08** | [`08_grounding_and_citation_integrity.md`](./08_grounding_and_citation_integrity.md) | Verification & Grounding | Claim entailment verification, 1:1 citation binding, ghost footnote elimination, and refusal fallback. |
