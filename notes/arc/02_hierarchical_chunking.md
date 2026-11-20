# 02 — Hierarchical Chunking Architecture

> **Lifecycle Stage:** Step 2 in Write Path (Ingestion Pipeline)  
> **Source Module:** `src/ragnarok/ingestion/chunker.py`  
> **Responsibility:** Multi-resolution document segmentation, markdown AST preservation, and parent-child provenance mapping.

---

## 1. Architectural Problem: The Granularity Dilemma

Standard RAG architectures face a fundamental tension in chunk sizing:
* **Small Chunks ($128$–$256$ tokens):** Optimize dense vector search precision and lexical BM25 matching because embeddings concentrate on specific concepts without semantic dilution. However, they lack surrounding narrative context, leading to hallucinations during generation.
* **Large Chunks ($1,000$–$2,000$ tokens):** Provide rich context for LLM generation and reasoning, but dilute embedding vectors with multiple topics, causing vector search to miss specific facts.

RAGnarok resolves this dilemma through **Hierarchical Parent-Child Chunking**:
1. Documents are segmented into **large parent chunks** (~$1,024$ tokens) representing coherent sections.
2. Each parent chunk is subdivided into **small child chunks** (~$256$ tokens with $64$-token overlap).
3. Search indexes store only the child chunks; upon retrieval, the system expands child hits back to their parent chunks in DuckDB.

---

## 2. Component Topology & Segmentation Flow

```mermaid
flowchart TD
    DOC["Clean Staged Document<br>(from 01_ingestion)"] --> SPLIT["Hierarchical Chunker Subsystem"]
    
    subgraph ParentStage["Stage 1: Parent Chunking (~1024 Tokens)"]
        SPLIT --> HEAD["Markdown Header AST Splitter<br>(Splits on #, ##, ### section boundaries)"]
        HEAD --> PAR_WIN["Token Window Guard<br>(Merges short sections, splits >1024 token sections)"]
        PAR_WIN --> P1["Parent Chunk 001<br>parent_id: doc_01_p001<br>Tokens: ~1024"]
        PAR_WIN --> P2["Parent Chunk 002<br>parent_id: doc_01_p002<br>Tokens: ~1024"]
    end
    
    subgraph ChildStage["Stage 2: Child Chunking (~256 Tokens)"]
        P1 --> C_SPLIT1["Sliding Window Sub-Chunker<br>(Window: 256, Overlap: 64)"]
        C_SPLIT1 --> C1A["Child 001 (doc_01_p001_c01)"]
        C_SPLIT1 --> C1B["Child 002 (doc_01_p001_c02)"]
        C_SPLIT1 --> C1C["Child 003 (doc_01_p001_c03)"]
        
        P2 --> C_SPLIT2["Sliding Window Sub-Chunker<br>(Window: 256, Overlap: 64)"]
        C_SPLIT2 --> C2A["Child 004 (doc_01_p002_c01)"]
        C_SPLIT2 --> C2B["Child 005 (doc_01_p002_c02)"]
    end
    
    P1 -. "Persisted directly to DuckDB" .-> DDB[("DuckDB Storage")]
    P2 -. "Persisted directly to DuckDB" .-> DDB
    
    C1A -. "Indexed in BM25 & TurboVec" .-> IDX[("Dual Search Indexes")]
    C1B -. "Indexed in BM25 & TurboVec" .-> IDX
    C1C -. "Indexed in BM25 & TurboVec" .-> IDX
    C2A -. "Indexed in BM25 & TurboVec" .-> IDX
    C2B -. "Indexed in BM25 & TurboVec" .-> IDX
```

---

## 3. Technical Invariants & Segmentation Mechanics

### 3.1 Markdown AST Section Splitting
* **Structural Preservation:** The chunker utilizes markdown syntax trees to avoid arbitrary mid-paragraph splits. It prioritizes splitting on H1 (`#`), H2 (`##`), and H3 (`###`) header boundaries.
* **Context Preservation:** When a section is split into a parent chunk, the active heading hierarchy (e.g., `# Architecture > ## Storage > ### DuckDB`) is prepended to the chunk metadata to preserve semantic context.

### 3.2 Dual Windowing Parameters
* **Parent Chunks:**
  * Target size: $1,024$ tokens ($\pm 15\%$).
  * Minimum threshold: Sections under $100$ tokens are merged into adjacent paragraphs to prevent fragmented parent context.
  * Storage: Persisted in DuckDB table `parent_chunks`. Parent chunks are **never** vectorized, keeping the vector index size minimal.
* **Child Chunks:**
  * Target size: $256$ tokens.
  * Overlap: $64$ tokens ($25\%$ overlap) using a sliding token window. This ensures compound phrases or multi-line code blocks spanning chunk edges remain searchable.
  * Foreign Key Binding: Every child chunk maintains a deterministic `parent_id` foreign key linking back to its enclosing parent section.

---

## 4. Failure Modes & Edge Cases

| Edge Case | Problem Scenario | Resolution & Architectural Invariant |
| :--- | :--- | :--- |
| **Monolithic Code / Tables** | A single continuous code block or table exceeds $1,024$ tokens without markdown headers. | The parent chunker falls back to recursive line-by-line token windowing, splitting on newline boundaries (`\n`) rather than breaking mid-line. |
| **Tiny Final Section** | A document ends with a $20$-token copyright notice or brief disclaimer. | If token count $< 64$ tokens, it is appended to the preceding parent chunk rather than creating an isolated orphan parent. |
| **Token Truncation Overflow** | Non-English or Unicode-dense documents with high character-to-token ratios. | BPE tokenizer (`tiktoken`) enforces token-based counting rather than character-length approximations. |
