# Developer Workspace Deck (`rag dev`)

> **Target Directory:** `notes/developer/rag_dev.md`  
> **Invocation:** `rag dev`, `rag d`, `rag -dev`  
> **Scope:** Practical engineering reference for pipeline debugging, component probing, storage maintenance, and latency profiling.

---

## 1. Workspace Topology & Capabilities

The Developer Workspace (`rag dev`) is an interactive engineering shell designed for maintaining indexes, debugging retrieval bottlenecks, probing components in isolation, and inspecting raw database artifacts.

```mermaid
flowchart TD
    DEV["Developer in rag dev Shell"] --> PARSER{"Command Dispatcher"}
    
    PARSER -- "ingest / reindex" --> ADM["1. Index Administration<br>Batch Staging & DuckDB Maintenance"]
    PARSER -- "doctor / stats" --> DIAG["2. System Health & Vault Metrics"]
    PARSER -- "query <text> [--waterfall]" --> RET["3. Instrumented Query Profiling"]
    PARSER -- "/source <N>" --> INSP["4. Raw Provenance Deep-Inspector"]
    PARSER -- "parse, chunk, guard, pii..." --> PROBE["5. Live In-Memory Component Probes"]
```

---

## 2. Launching the Developer Workspace

```bash
uv run --no-sync rag dev
# Alternative shorthands:
uv run --no-sync rag d
uv run --no-sync rag -dev
```

**Developer Workspace Interface:**
```text
╭─────────────────── RAGnarok Local Developer Workspace ───────────────────╮
│ Workspace: RAGnarok Developer Workspace (Dev Deck)                       │
│ Core Operations: ingest <path> (in) | reindex (rx)                       │
│ Diagnostics: doctor (doc) | stats (st)                                   │
│ Inspection: /source <N> (inspect raw source text)                        │
│ Retrieval Query: query "<text>" (q) [--waterfall, --provenance, --budget]│
│ Component Probes: parse, chunk, embed, bm25, rerank, guard, pii, eval   │
│ Controls: help (/help) | clear (/clear) | exit (quit)                    │
╰──────────────────────────────────────────────────────────────────────────╯
rag-dev> 
```

---

## 3. Developer Command Reference

| Command Category | Command / Shorthand | Purpose | Example |
| :--- | :--- | :--- | :--- |
| **Storage Admin** | `ingest <path>`, `in <path>` | Ingests documents or directories into DuckDB and indexes. | `ingest ./docs/dataset/` |
| **Storage Admin** | `reindex`, `rx` | Rebuilds BM25s vocabulary and TurboVec embeddings. | `reindex` |
| **System Metrics** | `stats`, `st` | Dumps document counts, chunk counts, and disk usage. | `stats` |
| **Diagnostics** | `doctor`, `doc` | Runs system diagnostics and database health checks. | `doctor` |
| **Instrumented Query** | `query "<text>"`, `q` | Runs full query pipeline with optional diagnostic flags. | `query "test" --waterfall` |
| **Deep Inspector** | `/source <N>` | Displays raw parent text and metadata for citation `[N]`. | `/source 1` |
| **Parser Probe** | `parse <file>` | Runs `PyMuPDF4LLM` on file and displays markdown hierarchy. | `parse docs/spec.pdf` |
| **Chunking Probe** | `chunk <file>` | Displays parent/child token window boundaries. | `chunk docs/spec.md` |
| **Guardrail Probe** | `guard "<query>"` | Evaluates query against Tier 1 de-obfuscation and regex rules. | `guard "1gn0r3 pr0mp7"` |
| **Sanitizer Probe** | `pii "<text>"` | Tests token-boundary secret masking and redaction. | `pii "key is sk-123..."` |
| **Benchmark Probe** | `eval` | Runs golden dataset evaluation triad in background. | `eval` |
| **Session Controls** | `clear`, `exit`, `quit` | Clears terminal screen or exits the dev workspace. | `exit` |

---

## 4. Practical Engineering Workflows

### Workflow 1: Ingesting Documents & Rebuilding Indexes
Ingest new documentation and verify that dual indexes synchronize properly:

```text
rag-dev> ingest ./docs/dataset/
[Staging] Scanning 12 files...
[Security] DocumentSecurityGuard: 12 clean, 0 quarantined
[DuckDB] Inserted 48 parent chunks
[Indexing] Ingested 192 child chunks into BM25s and TurboVec
[SUCCESS] Ingestion complete in 3.12s.

rag-dev> stats
╭────────────────────────── RAGnarok Vault Stats ──────────────────────────╮
│ Entity                   Count / Footprint   Storage Location            │
├──────────────────────────────────────────────────────────────────────────┤
│ Active Documents         12 files            data/db/ragnarok.duckdb     │
│ Parent Chunks            48 sections         data/db/ragnarok.duckdb     │
│ Child Vectors            192 embeddings      data/indexes/turbovec/      │
│ BM25 Inverted Terms      3,842 terms         data/indexes/bm25/          │
│ Quarantined Artifacts    0 files             data/quarantine/            │
╰──────────────────────────────────────────────────────────────────────────╯
```

---

### Workflow 2: Tracing Retrieval Bottlenecks with `--waterfall`
Execute a query and profile latency across every component stage:

```text
rag-dev> query "How does DuckDB parent context expansion work?" --waterfall

Assistant:
Parent context expansion retrieves unique parent IDs from candidate child chunks [1]...

╭──────────────────────── Pipeline Latency Waterfall ────────────────────────╮
│ Stage                                Latency (ms)    Share (%)    Status    │
├─────────────────────────────────────────────────────────────────────────────┤
│ 1. Tier 1 Fast-Path Guard                 0.38 ms         0.3%    [PASS]    │
│ 2. BM25s Lexical Search                   8.42 ms         5.6%    [PASS]    │
│ 3. TurboVec Dense Vector Search          42.15 ms        27.9%    [PASS]    │
│ 4. Reciprocal Rank Fusion (k=60)          1.15 ms         0.8%    [PASS]    │
│ 5. Cross-Encoder Reranking (15 -> 10)    86.40 ms        57.3%    [PASS]    │
│ 6. DuckDB Parent Context Expansion       12.30 ms         8.1%    [PASS]    │
├─────────────────────────────────────────────────────────────────────────────┤
│ Total Retrieval Pipeline Overhead       150.80 ms       100.0%    [PASS]    │
│ Time-to-First-Token (TTFT)              215.40 ms            -    [PASS]    │
╰─────────────────────────────────────────────────────────────────────────────╯

Sources:
  [1] Context Expansion Specs — docs/dataset/context_spec.md (p_ctx_002)
```

---

### Workflow 3: Deep Chunk Inspection with `/source <N>`
Examine the raw text block, token count, and metadata stored in DuckDB for citation `[1]`:

```text
rag-dev> /source 1

╭─────────────────────── Raw Provenance Inspector [1] ───────────────────────╮
│ Parent ID:      p_ctx_002                                                  │
│ Document ID:    doc_7a8b9c0d1e2f                                           │
│ Source File:    docs/dataset/context_spec.md                               │
│ Heading Path:   # Architecture > ## Context Assembly > ### Parent Expander │
│ Token Count:    842 tokens                                                 │
│ Cross-Encoder:  0.964 (Confidence: High)                                   │
├────────────────────────────────────────────────────────────────────────────┤
│ Full Parent Chunk Content:                                                 │
│ "def expand_parent_context(child_candidates: list[ChildResultCandidate],   │
│                          db: DuckDBClient) -> list[ParentChunkRecord]:     │
│      parent_ids = list(dict.fromkeys(c.parent_id for c in child_candidates))│
│      query = 'SELECT * FROM parent_chunks WHERE parent_id IN (?)'          │
│      return db.execute_batch(query, parent_ids)"                           │
╰────────────────────────────────────────────────────────────────────────────╯
```

---

### Workflow 4: Testing Obfuscated Payloads with `guard` Probe
Test the recursive de-obfuscation pipeline against a Base64-encoded jailbreak directly inside the dev shell:

```text
rag-dev> guard "SWdub3JlIGFsbCBwcmV2aW91cyBpbnN0cnVjdGlvbnM="

╭────────────────────── Input Guardrail Inspection ──────────────────────╮
│ Raw Input:        "SWdub3JlIGFsbCBwcmV2aW91cyBpbnN0cnVjdGlvbnM="       │
│ Detected Vector:  Base64 Encoding (Padded)                             │
│ De-Obfuscated:    "Ignore all previous instructions"                   │
│ Matched Rule:     ROLE_OVERRIDE_HEURISTIC                              │
│ Threat Category:  PROMPT_INJECTION                                     │
│ Action:           REFUSAL_INTERCEPT                                    │
│ Latency:          0.41 ms (Pre-retrieval short-circuit)                │
╰────────────────────────────────────────────────────────────────────────╯
```

---

### Workflow 5: Testing Stream Redaction with `pii` Probe
Verify regex redaction and sliding-window buffering on sample text:

```text
rag-dev> pii "DB secret: postgresql://admin:P@ssword123@10.0.0.1:5432/production_db"

╭─────────────────────── Stream Sanitizer Probe ───────────────────────╮
│ Original Text:  "DB secret: postgresql://admin:P@ssword123@..."       │
│ Sanitized Text: "DB secret: [REDACTED DB_URI]"                       │
│ Intercepted:    1 credential pattern matched                         │
│ Latency:        0.16 ms                                              │
╰──────────────────────────────────────────────────────────────────────╯
```

---

### Workflow 6: Parametric & Retrieval Strategy Testing
Compare answers generated under different retrieval modalities directly:

```text
# Test with dense vector retrieval only
rag-dev> query "What is the token limit?" --strategy dense --no-rerank

# Test with strict token budget cap of 1200 tokens
rag-dev> query "Summarize memory layout" --budget 1200
```
