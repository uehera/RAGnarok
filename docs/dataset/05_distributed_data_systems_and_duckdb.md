# Analytical Data Architecture, Vectorized Execution, and Embedded DuckDB

## Overview of Embedded OLAP Storage

Traditional transactional relational databases (OLTP systems like PostgreSQL or MySQL) store data in a row-oriented format (`RowStore`), optimizing for high-frequency point insertions, updates, and key-based record lookups. However, in modern analytical workloads and RAG metadata architectures, query patterns diverge significantly:
- Queries scan millions of metadata records across only a few specific columns (`SELECT checksum FROM documents`, `SELECT parent_index, heading_path FROM parent_chunks`).
- Aggregations, string lookups, and batch chunk joins predominate.
- High memory footprint, connection pooling overhead, and external database daemon processes introduce unnecessary operational friction for local and edge applications.

**DuckDB** is an embedded analytical database engine designed specifically for fast Online Analytical Processing (OLAP) workloads. Often described as the *"SQLite for Analytics"*, DuckDB runs directly in-process within the host Python runtime, eliminating socket network latency, client-server serialization, and container infrastructure dependencies.

---

## Architectural Pillars of DuckDB

### 1. Columnar Memory Layout
DuckDB organizes data by column rather than row. When executing analytical queries, only the columns referenced in the query projection are transferred from storage into CPU cache:

```
Row-Oriented Layout (OLTP):
[DocID_1, Filename_1, Checksum_1, Content_1], [DocID_2, Filename_2, Checksum_2, Content_2] ...

Columnar Layout (DuckDB OLAP):
DocID:    [DocID_1, DocID_2, DocID_3, ...]
Filename: [Filename_1, Filename_2, Filename_3, ...]
Checksum: [Checksum_1, Checksum_2, Checksum_3, ...]
Content:  [Content_1, Content_2, Content_3, ...]
```

This layout maximizes CPU cache utilization, eliminates wasted memory bandwidth, and allows compression algorithms (dictionary encoding, bitpacking, RLE) to achieve 5x-10x storage compression.

### 2. Vectorized Query Execution (Morsel-Driven Parallelism)
Traditional database engines process queries tuple-at-a-time (the Volcano Iterator model), incurring function call overhead on every single row.

DuckDB executes operations over **Vector Chunks**—continuous horizontal slices of contiguous columnar memory containing typically 1,024 to 2,048 values. Operations (filtering, arithmetic, string hashing) are mapped directly onto CPU SIMD vector instructions (AVX-2, AVX-512, ARM Neon), executing millions of record comparisons per CPU millisecond.

### 3. ACID Transactions and File-Level Single-Writer Locking
DuckDB implements Multi-Version Concurrency Control (MVCC) providing ACID guarantees (Atomicity, Consistency, Isolation, Durability):
- **Single-Writer, Multiple-Reader Concurrency:** One process can hold write access to the `.duckdb` database file at any given moment, while multiple threads or processes perform concurrent read operations without locking.
- **Embedded Persistence:** All tables, performance indexes, and schema definitions reside inside a single binary file (`data/db/ragnarok.duckdb`).

---

## The RAGnarok Storage Architecture

In RAGnarok, DuckDB serves as the authoritative single source of truth for all ingestion artifacts and relational metadata:

### Relational Schema Design
1. **`documents` Table:**
   Tracks source files, SHA-256 cryptographic checksums, raw file paths, parsed Markdown artifact paths, and ingestion lifecycle states (`stored`, `parsing`, `parsed`, `failed`).
2. **`parent_chunks` Table:**
   Stores 2,000-token contextual units with structural Markdown heading breadcrumbs (`heading_path`), enabling parent expansion during generation.
3. **`chunks` Table:**
   Stores 500-token child chunks mapped directly to dense vector embeddings and BM25 inverted index positions.
4. **`processing_jobs` Table:**
   Maintains atomic state tracking across ingestion stages (`queued`, `parsing`, `chunking`, `indexing`, `completed`, `failed`).

```
                    ┌────────────────────────┐
                    │       documents        │
                    │  (checksum, raw, md)   │
                    └───────────┬────────────┘
                                │ 1
                                │
                                │ N
                    ┌───────────▼────────────┐
                    │     parent_chunks      │
                    │ (2,000 tokens, header) │
                    └───────────┬────────────┘
                                │ 1
                                │
                                │ N
                    ┌───────────▼────────────┐
                    │         chunks         │
                    │  (500 tokens, vectors) │
                    └────────────────────────┘
```

---

## Operational Advantages Over Dockerized PostgreSQL

| Dimension | External PostgreSQL (Docker) | Embedded DuckDB (RAGnarok) |
| :--- | :--- | :--- |
| **Infrastructure** | Requires Docker engine, compose daemons, port bindings | Zero external dependencies; single local file |
| **Startup Latency** | 3,000 to 10,000 ms container initialization | < 2 ms in-process connection |
| **Memory Footprint** | ~150 MB background server process | In-process footprint scaled to active query |
| **Portability** | OS socket and container bridge dependent | Instant copy-paste single file backup (`data/db/ragnarok.duckdb`) |
| **Columnar Analytics** | Requires extension modules (Citus, pg_analytics) | Native SIMD vectorized engine out of the box |
