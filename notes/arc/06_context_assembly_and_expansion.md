# 06 — Context Assembly & Expansion Architecture

> **Lifecycle Stage:** Step 3 in Read Path (Query Pipeline)  
> **Source Module:** `src/ragnarok/context/expansion.py`, `src/ragnarok/context/formatter.py`  
> **Responsibility:** Child-to-parent expansion, token budgeting, and perimeter context ordering ("Lost in the Middle").

---

## 1. Architectural Problem: Context Dilution & "Lost in the Middle"

Passing raw child chunks (~$256$ tokens) directly to an LLM creates fragmented answers because surrounding context, section headers, and qualifying sentences are cut off at chunk boundaries. 

Conversely, passing all related parent sections without careful structuring introduces two critical vulnerabilities:
1. **Context Window Overflow:** Exceeding the model's optimal prompt length increases latency and causes prompt truncation.
2. **"Lost in the Middle" Degradation:** Empirical research shows language models recall information positioned at the extreme beginning and end of long prompts significantly better than information in the middle. Passages placed in the center of long prompts are often overlooked by the LLM.

RAGnarok implements a **Context Assembly & Perimeter Formatting Engine**:
* **Parent Expansion:** Maps top-ranked child hits back to complete parent chunks (~$1,024$ tokens) in DuckDB.
* **Deduplication & Budgeting:** Deduplicates parent IDs and truncates content to fit within a strict token budget.
* **Perimeter Context Placement:** Strategically positions the highest-ranked evidence chunks at the start and end of the context payload.

---

## 2. Component Topology & Assembly Flow

```mermaid
flowchart TD
    CAND["Top-10 Child Candidates<br>(from 05_retrieval)"] --> DEDUP["Parent ID Deduplicator<br>(Extracts unique parent_ids in rank order)"]
    
    DEDUP --> DDB[("DuckDB Relational Query<br>SELECT * FROM parent_chunks<br>WHERE parent_id IN (...)")]
    DDB --> PAR["Expanded Parent Records<br>(Full section text + Heading hierarchy)"]
    
    PAR --> BUD{"Token Budget Controller<br>(Max Budget: e.g. 3000 tokens)"}
    BUD -- "Fits Budget" --> ORD["Perimeter Context Reorderer"]
    BUD -- "Exceeds Budget" --> TRUNC["Prune Lowest-Ranked Parents"] --> ORD
    
    subgraph PerimeterOrdering["Perimeter Context Layout"]
        direction TB
        ORD --> P_TOP["Position 1: Rank 1 Chunk (Highest Confidence)"]
        ORD --> P_MID["Position 2..N-1: Lower-Ranked Context Chunks"]
        ORD --> P_END["Position N: Rank 2 Chunk (Second Highest Confidence)"]
    end
    
    PerimeterOrdering --> FORM["Citation Indexer & Formatter<br>• Injects [1], [2], [3] markers<br>• Builds Runtime Provenance Map"]
    FORM --> OUT["Assembled Context Payload<br>Dispatched to 07_generation"]
```

---

## 3. Technical Invariants & Mechanics

### 3.1 Relational Parent Expansion
* **Deduplication:** When multiple child chunks originate from the same parent section, the expander extracts unique `parent_id` foreign keys while preserving the highest candidate rank.
* **Batch DuckDB Read:** Executes a single parameterized relational query against `parent_chunks`:
  ```sql
  SELECT parent_id, doc_id, content, heading_path, metadata 
  FROM parent_chunks 
  WHERE parent_id IN (?, ?, ...)
  ```
  This restores complete paragraphs, formulas, code snippets, and section breadcrumbs in $< 15\text{ ms}$.

### 3.2 Dynamic Token Budgeting
* **Token Estimator:** Uses fast BPE token estimation (`tiktoken`) to measure exact prompt token consumption.
* **Budget Ceiling:** Caps total context length (configurable, default $3,000$ tokens). If adding another parent chunk would breach the ceiling, lower-ranked parent passages are pruned, leaving a buffer for generation tokens.

### 3.3 Perimeter Context Ordering ("Lost in the Middle" Solution)
To maximize LLM attention recall across long context windows:
* **Perimeter Distribution Algorithm:**
  * Chunk with **Rank 1** is placed at the **beginning** of the context block (first tokens seen after instructions).
  * Chunk with **Rank 2** is placed at the **end** of the context block (immediate recency prior to the query).
  * Chunks with **Rank 3..N** are positioned in the interior.
* **Citation Index Binding:** Each assembled parent chunk receives a sequential bracketed marker (`[1]`, `[2]`, `[3]`).
* **Provenance Map:** Simultaneously constructs an immutable lookup dictionary binding each number to its source document, file path, section title, and parent ID.

---

## 4. Failure Modes & Edge Cases

| Edge Case | Scenario | System Safeguard |
| :--- | :--- | :--- |
| **All Children from One Parent** | Top 10 child hits all map to a single large parent section. | Deduplicator condenses to 1 parent chunk, saving 90% of token budget and preventing redundant duplication. |
| **Single Huge Parent Chunk** | A parent section approaches the entire 3,000-token budget. | Truncation retains the parent but limits additional chunks, logging a warning to diagnostics. |
| **No Parent Chunks Found** | DuckDB record missing (index corruption). | Expander logs critical error, falls back to raw child chunk text, and marks context for reindexing. |
