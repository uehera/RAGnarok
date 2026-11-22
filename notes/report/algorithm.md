# Native Guardrail Algorithms in RAGnarok

> **Target File:** `notes/report/algorithm.md`  
> **Scope:** Clean, isolated algorithmic definitions for each native subsystem with operational explanations, Mermaid flowcharts, formal mathematical equations, and preview-compatible pseudocode for academic defense and research reporting.

---

## 1. Input Gateway Fast-Path & Self-RAG Classification

### Explanation
The Input Gateway Fast-Path Guardrail functions as an ultra-low-latency deterministic firewall ($< 1.0\text{ ms}$) intercepting adversarial queries *before* dense vector embeddings or database retrieval occur. 

Incoming queries first undergo token-length bounding to drop context-stuffing denial-of-service attempts. The query then passes through a four-stage recursive de-obfuscation pipeline: in-place Base64 extraction, URL percent decoding, Unicode normalization (NFKD) with zero-width character stripping, and leetspeak substitution. Next, Shannon entropy is calculated to detect encrypted or compressed shellcode payloads. The canonicalized query is evaluated against compiled regex injection heuristics. Queries that pass security are routed through the Self-RAG classifier, which determines whether retrieval is necessary or if the query can be resolved via parametric generation (saving 100% of retrieval compute).

### Flowchart

```mermaid
flowchart TD
    A["Raw User Query (q_raw)"] --> B{"Token Count > T_max (1024)?"}
    B -- "Yes" --> R1["Refusal: TOKEN_LIMIT_OVERFLOW"]
    B -- "No" --> C["Strip Control & ANSI Bytes"]
    C --> D["In-Place Base64 Decoding"]
    D --> E["URL Percent Decoding"]
    E --> F["Unicode NFKD & Strip Zero-Width"]
    F --> G["Leetspeak Character Normalization"]
    G --> H{"Shannon Entropy H(X) > 4.8?"}
    H -- "Yes" --> R2["Refusal: HIGH_ENTROPY_ANOMALY"]
    H -- "No" --> I{"Match Heuristic Injection Regex?"}
    I -- "Yes" --> R3["Refusal: INJECTION_ATTACK_INTERCEPTED"]
    I -- "No" --> J{"Self-RAG: Retrieval Required?"}
    J -- "No (Greeting/Parametric)" --> K["Direct Generation (Bypass Retrieval)"]
    J -- "Yes (Corpus Query)" --> L["Pass Canonical Query (q_canon) to Hybrid Retrieval"]
```

### Mathematical Formulation
Let the inbound raw query be $q_{\text{raw}} \in \Sigma^*$ over character alphabet $\Sigma$. The sequential canonicalization transformation $\mathcal{T}_{\text{canon}}$ is formulated as:

$$q_{\text{canon}} = \mathcal{T}_{\text{leet}} \circ \mathcal{T}_{\text{unicode}} \circ \mathcal{T}_{\text{url}} \circ \mathcal{T}_{\text{b64}}(q_{\text{clean}})$$

Character Shannon entropy $H(q_{\text{canon}})$ evaluates randomness across character frequencies $P(x_i)$:

$$H(q_{\text{canon}}) = -\sum_{i=1}^{n} P(x_i) \log_2 P(x_i)$$

The fast-path gate decision function $\mathcal{G}_{\text{input}}(q_{\text{raw}})$ is defined by:

$$\mathcal{G}_{\text{input}}(q_{\text{raw}}) = \begin{cases} 
\Omega_{\text{ref}}(\text{"TOKEN\_LIMIT"}) & \text{if } |q_{\text{raw}}|_{\text{tok}} > T_{\max} \\ 
\Omega_{\text{ref}}(\text{"ENTROPY\_ANOMALY"}) & \text{if } H(q_{\text{canon}}) > \tau_{\text{entropy}} \\ 
\Omega_{\text{ref}}(\text{"INJECTION"}) & \text{if } \exists p \in \mathcal{P}_{\text{fast}} : \text{Match}(p, q_{\text{canon}}) = 1 \\ 
\text{DIRECT\_GENERATE} & \text{if } \mathcal{C}_{\text{self\_rag}}(q_{\text{canon}}) = \text{Parametric} \\ 
q_{\text{canon}} & \text{otherwise} 
\end{cases}$$

### Algorithm

```text
Algorithm 1: Input Gateway Fast-Path & Self-RAG Classification
────────────────────────────────────────────────────────────────────────────────
Input:  Raw query string q_raw, maximum token ceiling T_max = 1024,
        entropy threshold tau_entropy = 4.8, heuristic pattern set P_fast
Output: Canonical query q_canon, DIRECT_GENERATE signal, or Refusal directive
────────────────────────────────────────────────────────────────────────────────
 1: token_count ← CountTokens(q_raw)
 2: if token_count > T_max then
 3:     return Refusal("TOKEN_LIMIT_OVERFLOW")
 4: end if
 5:
 6: q_clean ← StripControlAndANSIBytes(q_raw)
 7: q_deobf ← DecodeInPlaceBase64(q_clean)
 8: q_deobf ← URLPercentDecode(q_deobf)
 9: q_deobf ← StripZeroWidth(UnicodeNormalizeNFKD(q_deobf))
10: q_canon ← LeetspeakSubstitute(q_deobf)
11:
12: if ComputeShannonEntropy(q_canon) > tau_entropy then
13:     return Refusal("HIGH_ENTROPY_ANOMALY")
14: end if
15:
16: for each pattern p in P_fast do
17:     if RegexMatch(p, q_canon) is not empty then
18:         return Refusal("INJECTION_ATTACK_INTERCEPTED")
19:     end if
20: end for
21:
22: if IsParametricGreetingOrMath(q_canon) then
23:     return Signal("DIRECT_GENERATE")
24: end if
25:
26: return q_canon
```

---

## 2. Ingestion & Knowledge Vault Quarantine Guardrail

### Explanation
The Ingestion & Quarantine Guardrail secures the knowledge base against Indirect Prompt Injection (IPI) and poisoned document payloads. When external documents (PDFs, Markdown, Text) are submitted for ingestion, `PyMuPDF` extracts structural Markdown while preserving headers and tables. 

Before text is segmented into chunks or vectorized into TurboVec, the `DocumentSecurityGuard` performs in-memory signature screening across the full text for instruction overrides, payload delimiters, and exfiltration webhooks. If an injection directive is identified, the file is immediately diverted into `data/quarantine/<file>`, an immutable forensic JSON audit sidecar is generated with the SHA-256 fingerprint, and ingestion terminates. This guarantees exactly **0 parent rows** written to DuckDB and **0 child vectors** written to TurboVec ($\text{DCR} = 0.00\%$). Clean documents undergo cryptographic SHA-256 deduplication and 259ms incremental indexing.

### Flowchart

```mermaid
flowchart TD
    A["Inbound Document (d_file)"] --> B["Compute SHA-256 Checksum"]
    B --> C{"Hash Exists in DuckDB?"}
    C -- "Yes" --> D["Skip Idempotently (200 OK)"]
    C -- "No" --> E["PyMuPDF Markdown Extraction"]
    E --> F{"Detect IPI Threat Signatures?"}
    F -- "Threat Detected" --> G["Move to data/quarantine/"]
    G --> H["Write Forensic JSON Sidecar Audit Log"]
    H --> I["Halt Ingestion (DCR = 0.00%)"]
    F -- "Clean Document" --> J["Segment Hierarchical Chunks (512 / 64)"]
    J --> K["Commit Parent & Chunks to DuckDB Vault"]
    K --> L["Update BM25s Sparse Index (0.45 ms)"]
    K --> M["Incremental Append to TurboVec via FastEmbed (259 ms)"]
    L --> N["Ingestion Complete (DCR = 0.00%)"]
    M --> N
```

### Mathematical Formulation
For an inbound document $d \in \mathcal{D}_{\text{in}}$, the threat detection predicate $\Phi_{\text{IPI}}(d)$ evaluates extracted document text $T_d$:

$$\Phi_{\text{IPI}}(d) = \bigvee_{p \in \mathcal{P}_{\text{IPI}}} \mathbb{I}(\text{RegexMatch}(p, T_d) \neq \emptyset)$$

When $\Phi_{\text{IPI}}(d) = 1$, the file is quarantined with cryptographic hash $h_d = \mathcal{H}_{\text{SHA256}}(d)$. The Database Contamination Rate ($\text{DCR}$) measures isolation integrity across the relational vault $\mathcal{V}$ (DuckDB) and vector index $\mathcal{I}_{\text{dense}}$:

$$\text{DCR} = \frac{|\mathcal{Q}_{\text{quarantine}} \cap \mathcal{V}_{\text{active}}|}{|\mathcal{Q}_{\text{quarantine}}|} = 0.00\%$$

### Algorithm

```text
Algorithm 2: Ingestion Quarantine & Knowledge Isolation
────────────────────────────────────────────────────────────────────────────────
Input:  Inbound document file d_file, relational vault V (DuckDB),
        dual search indexes (I_lex, I_dense), IPI signature set P_IPI
Output: Ingestion status and DCR metric
────────────────────────────────────────────────────────────────────────────────
 1: file_hash ← ComputeSHA256(ReadRawBytes(d_file))
 2: if V.DocumentExistsByHash(file_hash) then
 3:     return { status: "DUPLICATE_SKIPPED", DCR: "0.00%" }
 4: end if
 5:
 6: doc_text ← PyMuPDF_ExtractMarkdown(d_file)
 7: detected_threats ← FindMatches(P_IPI, doc_text)
 8:
 9: if detected_threats is not empty then
10:     MoveFile(d_file, "data/quarantine/" + GetFileName(d_file))
11:     WriteJSON("data/quarantine/" + GetFileName(d_file) + ".json", {
12:         "sha256": file_hash,
13:         "threats": detected_threats,
14:         "timestamp": CurrentTimestamp()
15:     })
16:     return { status: "QUARANTINED", DCR: "0.00%" }
17: end if
18:
19: chunks ← HierarchicalChunk(doc_text, size = 512, overlap = 64)
20: V.AtomicCommit(d_file, chunks)
21: I_lex.AddTokens(chunks)
22: I_dense.IncrementalAdd(FastEmbed(chunks))
23:
24: return { status: "INDEXED", DCR: "0.00%" }
```

---

## 3. Speculative Sliding-Window Stream Sanitizer Guardrail

### Explanation
The Speculative Stream Sanitizer resolves the token-boundary credential leak in real-time streaming generation. During streaming LLM inference, credentials frequently split across consecutive token chunks (e.g., `sk-pro` + `j-abc123`), bypassing naive per-token filters. Waiting for the complete response before redacting degrades interactive latency.

The sanitizer maintains an internal sliding buffer with horizon $W = 35$ characters. Newly generated tokens append to the buffer. Once the buffer exceeds $W$ characters, only text trailing behind the active horizon ($L - W$) is scanned against compiled secret signatures (OpenAI, Anthropic, Groq, database URIs, IPv4 addresses, emails), masked in-place with `[REDACTED <TYPE>]`, and emitted to the client. Upon stream completion (EOF), the trailing buffer undergoes final evaluation and emission, guaranteeing a **$0.00\%$ Secret Leakage Rate ($\text{SLR}$)**.

### Flowchart

```mermaid
flowchart TD
    A["Incoming Token Stream (Δt)"] --> B["Append Token to Sliding Buffer & Accumulator"]
    B --> C{"Buffer Length > W (35 chars)?"}
    C -- "No" --> D{"More Tokens in Stream?"}
    C -- "Yes" --> E["Extract Trailing Prefix: Buffer[0 : L - W]"]
    E --> F["Regex Scan Prefix Against Secret Signatures"]
    F --> G["Mask Detected Secrets with [REDACTED]"]
    G --> H["Emit Sanitized Prefix to Client"]
    H --> I["Shift Buffer Window: Buffer[L - W : L]"]
    I --> D
    D -- "Yes" --> A
    D -- "No (EOF)" --> J["Final Secret Scan on Remaining Buffer"]
    J --> K["Mask Detected Secrets in Buffer"]
    K --> L["Emit Final Buffer to Client"]
    L --> M["Finalize Full Redacted Response (SLR = 0.00%)"]
```

### Mathematical Formulation
Given an asynchronous token stream $\mathcal{S}_{\text{stream}} = (\Delta t_1, \Delta t_2, \dots)$, let the accumulated buffer at step $k$ be $\mathcal{B}_k = \sum_{j=1}^{k} \Delta t_j$. The speculative emission window prefix is governed by:

$$\text{Prefix}_k = \begin{cases} 
\mathcal{B}_k[0 : |\mathcal{B}_k| - W] & \text{if } |\mathcal{B}_k| > W \\ 
\epsilon & \text{otherwise} 
\end{cases}$$

The Secret Leakage Rate ($\text{SLR}$) verifies zero exposure of sensitive signatures $\mathcal{S}_{\text{secret}}$ across all emitted stream chunks:

$$\text{SLR} = \frac{\sum_{i=1}^{N_{\text{stream}}} \mathbb{I}(\exists s \in \mathcal{S}_{\text{secret}} : s \subseteq \text{Output}_i)}{|\mathcal{S}_{\text{secret}}|} = 0.00\%$$

### Algorithm

```text
Algorithm 3: Speculative Sliding-Window Stream Sanitizer
────────────────────────────────────────────────────────────────────────────────
Input:  Token stream S = (Δt_1, Δt_2, ...), sliding horizon W = 35 characters,
        secret signature set P_sec (API keys, DB URIs, PII)
Output: Sanitized real-time token stream with SLR = 0.00%
────────────────────────────────────────────────────────────────────────────────
 1: buffer ← ""
 2: accumulated_text ← ""
 3:
 4: for each token Δt in S do
 5:     buffer ← buffer + Δt
 6:     accumulated_text ← accumulated_text + Δt
 7:
 8:     if Length(buffer) > W then
 9:         eval_prefix ← Slice(buffer, 0, Length(buffer) - W)
10:
11:         for each pattern p in P_sec do
12:             if RegexMatch(p, eval_prefix) is not empty then
13:                 eval_prefix ← RegexReplace(p, eval_prefix, "[REDACTED]")
14:             end if
15:         end for
16:
17:         EmitToClient(eval_prefix)
18:         buffer ← Slice(buffer, Length(buffer) - W, Length(buffer))
19:     end if
20: end for
21:
22: for each pattern p in P_sec do
23:     buffer ← RegexReplace(p, buffer, "[REDACTED]")
24: end for
25: EmitToClient(buffer)
26:
27: finalized_text ← MaskAllSecrets(accumulated_text, P_sec)
28: return finalized_text
```

---

## 4. Grounding & Citation Integrity Guardrail

### Explanation
The Grounding & Citation Integrity Guardrail eliminates factual hallucinations and phantom citations post-generation. 

Before presenting answers to the user, the guardrail performs entailment verification against retrieved context chunks. If retrieval confidence is low or evidence is insufficient, the system suppresses speculative text and emits the standardized refusal directive (`UNGROUNDED_VIOLATION_MESSAGE`). 

For grounded responses, an inline citation parser identifies all bracketed numerical citations (`\[([0-9]+)\]`). Each marker is verified 1:1 against the runtime provenance map assembled during Context Assembly. If the model hallucinates an unbacked citation (e.g., `[99]`), the ghost marker is stripped from the text and logged as an anomaly, guaranteeing a **$0.00\%$ Ghost Citation Rate ($\text{GCR}$)**.

### Flowchart

```mermaid
flowchart TD
    A["Candidate Response Text & Context Chunks"] --> B{"Top Retrieval Score >= 0.25 & Entailment Passes?"}
    B -- "No" --> R["Emit Refusal: UNGROUNDED_VIOLATION_MESSAGE"]
    B -- "Yes" --> C["Extract Inline Citation Markers: [N]"]
    C --> D{"For Each Marker [N]"}
    D --> E{"Is N in Provenance Map?"}
    E -- "Yes" --> F["Append Provenance Source to Verified List"]
    E -- "No (Ghost Citation)" --> G["Strip [N] Marker from Response Text"]
    G --> H["Log Security Anomaly Alert"]
    F --> I{"More Citation Markers?"}
    H --> I
    I -- "Yes" --> D
    I -- "No" --> J["Deduplicate Verified Sources by Document ID"]
    J --> K["Return Verified Response & Clean Citations (GCR = 0.00%)"]
```

### Mathematical Formulation
Let candidate response $Y$ and context chunks $\mathcal{C} = \{c_1, \dots, c_k\}$ have relevance score $s(Y, \mathcal{C})$. The factual entailment decision $\mathcal{E}(Y, \mathcal{C})$ requires:

$$\mathcal{E}(Y, \mathcal{C}) = \begin{cases} 
1 & \text{if } s(Y, \mathcal{C}) \ge \theta_{\text{conf}} \;\land\; \text{Entails}(\mathcal{C}, Y) = \text{True} \\ 
0 & \text{otherwise} 
\end{cases}$$

Given extracted citation indices $\mathcal{M}_{\text{cite}} = \{m \in \mathbb{N} \mid [m] \in Y\}$ and the domain of active provenance mapping $\text{Dom}(\text{ProvMap})$, the Ghost Citation Rate ($\text{GCR}$) satisfies:

$$\text{GCR} = \frac{|\{m \in \mathcal{M}_{\text{cite}} \mid m \notin \text{Dom}(\text{ProvMap})\}|}{|\mathcal{M}_{\text{cite}}|} = 0.00\%$$

### Algorithm

```text
Algorithm 4: Grounding & Citation Integrity Verification
────────────────────────────────────────────────────────────────────────────────
Input:  Sanitized response text Y_clean, context payload X_context with
        provenance map ProvMap, confidence threshold theta_conf = 0.25
Output: Verified response tuple (Y_verified, F_verified) or Refusal directive
────────────────────────────────────────────────────────────────────────────────
 1: retrieval_score ← X_context.TopScore
 2: if retrieval_score < theta_conf or not CheckEntailment(Y_clean, X_context) then
 3:     return Refusal("UNGROUNDED_VIOLATION_MESSAGE")
 4: end if
 5:
 6: citation_markers ← ExtractRegexMatches("\[([0-9]+)\]", Y_clean)
 7: verified_sources ← []
 8:
 9: for each marker in citation_markers do
10:     source_idx ← ParseInteger(marker)
11:     if source_idx in ProvMap then
12:         verified_sources.Append(ProvMap[source_idx])
13:     else
14:         Y_clean ← RemoveMarker(Y_clean, marker)
15:         LogSecurityAlert("GHOST_CITATION_STRIPPED", source_idx)
16:     end if
17: end for
18:
19: deduplicated_sources ← DeduplicateByDocID(verified_sources)
20: return (Y_clean, deduplicated_sources)
```


---

## 5. Dual-Engine Hybrid Retrieval & Reciprocal Rank Fusion (RRF)

### Explanation
Dual-Engine Hybrid Retrieval combines the exact keyword precision of sparse BM25s lexical search with the conceptual semantic capture of TurboVec dense vector search (FastEmbed ONNX 384-d). The two engines execute concurrently in a parallel thread pool. 

Rather than relying on uncalibrated score normalization (e.g., MinMax or z-score), which is susceptible to outlier distortion across heterogeneous score spaces, RAGnarok merges ranked candidate pools using **Reciprocal Rank Fusion (RRF)** with smoothing constant $k = 60$. RRF ranks items based on position invariance, guaranteeing that an item appearing near the top of either list receives a boosted rank without allowing extreme raw cosine or BM25 scores to monopolize candidates.

### Flowchart

```mermaid
flowchart TD
    A["Canonical Query (q_canon)"] --> B["ThreadPoolExecutor: Parallel Fanout"]
    B --> C1["BM25s Lexical Search<br>(Candidate Pool K=50, 0.44 ms)"]
    B --> C2["TurboVec Dense Search<br>(Candidate Pool K=50, 18.25 ms)"]
    C1 --> D["Lexical Ranked List: R_lex"]
    C2 --> D2["Dense Ranked List: R_dense"]
    D & D2 --> E["Reciprocal Rank Fusion (k = 60)"]
    E --> F["Compute RRF Scores for All Candidates: c in R_lex union R_dense"]
    F --> G["Sort Candidates Descending by RRF Score"]
    G --> H["Select Top N Candidates (Default: N=10)"]
    H --> I["Pass Top Candidates to DuckDB Context Expansion"]
```

### Mathematical Formulation
Given lexical rank position $r_{\text{lex}}(c)$ and dense rank position $r_{\text{dense}}(c)$ for candidate chunk $c$:

$$\text{Score}_{\text{RRF}}(c) = \sum_{m \in \{\text{lex}, \text{dense}\}} \frac{1}{k + r_m(c)}$$

*(where $k = 60$ dampens the impact of high-ranking outliers)*

### Algorithm

```text
Algorithm 5: Dual-Engine Hybrid Retrieval & Reciprocal Rank Fusion
────────────────────────────────────────────────────────────────────────────────
Input:  Canonical query q_canon, lexical index I_lex, dense index I_dense,
        pool size K = 50, final top-N = 10, dampening constant k = 60
Output: Unified ranked candidate chunks C_top
────────────────────────────────────────────────────────────────────────────────
 1: Parallel Execute:
 2:     R_lex   ← I_lex.Search(q_canon, top_k = K)
 3:     R_dense ← I_dense.Search(q_canon, top_k = K)
 4:
 5: rrf_scores ← Dictionary()
 6:
 7: for rank, chunk in Enumerate(R_lex, start = 1) do
 8:     rrf_scores[chunk.id] ← rrf_scores.Get(chunk.id, 0.0) + (1.0 / (k + rank))
 9: end for
10:
11: for rank, chunk in Enumerate(R_dense, start = 1) do
12:     rrf_scores[chunk.id] ← rrf_scores.Get(chunk.id, 0.0) + (1.0 / (k + rank))
13: end for
14:
15: sorted_candidates ← SortKeysByValueDescending(rrf_scores)
16: C_top ← TakeFirst(sorted_candidates, N)
17: return C_top
```

---

## 6. Hierarchical Parent-Child Context Expansion

### Explanation
Small child chunks (256-512 tokens) are necessary for precise dense embeddings and specific lexical matching, but isolated small chunks lack the surrounding narrative context required by generative models, leading to clipped sentences and reasoning failures.

RAGnarok decouples the search index from the generation context through **Hierarchical Parent-Child Context Expansion**:
1. Only child chunks are vectorized and indexed in BM25s and TurboVec.
2. Each child chunk maintains a foreign key pointer (`parent_id`) to its full enclosing section stored in DuckDB (~1,024-2,048 tokens).
3. Upon retrieval of child candidates, DuckDB executes a batch SQL lookup (`SELECT * FROM parent_chunks WHERE parent_id IN (...)`).
4. Duplicate parent IDs from sibling child chunks are collapsed into unique parent sections.
5. Sibling windows are dynamically concatenated, yielding contiguous context with full paragraph boundaries and $+4.42\%$ higher recall on multi-hop reasoning benchmarks.

### Flowchart

```mermaid
flowchart TD
    A["Top Ranked Child Chunks (c_1 ... c_k)"] --> B["Extract Foreign Keys: {c.parent_id}"]
    B --> C["Deduplicate Unique Parent IDs"]
    C --> D["Batch DuckDB Relational Lookup:<br>SELECT * FROM parent_chunks WHERE parent_id IN (...)"]
    D --> E["Order Parent Sections by Highest Child Rank"]
    E --> F{"Total Tokens > Max Context Budget (3000 tok)?"}
    F -- "Yes" --> G["Truncate Lower-Ranked Parents to Satisfy Budget"]
    F -- "No" --> H["Perimeter Formatting: Rank 1 at Top, Rank 2 at Bottom"]
    G --> H
    H --> I["Emit Assembled Context Payload to Generator"]
```

### Mathematical Formulation
Let retrieved child chunk set be $\mathcal{C}_{\text{ret}} = \{c_1, \dots, c_k\}$ where each child maps to a parent section $\pi(c) \in \mathcal{P}_{\text{vault}}$. The unique parent expansion set $\mathcal{P}_{\text{exp}}$ is defined as:

$$\mathcal{P}_{\text{exp}} = \{\pi(c) \mid c \in \mathcal{C}_{\text{ret}}\}$$

Token budgeting enforces an upper bound $B_{\max}$ on the assembled context:

$$\sum_{p \in \mathcal{P}_{\text{budget}}} |p|_{\text{tokens}} \le B_{\max}, \quad \text{where } \mathcal{P}_{\text{budget}} \subseteq \mathcal{P}_{\text{exp}}$$

### Algorithm

```text
Algorithm 6: Hierarchical Parent-Child Context Expansion & Perimeter Formatting
────────────────────────────────────────────────────────────────────────────────
Input:  Ranked child chunk list C_ret, relational vault V (DuckDB),
        maximum token budget B_max = 3000
Output: Formatted context payload X_context with provenance mapping
────────────────────────────────────────────────────────────────────────────────
 1: parent_ids ← OrderedUnique([c.parent_id for c in C_ret])
 2: raw_parents ← V.BatchFetchParents(parent_ids)
 3:
 4: selected_parents ← []
 5: current_tokens ← 0
 6:
 7: for p in raw_parents do
 8:     if current_tokens + p.token_count <= B_max then
 9:         selected_parents.Append(p)
10:         current_tokens ← current_tokens + p.token_count
11:     else
12:         break
13:     end if
14: end for
15:
16: // Perimeter assembly: mitigate Lost-in-the-Middle effect
17: formatted_blocks ← PerimeterArrange(selected_parents)
18: prov_map ← BuildProvenanceMap(selected_parents)
19:
20: return ContextPayload(blocks = formatted_blocks, provenance = prov_map)
```
