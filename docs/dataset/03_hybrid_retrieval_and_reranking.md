# Hybrid Retrieval Architectures, Reciprocal Rank Fusion, and Cross-Encoder Reranking

## The Need for Hybrid Search in Enterprise RAG

Pure dense vector retrieval is susceptible to specific failure modes:
1. **Keyword Specificity:** Inability to match exact alphanumeric codes, product IDs, UUIDs, or niche library functions (`re.sub`, `CVE-2024-1234`).
2. **Domain Vocabulary Shift:** Dense embeddings trained on general web corpora struggle with specialized clinical, financial, or engineering jargon.
3. **Out-of-Vocabulary Terms:** Misspellings and novel acronyms.

Conversely, traditional sparse lexical search (such as BM25) fails when users ask questions using different vocabulary than the document author (*"remuneration"* vs *"salary"*, or *"reboot machine"* vs *"power cycle system"*).

**Hybrid Retrieval** unifies sparse lexical search with dense vector embeddings, achieving Pareto-optimal recall across diverse query types.

---

## Lexical Search: The Okapi BM25s Algorithm

The BM25 algorithm computes lexical relevance scores using non-linear term frequency saturation and document length normalization:

$$\text{Score}(D, Q) = \sum_{i=1}^n \text{IDF}(q_i) \cdot \frac{f(q_i, D) \cdot (k_1 + 1)}{f(q_i, D) + k_1 \cdot \left(1 - b + b \cdot \frac{|D|}{\text{avgdl}}\right)}$$

Where:
- $f(q_i, D)$: Raw term frequency of word $q_i$ in passage $D$.
- $|D|$: Length of document $D$ in tokens.
- $\text{avgdl}$: Average document length across the entire indexed collection.
- $k_1$: Term frequency saturation parameter (typically $1.2 \le k_1 \le 2.0$). Prevents a single repeated word from dominating the score.
- $b$: Document length normalization parameter (typically $b = 0.75$). Penalizes overly verbose documents.
- $\text{IDF}(q_i)$: Inverse document frequency, dampening ubiquitous stop words.

In RAGnarok, **BM25s** provides pure-Python, zero-dependency, vectorized inverted indexing with blazing-fast serialization to disk (`data/indexes/bm25/`).

---

## Fusing Results: Reciprocal Rank Fusion (RRF)

Because BM25 scores (unbounded positive reals) and dense cosine similarity scores ($-1.0$ to $+1.0$) occupy entirely different scale distributions, direct score summation produces biased results.

**Reciprocal Rank Fusion (RRF)** is an unsupervised, parameter-free rank aggregation method that operates purely on positional ranks rather than raw scores:

$$\text{RRF\_Score}(d) = \sum_{m \in M} \frac{1}{k + r_m(d)}$$

Where:
- $M$: The set of retrieval models (e.g., $M = \{\text{BM25s}, \text{TurboVec}\}$).
- $r_m(d)$: The 1-based rank position of candidate document $d$ within model $m$'s result list.
- $k$: Smoothing constant (typically $k = 60$), mitigating the disproportionate advantage of top-ranked items.

```
BM25 Top-50 List    ──┐
                      ├──► Reciprocal Rank Fusion (k=60) ──► Unified Candidate Pool (Top-15)
Dense Top-50 List   ──┘
```

RRF guarantees that a document appearing near the top of both lists receives an overwhelmingly high fused score, while documents appearing in only one list remain gracefully supported.

---

## Stage 2: Cross-Encoder Reranking

While bi-encoders produce separate embeddings for query and document to enable fast vector indexing, they cannot model fine-grained token-to-token cross-attention between question words and passage words.

A **Cross-Encoder** takes the concatenated pair `[CLS] Query [SEP] Passage [SEP]` and processes them through all transformer attention layers simultaneously:

```
[Query + Passage] ──► Full Cross-Attention Layers ──► Dense Linear Head ──► Relevance Probability [0.0 - 1.0]
```

### FlashRank ONNX Reranking
To eliminate PyTorch latency and GPU requirements, RAGnarok integrates **FlashRank** running the `ms-marco-MiniLM-L-12-v2` model in 4-bit/8-bit quantized ONNX format:
- Evaluates top $N = 15$ fused candidates.
- Re-scores and re-sorts into top $K = 10$ highest-confidence passages.
- Executes in $< 25\text{ ms}$ on commodity CPU cores.

---

## Two-Stage Parent-Child Context Expansion

Passing tiny 500-token chunks directly to the generator strips away necessary background context, leading to incoherent responses. Conversely, indexing massive 2,000-token chunks dilutes vector specificity.

RAGnarok solves this through **Two-Stage Parent-Child Chunking**:
1. **Search Phase:** Small child chunks ($500$ tokens) are embedded and indexed for precise retrieval.
2. **Expansion Phase:** When a child chunk matches, the system retrieves its encompassing **Parent Chunk** ($2,000$ tokens) from DuckDB.
3. **Deduplication Phase:** Overlapping parent blocks are merged to maximize token budget efficiency without repetition.
