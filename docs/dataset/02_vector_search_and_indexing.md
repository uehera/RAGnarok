# Dense Vector Search, Embedding Spaces, and High-Performance Indexing

## Fundamentals of Dense Semantic Retrieval

Dense retrieval maps unstructured natural language passages and queries into continuous high-dimensional vector spaces ($\mathbb{R}^d$), where semantic similarity correlates with geometric proximity. Unlike traditional lexical keyword search (such as BM25) which relies strictly on exact token overlaps and inverse document frequencies, dense embeddings capture synonymy, paraphrasing, semantic intent, and cross-lingual concepts.

Modern dense retrieval pipelines rely on bi-encoder architectures, where queries and document chunks are embedded independently by an encoder neural network (such as BAAI/bge-small-en-v1.5 or all-MiniLM-L6-v2) into normalized dense vectors:

$$\vec{v}_q = \text{Encoder}(q), \quad \vec{v}_d = \text{Encoder}(d)$$

---

## Vector Similarity Metrics

The degree of semantic relatedness between query vector $\vec{q}$ and passage vector $\vec{d}$ is evaluated using one of three standard distance metrics:

### 1. Cosine Similarity
Measures the cosine of the angle between two vectors, regardless of their magnitude:

$$\text{CosineSimilarity}(\vec{q}, \vec{d}) = \frac{\vec{q} \cdot \vec{d}}{\|\vec{q}\| \|\vec{d}\|} = \frac{\sum_{i=1}^d q_i d_i}{\sqrt{\sum_{i=1}^d q_i^2} \sqrt{\sum_{i=1}^d d_i^2}}$$

When vectors are $L_2$-normalized beforehand ($\|\vec{v}\| = 1$), Cosine Similarity simplifies directly to the Inner Product, enabling significant computational acceleration.

### 2. Dot Product (Inner Product)
Computes the unnormalized scalar projection. Favored when vector magnitude conveys meaningful semantic weight or document length signal:

$$\text{DotProduct}(\vec{q}, \vec{d}) = \vec{q} \cdot \vec{d} = \sum_{i=1}^d q_i d_i$$

### 3. Euclidean Distance ($L_2$ Distance)
Measures the straight-line geometric distance between two vector coordinates:

$$D_{L2}(\vec{q}, \vec{d}) = \sqrt{\sum_{i=1}^d (q_i - d_i)^2}$$

Smaller Euclidean distances correspond to higher semantic similarity.

---

## Approximate Nearest Neighbor (ANN) Indexing Algorithms

For small datasets ($N < 10,000$), exact brute-force K-Nearest Neighbor (KNN) search with flat matrix multiplications is feasible and yields 100% recall. For enterprise-scale knowledge bases ($N > 100,000$), exhaustive linear scans induce prohibitive latency ($O(N \cdot d)$). Approximate Nearest Neighbor (ANN) structures trade small margins of recall for orders-of-magnitude faster queries.

### 1. Hierarchical Navigable Small World (HNSW)
HNSW builds a multi-layer graph where lower layers contain all data points with dense connections, and upper layers form sparse highway skips. Search starts at top layers with large step sizes and navigates greedily down to fine-grained neighbors.

Key HNSW Hyperparameters:
- **$M$ (Max links per node):** Controls graph connectivity (typically 16 to 64). Higher $M$ increases recall and memory consumption.
- **$efConstruction$ (Construction beam width):** Controls indexing time quality.
- **$efSearch$ (Search beam width):** Controls runtime tradeoff between latency and recall.

### 2. Inverted File Index with Flat Quantization (IVFFlat)
Partitions the vector space into $K$ Voronoi cells using $k$-means clustering. During search, only vectors belonging to the $n_{\text{probe}}$ closest centroids are evaluated:

```
Query Vector ──► Find Nearest Centroids (nprobe=8) ──► Scan only vectors within selected Voronoi cells
```

### 3. Vector Quantization Techniques
- **Product Quantization (PQ):** Decomposes $d$-dimensional vectors into $m$ orthogonal sub-vectors, quantizing each sub-vector into codebook centroids. Compresses memory footprint by up to 95%.
- **Scalar Quantization (SQ8):** Quantizes 32-bit floating point numbers (`float32`) into 8-bit integers (`int8`), reducing memory consumption by 4x with negligible loss in retrieval recall.

---

## TurboVec: Embedded High-Performance Vector Storage

In **RAGnarok**, vector indexing is executed natively using **TurboVec** paired with **FastEmbed ONNX**:
- **Zero Heavy External Daemons:** No separate Milvus, Pinecone, or Qdrant server containers.
- **Direct Memory Mapping:** High-throughput serialized binary formats loaded directly into memory or mapped via mmap.
- **ONNX Runtime Execution:** Eliminates PyTorch C++ runtime bloat, embedding queries via optimized CPU SIMD instructions (AVX-512, Neon).
