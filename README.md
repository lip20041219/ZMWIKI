# ZMRAG

**Deterministic-first RAG Retrieval & Evaluation Layer**

A model-agnostic, deterministic-first retrieval / ranking / evaluation layer
for RAG systems. It turns a collection of documents into an indexed, queryable
corpus and answers "which chunks are most relevant to this query?" with ranked
`SearchResult`s and offline evaluation metrics.

ZMRAG focuses on **retrieval, ranking, context construction, and offline
evaluation**. It is **not** a chatbot, an LLM answer-generation framework, or
an Agent framework — those belong to upper-layer systems.

## Features

- Document ingestion (plain text, `.txt` files, directories)
- Fixed-size chunking
- Lexical (BM25) retrieval
- Dense (embedding) retrieval
- Hybrid retrieval (dense + lexical)
- Weighted score fusion
- Reciprocal Rank Fusion (RRF)
- Deterministic reranking
- Optional real semantic embeddings (sentence-transformers)
- Optional Cross-Encoder reranking
- Context building
- Offline retrieval evaluation
- Hit Rate / Recall / Precision / MRR / MAP / NDCG (optional `@K` cutoffs)
- Dependency-injected, unit-tested, network-free by default

## Architecture

```
Documents
   ↓
Ingestion            (text / .txt / directory → Document)
   ↓
Chunking             (Document → Chunk[])
   ↓
Embedding ───────────────►  Dense index    Lexical (BM25) index
   │                                        │
   └────────────►  Dense / Lexical / Hybrid Retrieval
                               ↓
                         Fusion (Weighted / RRF)
                               ↓
                        Optional Reranking
                               ↓
                       Context Construction
                               ↓
                             Evaluation
```

`retrieve()` runs the shared retrieval path and returns a ranked
`list[SearchResult]`; `query()` is that same path followed by context packing,
so the two never diverge.

## Installation

Default install — deterministic, **zero ML dependencies, no model download,
no API key**:

```bash
pip install -e .
```

Development (adds pytest):

```bash
pip install -e ".[dev]"
```

Semantic (adds `sentence-transformers` for the optional real-model features):

```bash
pip install -e ".[semantic]"
```

`sentence-transformers` is an optional dependency — it is not part of the
default or `dev` installs, so default setups and CI stay ML-free.

## Quick Start

Deterministic retrieval end to end, no model, no network, no API key:

```python
from zmrag import (
    DeterministicRAGPipeline,
    DeterministicReranker,
    DeterministicEmbedder,
    DeterministicContextBuilder,
    DenseRetriever,
    LexicalRetriever,
    HybridRetriever,
    FixedSizeChunker,
    InMemoryVectorStore,
    InMemoryLexicalIndex,
    WeightedScoreFusion,
    PlainTextIngestor,
)

# text -> Document
text = (
    "ZMRAG is a deterministic retrieval and ranking layer for RAG systems. "
    "Retrieval turns a query into ranked chunks. Chunking splits a document into pieces. "
    "Embedding maps text to vectors for dense retrieval. Lexical retrieval uses BM25. "
    "Hybrid retrieval fuses both dense and lexical signals."
)
doc = PlainTextIngestor().ingest(document_id="quickstart-1", text=text)

# indexing + retrieval stages
chunker = FixedSizeChunker(chunk_size=90)
embedder = DeterministicEmbedder(dimension=8)   # deterministic stand-in, not a real model
vector_store = InMemoryVectorStore()
lexical_index = InMemoryLexicalIndex()
dense = DenseRetriever(embedder=embedder, store=vector_store)
lexical = LexicalRetriever(index=lexical_index)
retriever = HybridRetriever(
    dense_retriever=dense,
    lexical_retriever=lexical,
    fusion=WeightedScoreFusion(),
)

pipeline = DeterministicRAGPipeline(
    chunker=chunker,
    embedder=embedder,
    vector_store=vector_store,
    lexical_index=lexical_index,
    retriever=retriever,
    reranker=DeterministicReranker(),           # deterministic lexical-overlap reranker
    context_builder=DeterministicContextBuilder(),
)

pipeline.ingest(doc)

# query -> ranked SearchResult[]
results = pipeline.retrieve(query="dense retrieval", top_k=3, candidate_k=33)
for r in results:
    print(r.chunk.id, round(r.score, 4), r.chunk.text[:40])
```

ZMRAG is a **library**, not a standalone command-line product. You compose the
stages above through the Python API (or run the benchmark scripts in
[Evaluation & Benchmarks](#evaluation--benchmarks)).

## Retrieval

- **Lexical** — BM25 over a lexical index; term-overlap based, no model.
- **Dense** — embeddings plus cosine search in a vector store.
- **Hybrid** — combines dense and lexical signals through a fusion strategy
  (weighted score fusion or Reciprocal Rank Fusion) into one ranked list.

A reranker narrows a candidate pool further. The default `DeterministicReranker`
is a deterministic lexical-overlap stand-in for tests and local development —
**not** a semantic reranker.

## Semantic Retrieval (optional)

Swap the deterministic embedder for real sentence-transformers embeddings:

```python
from zmrag.semantic import SemanticEmbeddingProvider  # not in the top-level public API

embedder = SemanticEmbeddingProvider(
    model_source="sentence-transformers/all-MiniLM-L6-v2"
)
```

- Model: `sentence-transformers/all-MiniLM-L6-v2` (384-dim)
- Install: `pip install -e ".[semantic]"`
- The first run downloads the model; later runs use the local cache.
- A plain `import zmrag` never loads torch / transformers — the semantic code
  stays behind the optional extra.
- Run the real-model profile: `python benchmark/run_semantic_profile.py`

## Cross-Encoder Reranking (optional)

True two-stage retrieval:

```
Query
 ↓
First-stage retrieval   (recall a candidate pool)
 ↓
CrossEncoder scoring    (per (query, chunk) pair)
 ↓
Top-K
```

```python
from zmrag.semantic_rerank import CrossEncoderReranker  # not in the top-level public API

reranker = CrossEncoderReranker(
    model_source="cross-encoder/ms-marco-MiniLM-L-6-v2"
)
```

The reranker **only re-scores the candidate pool it is given** — it never
recalls documents or queries a store/index itself. The benchmark profile runs
33 candidates → 8 results; that is a benchmark setting, not a fixed API limit.

Run the real-model profile: `python benchmark/run_semantic_rerank_profile.py`

## Evaluation & Benchmarks

Offline, deterministic retrieval evaluation over prepared
`(query, expected chunk ids)` cases, macro-averaged over queries:

- Hit Rate / Recall / Precision
- MRR / MAP / NDCG, with optional `@K` cutoffs

The frozen benchmark corpus is **local, synthetic, and deliberately small** —
6 documents, 33 chunks, 8 evaluation cases (one has no ground truth, capping
observable recall at 7/8 = 0.875). It exists to verify that the retrieval /
ranking / evaluation pipeline is correct and reproducible. It is **not** BEIR,
**not** MS MARCO, **not** a large production dataset, and **not** a real-world
or SOTA comparison.

Phase 11 / Phase 12 MRR results (from the committed evidence JSON, same fixed
corpus):

| Retrieval config      | Semantic only | Semantic + CrossEncoder rerank |
| --------------------- | ------------- | ------------------------------ |
| Dense                 | 0.708         | 0.813                          |
| Hybrid + weighted     | 0.813         | 0.813                          |
| Hybrid + RRF          | 0.729         | 0.813                          |

These are observations on this one fixed corpus — not a claim that semantic
embeddings or CrossEncoder reranking are universally superior to BM25 / dense /
hybrid.

Evidence files: `benchmark/results/retrieval_small_*.json`,
`benchmark/results/retrieval_semantic_profile.json`,
`benchmark/results/retrieval_semantic_rerank_profile.json`.

## Tests

```bash
python -m pytest -q
```

Current result: **879 passed, 3 skipped**.

## Project Scope

ZMRAG focuses on **retrieval, ranking, and evaluation**. Not in scope:

- LLM answer generation
- Agents
- Tool calling
- MCP
- Memory
- Runtime / serving
- Chat UI

## Documentation

Design intent and per-stage contracts: [docs/architecture.md](docs/architecture.md)

## License

Proprietary.
