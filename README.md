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
- Optional query rewriting (`QueryRewriter`; deterministic normalization baseline)
- Weighted score fusion
- Reciprocal Rank Fusion (RRF)
- Deterministic reranking
- Optional real semantic embeddings (sentence-transformers)
- Optional Cross-Encoder reranking
- Context building
- Optional answer generation (`Generator`; deterministic extractive stand-in)
- Optional real LLM generation (`OpenAICompatibleGenerator`, stdlib HTTP)
- Source attribution (`answer_with_sources()` → answer + cited chunks)
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
   └────────────►  Dense / Lexical / Hybrid Retrieval  ◄── retrieval query
                               ↓                             ▲
                         Fusion (Weighted / RRF)             │
                               ↓                   Optional Query Rewriting
                        Optional Reranking                   ▲
                               ↓                          user Query
                       Context Construction
                               ↓
                       Optional Generation  ◄── the original user query
                               ↓
                        Answer + Sources
                               ↓
                             Evaluation
```

`retrieve()` runs the shared retrieval path and returns a ranked
`list[SearchResult]`; `query()` is that same path followed by context packing;
`answer()` is `query()` followed by generation; `answer_with_sources()` is
`answer()` plus the source attribution for that same Context. One path, four
entry points, so they never diverge. A configured `query_rewriter` runs inside
that one path, so every entry point searches the retrieval query while the
generator still receives the original one.

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
    DeterministicGenerator,
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
    generator=DeterministicGenerator(),         # optional; deterministic extractive stand-in
)

pipeline.ingest(doc)

# query -> ranked SearchResult[]
results = pipeline.retrieve(query="dense retrieval", top_k=3, candidate_k=33)
for r in results:
    print(r.chunk.id, round(r.score, 4), r.chunk.text[:40])

# query -> ... -> Context -> Generator -> answer
print(pipeline.answer(query="dense retrieval", top_k=3, candidate_k=33))

# answer + the chunks it was grounded in
result = pipeline.answer_with_sources(query="dense retrieval", top_k=3, candidate_k=33)
print(result.answer)
for source in result.sources:
    print(source.document_id, source.chunk_id, source.position, source.text[:40])
```

ZMRAG is a **library**, not a standalone command-line product. You compose the
stages above through the Python API (or run the benchmark scripts in
[Evaluation & Benchmarks](#evaluation--benchmarks)).

## Query Transformation

```
user Query ──► QueryRewriter (optional) ──► retrieval query ──► Retriever
     │
     └──────────────────────────────────────────────────────► Generator
                                                    (the original query)
```

`QueryRewriter.rewrite(query) -> str` sits in front of retrieval and does one
thing: turn the question the user asked into the query the retriever searches
with. It never retrieves, reranks, packs context, or generates.

**The original query is never overwritten.** The rewritten query is the
*retrieval* query -- it reaches the retriever and the reranker. The Generator is
handed the query the user actually asked, character for character: rewriting
decides how a question is looked up, never what was asked. Citations are
unaffected either way, because they project the Context retrieval really
produced.

Rewriting is optional and off by default. Omit `query_rewriter` and the query
reaches the retriever verbatim.

```python
from zmrag import DeterministicQueryRewriter

pipeline = DeterministicRAGPipeline(
    ...,
    query_rewriter=DeterministicQueryRewriter(),
)
```

The bundled `DeterministicQueryRewriter` is a **normalization baseline, not a
semantic rewriter**: it trims the query and collapses runs of whitespace, and
that is the whole thing. No synonyms, no expansion, no model, no network --
case, punctuation, and word order stay exactly as typed, so it cannot
reinterpret what you asked. Do not expect it to improve retrieval; it exists so
the stage has a deterministic implementation, and so `query_rewriter=None` has
an honest baseline to be compared against.

Real query rewriting -- an LLM rewriter, HyDE, multi-query expansion -- belongs
behind this same contract, carrying its own optional dependency, the way
`semantic.py` carries `sentence-transformers`. None of it is part of the core
runtime.

A configured rewriter's contract is enforced, not repaired: a rewriter that
raises, returns a non-`str`, or returns an empty string fails the call rather
than silently falling back to the original query.

Because the rewriter is a pipeline collaborator, an evaluation run is
"baseline" or "query-rewritten" purely by which pipeline you inject into
`RetrievalEvaluationRunner` -- no metric, harness, or dataset change. That
comparison was run on the frozen benchmark: see
[Query Rewriter Benchmark](docs/query_rewriter_benchmark.md)
(`python benchmark/run_query_rewriter_profile.py`) -- it reports identical
metrics for both variants, and explains why normalization cannot move them.

## Retrieval

- **Lexical** — BM25 over a lexical index; term-overlap based, no model.
- **Dense** — embeddings plus cosine search in a vector store.
- **Hybrid** — combines dense and lexical signals through a fusion strategy
  (weighted score fusion or Reciprocal Rank Fusion) into one ranked list.

A reranker narrows a candidate pool further. The default `DeterministicReranker`
is a deterministic lexical-overlap stand-in for tests and local development —
**not** a semantic reranker. The stage is optional: omit `reranker` (or pass
`reranker=None`) and the pipeline returns the retriever's own ranking cut to
`top_k`. `candidate_k` (the recall pool handed to the reranker) and `top_k`
(final result count) are configured independently either way.

## Semantic Retrieval (optional)

Swap the deterministic embedder for real sentence-transformers embeddings:

```python
from zmrag.semantic import SemanticEmbeddingProvider  # not in the top-level public API

embedder = SemanticEmbeddingProvider(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
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
    model_name="cross-encoder/ms-marco-MiniLM-L-6-v2"
)
```

The reranker **only re-scores the candidate pool it is given** — it never
recalls documents or queries a store/index itself. The benchmark profile runs
33 candidates → 8 results; that is a benchmark setting, not a fixed API limit.

Run the real-model profile: `python benchmark/run_semantic_rerank_profile.py`

## Generation (optional)

```
Query → Retrieval → Rerank → Context → Generator → Answer
```

`Generator.generate(query, context) -> str` is the last stage and the only one
that produces user-facing text. It never retrieves, reranks, re-packs, or
mutates the `Context` it is given. Generation is optional: with `generator=None`
(the default) the pipeline stops at `query()`'s `Context`, and
`pipeline.answer(...)` says so rather than returning context where an answer was
expected.

The bundled `DeterministicGenerator` is **extractive, not generative**: it
returns the context chunk with the highest query-token overlap, verbatim, and
`""` for an empty context. No model, no network, no dependency — for tests,
quick start, and CI stability. A real model belongs behind the same `Generator`
contract: `build_prompt(query, context)` is the minimal, dependency-free prompt
helper (delimited context and question sections) that
[`OpenAICompatibleGenerator`](#llm-generation-optional) reuses, and no LLM SDK
is part of the core runtime.

## LLM Generation (optional)

`Generator` is pluggable, and two implementations ship:

| Generator | What it is |
| --- | --- |
| `DeterministicGenerator` | Extractive stand-in. No model, no network, no dependency. |
| `OpenAICompatibleGenerator` | A real LLM call over an OpenAI-compatible chat completions endpoint. |

The pipeline stays provider-agnostic: it hands the query and the `Context` to
whatever `Generator` you inject and knows nothing about URLs, headers, JSON, or
vendors.

```python
from zmrag import OpenAICompatibleGenerator

generator = OpenAICompatibleGenerator(
    base_url="https://api.example.com/v1",  # the API root, not the full endpoint
    model="your-model-name",
    # api_key omitted -> ZMRAG_API_KEY from the environment is used, if set
    temperature=0.0,
    max_tokens=512,   # optional; left out of the request when None
    timeout=30.0,     # seconds; one attempt, no retry loop
)

pipeline = DeterministicRAGPipeline(..., generator=generator)
result = pipeline.answer_with_sources("...", top_k=5)
```

Any server exposing the same `POST {base_url}/chat/completions` shape works --
OpenAI, DeepSeek, Groq, Together, vLLM, llama.cpp, Ollama, LM Studio, LiteLLM,
and similar. No vendor SDK is involved: the adapter speaks HTTP through the
standard library (`urllib.request` + `json`), so `dependencies` stays empty and
nothing pins ZMRAG to one provider.

**API keys** come from the `api_key` argument or from `ZMRAG_API_KEY` when the
argument is omitted (a blank environment value counts as unset). With no key at
all, no `Authorization` header is sent -- local servers legitimately need none.
No vendor-specific variable is read, and no key is written to a file, logged, or
rendered by a `repr`: the generator reports only *whether* a key is set, and
`HttpRequest.__repr__` masks credential headers.

**Citations are not generated by the model.** The adapter returns a plain
answer string; sources remain a projection of the `Context` (see
[Source Attribution](#source-attribution)), so a model cannot fabricate, drop,
or rename a source -- even if it writes a chunk id into its prose.

**Failures are raised, never smoothed over**: a non-2xx status, a provider
`error` payload, invalid JSON, or a missing/empty `choices`/`message`/`content`
raise `GeneratorResponseError`; an unreachable host or a timeout raises
`GeneratorTransportError`. There is no fallback answer and no retry loop.

**Testing never touches a network.** The HTTP boundary is an injectable
transport -- `Transport = Callable[[HttpRequest], HttpResponse]` -- so the whole
adapter is driven with a fake in the test suite: no socket, no key, no cost.

```python
from zmrag import HttpResponse

def transport(request):                      # HttpRequest -> HttpResponse
    return HttpResponse(status=200, body=b'{"choices":[{"message":{"content":"hi"}}]}')

OpenAICompatibleGenerator(base_url="...", model="...", transport=transport)
```

```bash
export ZMRAG_API_KEY=...    # your own key; never commit one to this repository
```

## Source Attribution

```
Answer + Sources
```

RAG answers can expose where they came from: `answer_with_sources(...)` returns
an `AnswerResult` — the generator's answer plus a `Citation` per chunk that
actually reached the Context, in Context order (best-ranked first), each
carrying `document_id` (source identity), `chunk_id` / `position` (chunk
identity), the chunk's `metadata`, and its text verbatim.

```python
result: AnswerResult = pipeline.answer_with_sources(query="hybrid retrieval", top_k=2)
result.answer          # -> str, the generator's own output, unmodified
result.sources         # -> tuple[Citation, ...], empty when nothing was retrieved
```

Sources are **derived from the retrieved/context chunks, never generated**: the
Generator is not asked for citations and its output is never parsed for them.
`context.chunks` *is* the packing record, so a citation cannot name a chunk that
did not participate in the answer's context. Empty retrieval → `answer == ""`
and `sources == ()`; nothing is invented to justify an answer. `answer(...)`
itself is unchanged and still returns a plain `str`.

## Retrieval Evaluation

Retrieval is evaluated offline and reproducibly: **Dense**, **BM25 / lexical**,
**Hybrid** (weighted score fusion or RRF) and an optional **reranking** stage,
scored by the library's own metrics -- Hit Rate, Recall, Precision, MRR, MAP,
NDCG, each with optional `@K` cutoffs -- over a frozen local benchmark (6
documents / 33 chunks / 8 queries).

The strategies are compared head to head in a controlled experiment that holds
the corpus, chunking, embedding, BM25 tokenizer, stores, queries, ground truth,
cutoffs and evaluator identical, and varies only the strategy:

| Strategy (`top_k=5`) | Recall@3 | Recall@5 | MRR@5 | NDCG@5 |
| --- | ---: | ---: | ---: | ---: |
| Dense | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| Hybrid + Weighted | 0.8125 | 0.8750 | 0.7500 | 0.7852 |
| Hybrid + RRF | 0.5000 | 0.7500 | 0.3271 | 0.4399 |
| Hybrid + Weighted + Rerank | 0.8750 | 0.8750 | 0.7500 | 0.7905 |
| Hybrid + RRF + Rerank | 0.8750 | 0.8750 | 0.7500 | 0.7905 |

Read them with their caveat: the bundled `DeterministicEmbedder` is a hash
stand-in, not a model, so the Dense row measures plumbing rather than dense
retrieval. The full analysis -- per-query changes, the candidate-pool sweep, and
what these numbers do and do not support -- is in the
[Retrieval Strategy Benchmark](docs/retrieval_strategy_benchmark.md).

Reproducible evidence: `benchmark/results/retrieval_strategy_comparison.json`,
regenerated by `python benchmark/run_retrieval_strategy_profile.py`.

## Answer Evaluation

ZMRAG evaluates both halves of a RAG answer:

| Layer | Question |
| --- | --- |
| **Retrieval Evaluation** | Did we retrieve the right context? |
| **Answer Evaluation** | Did the generated answer cover the expected answer, and stay grounded in the context it was given? |

`AnswerEvaluator` scores one already-produced answer -- it never retrieves,
reranks, generates, or reads a store -- on three deterministic lexical
baselines:

| Metric | Definition |
| --- | --- |
| **Lexical reference coverage** | share of the reference answer's unique tokens that appear in the answer |
| **Lexical groundedness** | share of the answer's sentences whose tokens are mostly found in the context (threshold 0.5) |
| **Source-level citation coverage** | the same sentence measure, against the cited sources' text instead of the whole context |

> These are deterministic evaluation baselines, not semantic LLM-judge metrics.

They share one blind spot worth stating plainly: token overlap cannot see
meaning. On the bundled profile a sentence and its negation score identically --
"LRU evicts the least recently used entry" and "LRU never evicts the least
recently used entry" both reach coverage 1.00 and groundedness 1.00, because
every token is present either way. That is the ceiling of a lexical baseline,
and the profile records it rather than hiding it.

Measured over the 12-case `answer_small` dataset with the deterministic
generator (6 documents / 33 chunks, `top_k=5`, `candidate_k=33`):

| Aggregate | Value |
| --- | ---: |
| mean reference coverage | 0.6533 |
| mean groundedness | 0.9167 |
| mean citation coverage | 0.8333 |
| answer rate | 1.0000 |
| source presence rate | 0.9167 |
| mean source count | 4.5833 |

The dataset deliberately includes negative cases, and they are the point: one
case is fully grounded yet does not answer the question (coverage 0.22,
groundedness 1.00), and another matches its reference exactly while the
retrieved context does not support it (coverage 1.00, groundedness 0.00).
Retrieval quality, answer correctness, and groundedness are three different
things.

Evidence: `benchmark/results/answer_evaluation_profile.json`, reproduced by
`python benchmark/run_answer_evaluation.py` -- offline, no key, no model.

## Claim-Level Attribution

[Answer Evaluation](#answer-evaluation) reports **source-level** citation
coverage: are the answer's sentences covered by the cited material at all?
`attribution.py` asks a narrower question, one claim at a time: is *this* claim
supported by the context, and *which* cited sources carry it?

| Layer | Question |
| --- | --- |
| Source-level citation coverage | Is the answer covered by the cited sources? |
| Claim-level attribution | Is this claim supported, and by which citations? |

They are not substitutes for each other, and neither is collapsed into a score.

```python
from zmrag import DeterministicClaimAttributor, DeterministicClaimExtractor

claims = DeterministicClaimExtractor().extract(answer)   # one sentence = one claim
result = DeterministicClaimAttributor().attribute(
    claims, context=context, citations=answer_result.sources
)
result.supported_claim_rate    # claims the context covers
result.attributed_claim_rate   # claims a citation covers -- a different number
```

`supported` and `attributed` are deliberately separate: a claim can be supported
by the retrieved context while carrying no citation at all, and a claim can cite
a source that does not cover it. The cell between them is counted, not hidden.

> This is a deterministic lexical baseline, not a semantic factuality judge.

What it is: sentence splitting for claims, token overlap for support, structured
`Citation` objects for attribution. What it is not: no LLM judge, no entailment
or NLI, no embeddings, no claim decomposition beyond sentence boundaries, and no
parsing of the answer for citation markers -- a `Citation` goes in, a chunk id
comes out, and no marker convention exists anywhere in ZMRAG. The consequences
are measured rather than papered over: a negated claim ("LRU **never** evicts
the least recently used entry") is fully supported *and* attributed to the very
source it contradicts, and a correct paraphrase scores as unsupported.

Measured over the same 16-case dataset
(`python benchmark/run_attribution_profile.py`): 25 claims, supported claim rate
**0.9600**, attributed claim rate **0.8400**, mean support score 0.9427 -- with
3 claims supported by the context but left uncited. The bundled generator is
extractive and quotes its context, which is why support runs high here; a
generative model would produce different numbers.

ZMRAG now covers: ingestion → chunking → embedding → dense / lexical / hybrid
retrieval → fusion → reranking → query transformation → context construction →
answer generation → source attribution → retrieval evaluation → answer
evaluation → claim-level attribution.

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
`benchmark/results/retrieval_semantic_rerank_profile.json`,
`benchmark/results/retrieval_query_rewriter_*.json`.

The query rewriter was measured against that frozen benchmark in a controlled
baseline-vs-rewritten run: [Query Rewriter Benchmark](docs/query_rewriter_benchmark.md).

## Tests

```bash
python -m pytest -q
```

Current result: **1281 passed, 5 skipped**.

## Project Scope

ZMRAG focuses on **retrieval, ranking, evaluation**, and the thin stages around
them (query transformation, generation, attribution). Not in scope:

- LLM answer generation as a *product*: generation is a pluggable adapter (see
  [LLM Generation](#llm-generation-optional)) — no prompt framework, no
  streaming, no retries/backoff, no agent loop
- Semantic query rewriting (HyDE, multi-query, expansion)
- Agents
- Tool calling
- MCP
- Memory
- Runtime / serving
- Chat UI
- Faithfulness / RAGAS-style evaluation

## Documentation

Design intent and per-stage contracts: [docs/architecture.md](docs/architecture.md)

## License

Proprietary.
