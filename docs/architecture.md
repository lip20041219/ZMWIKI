# ZMRAG Architecture

Design intent and implemented-stage documentation for the ZMRAG knowledge /
retrieval pipeline -- phases 1-7, through Pipeline Orchestration.

## Mission

ZMRAG is an engineering-grade **Knowledge / Retrieval Layer**: a reusable,
testable, model-agnostic RAG foundation. It is an independent project, not an
agent, and is intentionally decoupled from ZMAI and ZMForge at this stage.

## Responsibilities

The data-to-context pipeline ZMRAG owns:

| Stage                | Concern                                              |
| -------------------- | ---------------------------------------------------- |
| Ingestion            | Pull data in from external sources                   |
| Document normalization | Canonical document representation                   |
| Chunking             | Split documents into retrievable units               |
| Embedding            | Map chunks to vectors via an embedder                |
| Indexing             | Store chunk/embedding pairs                          |
| Retrieval            | Query the index for candidate chunks                 |
| Reranking            | Reorder candidates by relevance                      |
| Context construction | Assemble winning chunks into prompt context          |
| Evaluation           | Measure retrieval / context quality                  |

## Explicitly NOT responsible

- Agent reasoning / task planning
- Tool execution / agent lifecycle / autonomous action
- UI

Those belong to ZMAI (reasoning/planning) and ZMForge (execution/tools). ZMRAG
exposes retrieval capabilities and consumes nothing from them.

## Core design principles

- **Model-agnostic** — no baked-in LLM; callers supply models where inference is needed.
- **Vector-store-agnostic** — indexes behind narrow interfaces, swappable backends.
- **Embedding-provider-agnostic** — embedders behind interfaces, not concrete clients.
- **Retriever-agnostic** — retrieval strategy is pluggable (dense, sparse, hybrid).
- **Deterministic where possible** — non-inference stages are pure functions.
- **Testable** — every stage independently testable with no network / no models.
- **Composable** — stages are small, typed, and chainable.

## Data contract: Document

`Document` is the base data object of the ZMRAG pipeline:

```
External Input -> Document -> (later) Chunk
```

Shipped in `src/zmrag/schema.py` and exposed at the package root. It is an
immutable value object holding `id`, `content`, `metadata`, an optional typed
`Source`, and an open `content_type`. It owns **no logic**: no chunking,
embedding, indexing, retrieval, ranking, LLM generation, or agent reasoning.
Later stages read from it; they never mutate it.

- `id` is caller-supplied (never database-generated) so it is stable and
  explicit in tests; the ingestor (see Ingestion) requires the caller to
  provide it.
- `metadata` is a free-form `dict[str, Any]` — no store/provider/format
  binding — copied on construction to prevent aliasing.
- `Source` is an abstract contract (a `kind` discriminator) for provenance.
  Concrete source types (file / URL / database / API / generated) are
  *not* implemented yet.

**Frozen rationale**: Documents are shared value objects flowing through
pipeline stages; immutability makes accidental mutation impossible and gives
clean structural equality. The `metadata` dict is only *shallowly* immutable
(nested values stay mutable) — a deliberate trade-off, since deep immutability
would break ordinary user-supplied dicts.

## Data contract: Chunk

A `Chunk` is the smallest retrievable unit a future Chunker cuts a Document
into. Shipped alongside `Document` in `src/zmrag/schema.py`; exported at the
package root. It is an immutable value object holding `id` (its own chunk id),
`document_id` (its parent), `text`, a 0-based integer `position`, and its own
`metadata`.

**Relationship to Document** — one Document yields many Chunks (1 -> N). A
Chunk traces back to its parent through `document_id` only. It deliberately
does **not** hold the Document instance, so the data layer stays low-coupling:
Chunks and Documents remain independent values and no stage can reach through
a Chunk to mutate or depend on the whole Document.

**Position** — a simple, deterministic 0-based integer index locating the
Chunk within its Document. It preserves document order (sorting by position
recovers it) and gives a natural seed for deterministic chunk ids. Character
offsets are consciously deferred; they can be added later only if a real stage
needs them.

**Chunk metadata** — its own free-form dict, structurally identical to
Document metadata. A future Chunker is expected to *inherit* parent Document
metadata into it, but that inheritance is not implemented here and no Document
object is copied into a Chunk.

**Immutability** — same rationale as Document: Chunks are shared value
objects across pipeline stages (indexing, retrieval, context construction);
frozen makes accidental mutation impossible and equality structural.

## Ingestion

```
Raw Text
   ↓
TextIngestor
   ↓
Document
```

`TextIngestor` (abstract: `ingest(document_id, text, *, metadata=None) ->
Document`) is the pipeline's front door: it converts raw external text into a
normalized `Document`, and nothing more. It:

- preserves the text exactly — never strips, normalizes whitespace,
  lowercases, tokenizes, or truncates (ingestion brings data *in*; it does not
  silently rewrite it);
- requires an explicit, deterministic `document_id` from the caller — it never
  generates an id (no UUID / random / timestamp / db-generated id);
- does not chunk, embed, index, retrieve, rerank, or call an LLM.

**Current implementation** — `PlainTextIngestor` in `src/zmrag/ingestion.py`,
exported at the package root. It builds `Document(id, content=text,
content_type="text/plain", source=None)`, optionally forwarding caller
`metadata`. All validation is delegated to `Document`'s own constructor
(non-empty str id/content, whitespace-only rejected, dict metadata, copy-on-
construction) so ingestion maintains no parallel rule set. `content_type` is
always `"text/plain"` and `source` is always `None` here.

**Current scope** — Phase 6A implements `text -> Document`; Phase 6B adds a
minimal local `.txt` file path; Phase 6C adds single-directory `.txt` batch
ingestion (see File Ingestion / Directory Ingestion below). Recursive /
URL ingestion and dedicated parsing (PDF / Markdown / HTML) remain future
design, not implemented, and no URLSource scaffolding exists yet.

## File Ingestion

```
TXT File
   ↓
PlainTextFileIngestor
   ↓
Document
```

`FileIngestor` (abstract: `ingest(path, *, document_id, metadata=None) ->
Document`) is the file front door of the pipeline, built on the same
deterministic principles as Text Ingestion.

**Current implementation (Phase 6B)** — `PlainTextFileIngestor` in
`src/zmrag/file_ingestion.py`, exported at the package root. It is strictly
minimal:

- **`.txt` only** — the extension is matched case-insensitively
  (`.txt` / `.TXT` / `.Txt`); any other extension is refused with `ValueError`.
  Nothing is parsed and no PDF / Word / HTML / Markdown / CSV parsing library
  is pulled in.
- **UTF-8 only** — decoded with `read_text(encoding="utf-8")`; a decode
  failure raises `UnicodeDecodeError` and is never silently substituted or
  fallen back.
- **Verbatim content** — the text is preserved exactly, equal to
  `path.read_text(...)`: no strip, no lower, no normalization, no truncation.
  File ingestion *reads*; it does not *clean*.
- **Explicit deterministic id** — `document_id` is caller-supplied, never
  auto-generated (no UUID / random / timestamp).
- **Provenance** — `FileSource(path)` (below) is placed on `Document.source`.
  No metadata is auto-injected (no filename / timestamp / hash / size); file
  provenance lives in `source`, not in `metadata`.
- **Validation reuse** — non-str/empty `document_id` and non-dict `metadata`
  are delegated to `Document`'s own constructor, exactly as Text Ingestion
  does; no parallel rules. Missing file -> `FileNotFoundError`, a directory ->
  `IsADirectoryError`.

**`FileSource`** (also in `file_ingestion.py`) is a minimal concrete `Source`:
a frozen value holding only the `path` the Document came from, with
`kind == "file"`. Pure provenance — it never reads, parses, or ingests, and
adds no database or third-party dependency. It is the first concrete `Source`;
the schema's `Source` abstraction was not modified to add it.

**Future direction (NOT implemented)** — recursive directory ingestion (see
Directory Ingestion), PDF / Markdown / HTML parsing, URL and database loading.
These exist only as design intent above; no scaffolding for them ships.

## Directory / Batch Ingestion

```
Directory
   ↓
TXT discovery (current dir only, case-insensitive .txt)
   ↓
deterministic ordering
   ↓
PlainTextFileIngestor
   ↓
Document[]
```

`DirectoryIngestor` (abstract: `ingest(directory, *, document_id_prefix="",
metadata=None) -> list[Document]`) ingests every `.txt` file in one directory
as Documents, in one deterministic batch.

**Current implementation (Phase 6C)** — `PlainTextDirectoryIngestor` in
`src/zmrag/directory_ingestion.py`, exported at the package root. Strict
separation of duties:

- `DirectoryIngestor` = **discovery + ordering + batch delegation** only;
- `PlainTextFileIngestor` = **file reading + Document construction** (injected
  via the constructor, `PlainTextFileIngestor` by default). The directory
  ingestor never reads bytes and never builds a Document, so 6B logic is not
  duplicated.

Properties:

- **Single level only** — the current directory is scanned, never subdirectories
  (no `rglob`, no recursion).
- **`.txt` only, case-insensitive** — `.txt` / `.TXT` / `.Txt` are ingested;
  every other file (`.md`, `.pdf`, ...) and subdirectories are *ignored*, never
  errored on and never opened.
- **Deterministic ordering** — sorted by `(name.casefold(), name)`, independent
  of filesystem iteration or creation order; repeated ingestion yields the same
  order and structurally equal Documents.
- **Deterministic ids** — `document_id = "<prefix>/<filename>"` (or just
  `"<filename>"` with an empty prefix). Derived from the *relative filename
  only* -- never the absolute path -- so ids survive the directory being moved.
  No UUID / random / timestamp / auto-increment; ids never depend on discovery
  order.
- **Provenance** — each Document's `FileSource` points at its actual file; the
  shared `metadata` dict is forwarded to each Document (which copies it), and
  nothing is auto-added to metadata.
- **Errors propagate** — a missing directory raises `FileNotFoundError`, a file
  passed as `directory` raises `NotADirectoryError`, an invalid type raises
  `TypeError`, an empty (or `.txt`-free) directory returns `[]`. A per-file
  failure (e.g. `UnicodeDecodeError`, an empty/whitespace file rejected by
  `Document`) is never swallowed or skipped -- it propagates.
- No chunking, embedding, indexing, or retrieval happens here.

**Future design (NOT implemented)** — recursive directory ingestion, batch
configuration, parallel ingestion, incremental ingestion, file change
detection. These are design intent only.

## Chunking

```
Document -> Chunker -> Chunk[]
```

`Chunker` (abstract contract: `chunk(document) -> list[Chunk]`) turns one
Document into ordered, immutable Chunks. A Chunker never mutates its input,
never embeds/indexes/retrieves, never calls an LLM, and is deterministic.

**Current implementation** — `FixedSizeChunker(chunk_size, overlap=0)` in
`src/zmrag/chunkers.py`. Purely lexical: text is sliced into windows of at
most `chunk_size`, never stripped or rewritten; chunking is "split, don't
clean".

- **chunk_size** — maximum window size, in **characters (Python code points)**, *not*
  tokens. ZMRAG has no tokenizer abstraction yet, so no token-aware counting.
- **overlap** — max characters shared between adjacent windows; windows begin
  every `chunk_size - overlap` characters; the final window may be shorter.
  With `overlap == 0`, concatenating chunk texts reproduces the input exactly.
- **position** — 0-based integer index, emitted in order.
- **chunk id** — deterministic `"<document_id>::<position>"`; no UUID, no DB.

**Current limitations** — no token-aware chunking, no semantic chunking, no
sentence-aware chunking, no Markdown-aware chunking. These are *not*
implemented; only deterministic fixed-size splitting exists today.

## Embedding

```
Text -> EmbeddingProvider -> Embedding
```

**Provider abstraction** — `EmbeddingProvider` (abstract: `embed(text)`,
`embed_many(texts)`, `dimension`) is the only thing core code depends on. No
concrete provider is coupled in, so ZMRAG stays agnostic of OpenAI / DeepSeek /
Gemini / local models / any third-party embedding library. `embed_many` equals
`[embed(t) for t in texts]` in content and order by default.

`Embedding` is a frozen value object: a fixed-dimension vector of finite
floats (stdlib types only, no numpy). VectorStores later depend on this value,
never on a provider library's return object. Embeddings stay **out** of the
Chunk schema — a Chunk stays lightweight and maps to its embedding separately.

**Current implementation** — only `DeterministicEmbedder(dimension)`, a
**test-only / local-development** stand-in. It derives vectors purely from the
text via sha256 + struct: no `random`, no `hash()`, so results are
byte-for-byte stable across calls, instances, and processes. It is *not* a
semantic model and must not be used to benchmark retrieval quality.

**Current limitations** — no real API provider, no local model provider, no
GPU embedding, no semantic-quality evaluation.

## Vector Store

```
Chunk + Embedding -> VectorStore -> similarity search -> SearchResult[]
```

`VectorStore` (abstract: `upsert`, `get`, `delete`, `count`, `search`) stores
chunk+embedding pairs and searches them by cosine similarity. It is decoupled
from any embedding provider, database, or vector library, and it **does not**
embed queries (``search`` takes an ``Embedding``, never text), chunk, rerank,
build context, or call an LLM. `SearchResult` carries just the stored `Chunk`
and its `score`. `SearchResult.score` is a *retrieval-relevance* score whose
range and meaning are backend-specific -- cosine in [-1, 1] for dense search,
unbounded BM25 for lexical search; never normalized across backends.
Embeddings remain separate from the Chunk schema; the store keys records by
`chunk.id`.

**Current implementation** — `InMemoryVectorStore`: stdlib dict, brute-force
per search, deterministic ordering by `(-score, chunk_id)`.

**Similarity** — `cosine_similarity(a, b)`: `(a·b) / (||a|| ||b||)`, pure
stdlib. Dimension must match; a zero vector scores 0.0 (no direction).

**Dimension** — the first upsert establishes the store dimension; any later
upsert or query with a different dimension is rejected (never truncated,
padded, or projected).

**top_k** — int >= 1 (bool excluded); more than `count` returns everything.
Empty store + `search` returns `[]`, never raises.

**Current limitations** — in-memory only: no persistence, no distributed
index, no ANN, no external vector DB.

## Retrieval

```
Query -> Retriever -> EmbeddingProvider -> VectorStore -> SearchResult[]
```

Layered responsibilities:
- **EmbeddingProvider** — text -> embedding.
- **VectorStore** — embedding -> similarity search.
- **Retriever** — orchestration only: embeds the query and searches the store.

`Retriever` (abstract: `retrieve(query, top_k=10) -> list[SearchResult]`) never
computes similarity, sorts, reranks, mutates results, or calls an LLM — those
belong to other layers. It returns the store's `SearchResult` objects
unchanged.

**Current implementation** — `DenseRetriever(embedder, store)`: both
collaborators are dependency-injected (constructor, never built internally, no
keys/db reads), so it is agnostic of the concrete provider/store. Its exact
pipeline is `embedder.embed(query) -> store.search(embedding, top_k)`. Query is
validated (non-empty str); top_k and dimension consistency are the store's
contract and propagate unchanged.

**Current limitations** — only dense retrieval: no lexical retrieval, no BM25,
no hybrid retrieval, no reranking, no query expansion, no semantic filtering,
no LLM-based retrieval.

## Lexical Retrieval

```
Chunk -> Tokenizer -> LexicalIndex -> BM25 -> LexicalRetriever -> SearchResult[]
```

An independent lexical strategy, fully separate from dense retrieval:

- **Dense retrieval** = EmbeddingProvider + VectorStore (cosine).
- **Lexical retrieval** = Tokenizer + LexicalIndex + BM25.

`LexicalIndex` (abstract: `upsert`/`get`/`delete`/`count`/`search`) indexes
`Chunk.text`; it never touches embeddings. `InMemoryLexicalIndex` implements
standard BM25 (k1=1.2, b=0.75 configurable, Robertson-smoothed IDF, TF
normalization and length normalization); it scores only from `chunk.text`,
never from metadata. `LexicalRetriever` is a thin orchestrator over an injected
`LexicalIndex`. `SearchResult.score` here is non-negative BM25 -- deliberately
*not* normalized into a cosine-like range, which matters for future hybrid
fusion.

**Tokenizer** — `SimpleTokenizer`, stdlib-only and deterministic: ASCII
letter/digit runs become lowercased tokens; other Unicode alphanumeric code
points (CJK, kana, ...) become single-character tokens; punctuation and emoji
are dropped. Chinese is handled at character granularity so it stays
retrievable -- explicitly not semantic word segmentation, no dictionary/CRF/
jieba/stemming.

**Consistency** — every mutation rebuilds postings from stored chunks, so
index state never goes stale (correctness over incremental-index performance;
a real inverted index belongs to a later phase).

**Current limitations** — no hybrid retrieval, no BM25/vector score fusion,
no reranking, no query expansion, no context construction.

## Hybrid Retrieval

```
                   ┌── DenseRetriever  ──┐
Query ────────────┤                      ├── Fusion ── Hybrid SearchResult[]
                   └── LexicalRetriever ──┘
```

`HybridRetriever(dense_retriever, lexical_retriever, fusion)` fetches
`candidate_k` candidates from each underlying retriever (each >= the final
`top_k`), then the injected `FusionStrategy` deduplicates by `chunk.id`,
assigns a hybrid score, ranks, and returns the top `top_k`. All collaborators
are dependency-injected. `SearchResult.score` here is a *hybrid relevance
score* -- neither cosine nor BM25 -- with a meaning that depends on the fusion
strategy.

**Why raw scores cannot be added** — dense cosine (~[-1, 1]) and BM25
([0, +inf)) live on different scales. Both fusion strategies below resolve
that explicitly; direct `dense + bm25` (weighted or not) is forbidden.

**Weighted Score Fusion** — query-local min-max normalization of each source
within the current query's candidates (never global/historical/training
statistics): `norm = (score - min)/(max - min)` -> [0, 1]; empty or
max == min -> all 1.0. Final: `dense_weight*norm_dense + lexical_weight*norm_lexical`.
Weights >= 0, not both zero; not required to sum to 1 (ranking is
scale-invariant). Defaults 0.5 / 0.5.

**RRF (Reciprocal Rank Fusion)** — `rrf(d) = sum 1/(k + rank(d))`, rank
1-based within each source; `rrf_k` default 60. Rank-based, so no score
normalization at all; documents hit by both sources rank above single-source
hits.

**Current limitations** — no learned fusion, no score calibration, no
reranking, no cross-encoder, no query expansion, no LTR.

## Reranking

```
                          ┌─ retriever (dense / lexical / hybrid) ─ Recall, wide candidate pool
Query ───────────────────┤
                          └─ Reranker.rerank(query, candidates, top_k) ─ Order, narrow to best
```

Reranking is the post-retrieval stage: retrievers are responsible for
*recall* (generating a wide candidate pool); a reranker is responsible only for
*re-ordering* the candidates it is given. A `Reranker` never queries a
VectorStore/LexicalIndex, never embeds, never fuses, and never recalls new
chunks — its output `chunk.id` set is always a subset of the input candidates.
It also never mutates the candidate `SearchResult`s or `Chunk`s; it emits fresh
`SearchResult`s re-scored for this stage.

Full pipeline (implemented stages only; context construction is future work):

```
                        Indexing                            Recall
Document -> Chunker -> Chunk[] -> embed -> VectorStore ─┐
                               `-> tokenize -> LexicalIndex ─┤
                                                             v
Query ------------------------------------------------> HybridRetriever
                                                             |
                                                             v   (optional ordering)
                                                      Reranker.rerank
                                                             |
                                                             v
                                                    Reranked SearchResult[]
                                                             |
                                                             v   (future)
                                                       Context Builder
```

Recall (VectorStore / LexicalIndex) widens the pool; hybrid fusion combines the
strategies; the reranker narrows the pool to the best `top_k`. The reranker is
*optional* in the chain (it is never required to reach a correct answer from a
single strategy) but becomes the correctness lever once more candidates than
fit context must be selected.

**Reranker abstraction** — `Reranker` (abstract: `rerank(query, candidates,
top_k=10) -> list[SearchResult]`). It re-orders and re-scores `candidates`
for a single `query`; returns at most `top_k`, best first. Duplicate
`chunk.id` across candidates is rejected (ValueError); candidates that are not
`SearchResult`s, a non-list `candidates`, and `top_k < 1` (or bool) are
rejected like every other stage. `SearchResult.score` after reranking is a
*reranker relevance* score that overwrites the incoming retrieval score — its
meaning is stage-specific, exactly as fusion already overwrote cosine/BM25.

**Current implementation** — `DeterministicReranker(tokenizer)`, a
**test-only / local-development** stand-in. It is *not* a semantic model:
scoring is lexical token-overlap relevance — for each unique query token it
adds that token's frequency in the candidate's `chunk.text`. No BM25, no IDF,
no learned model, no network. Purely deterministic: repeated calls and
independent instances agree byte-for-byte, so it can stand in as a swappable
reranking stage while everything downstream is built and tested.

**Future design: CrossEncoderReranker (NOT implemented)** — the intended
production reranker is a cross-encoder: a transformer that scores a
(query, candidate) pair jointly (as opposed to a bi-encoder that embeds each
side separately), giving stronger relevance than first-stage retrieval scores.
Planned shape only — deferred so a real model/API contract can be defined
against actual provider constraints:

- `CrossEncoderReranker(provider)` where `provider` exposes a
  `score_pairs(query, texts) -> list[float]` contract (model-agnostic, matching
  ZMRAG's injection style: the provider, not ZMRAG, holds the model).
- Scored outside ZMRAG's deterministic core; candidates pass through an
  injected scorer so ZMRAG never depends on torch/transformers/FlagEmbedding.
- Chunk metadata could later hold query-independent cached encodings; query
  expansion / learned ranking remain separate, unbuilt stages.

**Current limitations** — no cross-encoder, no learned/production reranker, no
chunk-content-based recall (rerankers only reorder, by contract), no query
expansion, no context construction, no evaluation.

## Context Building

```
Reranked SearchResults
        ↓
   ContextBuilder
        ↓
      Context
        ↓
   Future Generator / LLM   (NOT implemented -- design intent only)
```

`ContextBuilder` is the adapter between retrieval/reranking and a future
Generator: it receives **already-ordered** SearchResults and packs the chunks
that fit a character budget into one structured `Context`. It never retrieves,
never reranks, never re-scores, never summarizes/compresses, never generates a
prompt, and never calls a model. The reranker decides the order; ContextBuilder
only packs.

**`Context`** (in `src/zmrag/context.py`) is a frozen value object with just
two fields: `text: str` (the assembled text handed to the Generator) and
`chunks: tuple[Chunk, ...]` (the exact Chunk objects whose entries appear in
`text`). Provenance is structural: `chunks[i]` corresponds to the `[i+1]`
entry in `text`, so "which chunks produced this context" is always recoverable
without embedding source metadata or a score schema in the text. Scores are
deliberately **not** written into `text`; they belong to the retrieval /
reranking stage, not to the Generator's knowledge text. A Chunk's own
`metadata` (e.g. `filename`/`title`) travels with the Chunk object untouched;
ContextBuilder assumes no particular metadata keys exist.

**Packing (`DeterministicContextBuilder`)** — ordered greedy packing, nothing
more. Chunk texts are emitted in input order as numbered entries
(`[1] <text>\n\n[2] <text>...`), each entry's `chunk.text` included verbatim
(never truncated or rewritten). Rules:

- **Order** — the input order is preserved exactly; ContextBuilder never sorts
  by score / id / length. Reranker output order *is* the context order.
- **max_chars** — a budget on the **final `Context.text` length in Python
  characters / code points**, numbering and separators (`\n\n`) included. It is
  *not* an LLM token budget; ZMRAG has no token-counting abstraction (a future
  `TokenBudget` may be layered on, but is not implemented here). `max_chars`
  is a required, validated int >= 1 (bool/float/str/None/0 rejected).
- **Greedy stop** — chunks are added in order until the next one would push
  `text` over `max_chars`, then packing stops; later candidates are never
  reconsidered.
- **First oversized chunk** — if the very first chunk alone exceeds
  `max_chars`, it is still included, untouched: a chunk is an upstream-defined
  semantic/text unit and ContextBuilder must not silently break it. Once that
  happens the text is already over budget, so later chunks cannot be added.
- **Duplicates** — duplicate `chunk.id` is rejected (`ValueError`); upstream
  Hybrid/Reranker is expected to have deduplicated, and ContextBuilder does not
  silently paper over that bug.
- **Empty results** — returns `Context(text="", chunks=())`, no error.

Input `SearchResult`s / `Chunk`s are never mutated; output `Context.chunks`
references the same Chunk objects (no copies).

**Current limitations** — no prompt generation, no system prompts, no token
budget, no summarization / compression / semantic filtering, no LLM / API
calls, no Generator. End-to-end pipeline implemented through Context:

```
Raw Text -> TextIngestor -> Document -> Chunker -> Chunk[]
        --(dense / lexical)--> HybridRetriever
        -> Reranker -> ContextBuilder -> Context -> (Future Generator -> LLM)
```

The last two hops (Generator, LLM) are future design intent only and are not
claimed as implemented anywhere in this repository.

## Pipeline Orchestration

Phase 7 wires every stage above into one dependency-injected, deterministic,
end-to-end pipeline. The pipeline adds **no new algorithm** -- chunking,
embedding, vector similarity, BM25, fusion, reranking, and context packing all
live in the injected collaborators. `DeterministicRAGPipeline` in
`src/zmrag/pipeline.py` (ABC: `RAGPipeline`, exported at the package root)
owns orchestration, dependency wiring, indexing coordination, and query
coordination, and nothing else.

### Indexing

```
Document
   ↓
Chunker
   ↓
Chunk[]
   ├── embed(text) ──> Embedding ──> VectorStore.upsert(chunk, embedding)
   └──────────────────────────────── LexicalIndex.upsert(chunk)
```

`ingest(document)` chunks once, embeds **each chunk text exactly once**, and
upserts the chunk into both the vector store (with its embedding) and the
lexical index (no embedding). Chunks and Documents are never mutated, no
metadata / timestamp / UUID is auto-generated, and no record id is invented --
re-upserting the same deterministic `"<document_id>::<position>"` chunk id
overwrites in both stores, so re-ingesting a Document never grows the index.

### Query

```
Query
   ↓
HybridRetriever.retrieve(query, top_k=candidate_k, candidate_k=candidate_k)  (recall: candidate_k-wide pool)
   ↓  list[SearchResult]
Reranker.rerank(query, candidates, top_k)                                    (order: narrow to best top_k)
   ↓  list[SearchResult]
ContextBuilder.build(reranked, max_chars)                                    (pack: budget-bound)
   ↓
Context
```

`query(query, *, top_k=5, candidate_k=20, max_chars=2000) -> Context`.
Retrieval is asked for a `candidate_k`-wide pool; the reranker narrows it to
the best `top_k`; ContextBuilder packs whatever fits `max_chars`. Every stage
returns "at most k", so `top_k > candidate_k` simply cannot be filled. All
scores, ordering, and selection come from the injected stages; the pipeline
never sorts, filters, re-scores, or re-packs results itself. The return value
is the existing `Context` (text + chunks); no `Answer`/`RAGResponse` wrapper is
invented.

### Dependency injection

All seven collaborators are **required keyword arguments**, validated against
their ABCs at construction and never constructed internally: `chunker`
(`Chunker`), `embedder` (`EmbeddingProvider`), `vector_store` (`VectorStore`),
`lexical_index` (`LexicalIndex`), `retriever` (`Retriever`, hybrid-shaped
`retrieve(query, top_k, candidate_k)` -- `HybridRetriever` in practice),
`reranker` (`Reranker`), and `context_builder` (`ContextBuilder`). The wiring
contract is that the retriever searches the same `vector_store` /
`lexical_index` that `ingest` writes to; the caller owns that wiring.

### Responsibilities

Pipeline is responsible for: orchestration, dependency wiring, indexing
coordination (chunk -> embed -> both stores), query coordination (retrieve ->
rerank -> build).

Pipeline is NOT responsible for (all delegated to injected stages): chunking
algorithm, embedding algorithm, vector similarity, BM25, fusion algorithm,
reranking algorithm, tokenization, context packing, LLM generation.

### Determinism and atomicity

The pipeline itself uses no `random` / `uuid` / `timestamp` / `hash()`; it is
deterministic whenever its injected stages are (all shipped implementations
are), so the same Documents + configuration + query yield structurally equal
Contexts across calls and across independently constructed equivalent
pipelines.

**Phase 7's indexing operation is NOT transactional.** If embedding/upserting
chunk *N* fails, chunks *0..N-1* already sit in the stores and the exception
propagates unchanged; nothing is silently skipped or rolled back, and the
pipeline never reports a partial success as complete. No rollback is
implemented because the in-memory stores expose no safe rollback primitive and
the task scope does not warrant inventing one.

**Current scope** -- single-Document `ingest` and single `query`; no batch /
recursive / PDF / Markdown / URL ingestion, no convenience ingestion beyond
`pipeline.ingest(Document)` (Directory / File / Text ingestion already produce
Documents, which the caller hands over).

**Current limitations** -- no answer generation (LLM / prompt / generation /
agent / query expansion / streaming), no persistence / external vector DB. The
pipeline ends at `Context`, exactly as Context Building does; Evaluation (below)
is a separate, offline consumer.

## Evaluation

```
Pipeline
   ↓  (retrieval / reranking / context output)
Predictions
   ↓
Evaluator
   ↓
EvaluationResult
```

Evaluation (Phase 8A, in `src/zmrag/evaluation.py`, exported at the package
root) is an **offline, deterministic retrieval-metric contract**. A pipeline /
retriever / reranker runs first and produces results; the evaluator only
computes metrics over those already-obtained results. Evaluation does **not**
participate in ingestion, chunking, embedding, retrieval, fusion, reranking,
context building, or generation, and it never drives the pipeline.

**Data contract** -- `EvaluationCase(query, relevant_chunk_ids)` describes one
sample: a non-empty `query` and its ground-truth relevant `chunk.id` set
(canonical sorted tuple; empty or duplicated ids rejected). `EvaluationPrediction
(query, results)` carries the ordered `SearchResult` list a stage returned for
one query (duplicate `chunk.id` rejected -- metrics must sit on an unambiguous
candidate set, never a silently-deduplicated one). Cases and predictions are
matched **by position** (`cases[i]` <-> `predictions[i]`), never by dict/set/
hash order. `EvaluationResult(hit_rate, recall, precision, total_cases,
evaluated_cases, mrr, map, ndcg)` aggregates the run (the three ranking
fields default to `0.0` and are Phase 8C additions; the original five can
still be passed positionally).

**Phase 8A metrics** -- the minimal reliable retrieval set, nothing more:

- **Hit Rate** -- fraction of queries with at least one retrieved relevant chunk.
- **Recall** -- macro average of |relevant ∩ retrieved| / |relevant|.
- **Precision** -- macro average of |relevant ∩ retrieved| / |retrieved|;
  precision over an empty retrieved set is defined as `0.0`.

Phase 8C adds **MRR / MAP / NDCG** (see [Ranking Metrics](#ranking-metrics)
below); Hit Rate / Recall / Precision keep exactly their Phase 8A meaning and
are never turned into `@K` variants.

**Statistical semantics** -- all aggregation is a **macro average over
queries**: each query is scored (its own recall / precision / hit / RR / AP /
NDCG) and the query scores are averaged. Micro pooling (merging all
retrieved/relevant sets and computing one ratio) is deliberately not used.
Example: recalls `1.0`, `0.5`, `0.0` over three queries average to `0.5`.

**Top-K semantics** -- without a cutoff, `K` is the depth of the predictions
supplied: whatever the retrieval stage actually returned *is* the candidate
set. The evaluator never re-retrieves and never truncates by a second
evaluator-side `K`; the reported recall / precision correspond to that
returned depth.

**Identity** -- only `chunk.id` is compared. `chunk.text`, Chunk object
identity, metadata, and `SearchResult.score` are never used (scores are not
recomputed, and the evaluator holds no scorer). All computation is
deterministic: no `random` / `uuid` / `timestamp` / `hash()`; the same
`(cases, predictions)` always yields an equal `EvaluationResult`. Ground truth
must be non-empty (recall needs a denominator); empty is rejected, never
silently scored as a fake number.

**Coupling** -- Evaluation imports only the `SearchResult` value type (the
retrieval-output contract it must validate) plus `schema`. It imports no
`pipeline`, retriever, store, embedder, chunker, reranker, or context module,
so any stage that emits `SearchResult`s -- dense, lexical, hybrid, reranker,
or the full pipeline -- can be scored later with no reverse coupling.

**Current limitations** -- no F1 / BLEU / ROUGE / BERTScore / LLM-as-a-Judge,
no generation-quality evaluation, no micro pooling, no cross-query pooling, no
learned/calibrated metrics, no benchmark datasets, no external evaluation
framework. The E2E tests prove only that pipeline output feeds the evaluator;
they never assert (or fake) benchmark quality.

## Evaluation Harness

Phase 8B (in `src/zmrag/evaluation_harness.py`, exported at the package root)
adds an orchestration shell around an `Evaluator` -- execution, validation,
and reporting -- while keeping the metric math exactly where Phase 8A put it.

```
Pipeline                    EvaluationHarness                Evaluator
   ↓ (retrieve/rerank/         ↓ run(cases, predictions)        ↓ evaluate(...)
   build already done)         validates:                        computes metrics
   SearchResult[]s             • both lists, equal length         (injected, never
   ↓                           • per-element types               constructed here)
Build cases + predictions      • position-wise query match
   ↓                           ↓
EvaluationCase[] /         EvaluationResult
EvaluationPrediction[]          ├── .to_dict()   machine-readable (8 stable keys)
                                └── .summary()   human-readable (fixed format)
```

The harness is the **execution** layer only:

- **`EvaluationHarness.__init__(evaluator)`** -- the `Evaluator` is constructor-
  injected and validated; the harness never constructs an evaluator internally,
  so callers may substitute their own metric implementation.
- **`run(cases, predictions) -> EvaluationResult`** -- validates that both are
  lists, that they match **one-to-one by position** (a length mismatch raises
  `ValueError`; inputs are never truncated, padded, ignored, or auto-retrieved),
  that every element is the right type (`TypeError`), and that each
  prediction's `query` equals its case's `query` (a mismatch raises
  `ValueError`; queries are never auto-corrected). It then calls
  `evaluator.evaluate(cases, predictions)` **exactly once** and returns that
  `EvaluationResult` unchanged. Validation happens before delegation, so a bad
  run never reaches the evaluator.

The harness computes no metrics itself, reads no files or database, never
retrieves, and never imports or drives the pipeline. Inputs are never mutated
(no sort / pop / clear / append / field writes) and no state is cached between
calls. An empty run `([], [])` is delegated to the evaluator unchanged, so its
empty-run contract (zero aggregates, no fake NaN/None/0.0) is preserved.

**Reporting lives on the result value, not the harness** (added in Phase 8B to
`EvaluationResult` in `evaluation.py`, extended by Phase 8C):

- `to_dict()` -- a plain dict with the eight stable keys `hit_rate`, `recall`,
  `precision`, `mrr`, `map`, `ndcg`, `total_cases`, `evaluated_cases`;
  ints/floats only (JSON-friendly), fixed key order, no timestamps, no paths,
  no object reprs.
- `summary()` -- a fixed-format human string (`## Evaluation Result` +
  `Queries: N` + each rate at six decimals). Both are deterministic across
  calls and instances.

**Layer separation (unchanged contract):** `Pipeline` produces retrieval
output; `Evaluator` computes metrics; `EvaluationHarness` executes a run,
validates inputs, and reports. `evaluation_harness.py` imports only
`evaluation` types, never a pipeline / retriever / store / embedder / chunker /
reranker / context module.

**Implemented (Phase 8B)** -- injected `Evaluator` with exactly-once
delegation; positional length + element-type + query-match validation;
immutability of all inputs; deterministic re-runs (equal result / dict /
summary across calls and evaluator instances); `EvaluationResult.to_dict()` /
`summary()`; empty-run delegation; end-to-end tests proving
`DeterministicRAGPipeline` context output feeds the harness (one hit, one
miss, macro metrics correct).

**Not implemented (Phase 8B)** -- no F1 / BLEU / ROUGE / BERTScore /
LLM-as-a-Judge; no generation-quality evaluation; no benchmark datasets; no
external evaluation framework; no `save_json` / file output / CLI / Web UI; no
persistent result storage; no auto-retrieval (the harness only consumes
already-produced predictions). The harness itself gained nothing in Phase 8C --
ranking metrics live on the `Evaluator` / `EvaluationResult`, which the harness
already delegates to unchanged.

## Ranking Metrics

Phase 8C adds the classic **offline ranking metrics** to the Phase 8A / 8B
evaluation contract (still in `src/zmrag/evaluation.py`, on
`RetrievalEvaluator` / `EvaluationResult`). Evaluation *consumes ranked
results*; it never produces a ranking.

```
Ranked SearchResult[] per query            (authoritative order from a stage)
        ↓  RetrievalEvaluator.evaluate(cases, predictions, *, k=None)
EvaluationResult(..., mrr, map, ndcg)      (macro-averaged over queries)
```

**The prediction's order is authoritative.** `EvaluationPrediction.results`
already *is* the ranked retrieval output (dense / lexical / hybrid / reranker /
pipeline). MRR / MAP / NDCG read that order verbatim and never re-sort by
`SearchResult.score`, `chunk.id`, or anything else -- evaluation judges a
given ranking, it does not re-rank.

**Relevance is binary and id-only:** a result is relevant iff its `chunk.id`
is in `relevant_chunk_ids`. No score, text, embedding, or metadata is ever
inspected.

Definitions (standard IR, per query):

- **MRR (Mean Reciprocal Rank)** -- `1 / rank` of the first relevant result
  (rank starts at 1); `0` when nothing is relevant. Macro-averaged.
- **MAP (Mean Average Precision)** -- `AP = Σ_k Precision@k × rel(k) /
  |relevant|`. The denominator is the **ground-truth relevant count**, so an
  unretrieved relevant id still dilutes AP; `rel(k)` marks whether the `k`-th
  result is relevant. Macro-averaged.
- **NDCG@k** -- binary `DCG = Σ (2^rel - 1) / log2(rank + 1)` (a relevant id
  at rank `r` contributes `1 / log2(r + 1)`); `IDCG` is DCG of the ideal
  ranking (all relevant ids first). `NDCG = DCG / IDCG`, `0.0` when IDCG is
  zero (an empty candidate list).

**@K cutoff** -- `RetrievalEvaluator.evaluate(cases, predictions, *, k=None)`
adds an optional keyword-only cutoff (backward compatible: the old two-argument
call is unchanged). When `k` is an int >= 1 (bool/float/str/0/negative
rejected), MRR / MAP / NDCG become **MRR@K / MAP@K / NDCG@K** over the first
`k` results of each prediction; `k=None` uses the full prediction and performs
no implicit truncation. Hit Rate / Recall / Precision always use the full
prediction regardless of `k` -- Phase 8A semantics are untouched and were not
renamed to `@K` variants.

**Evaluation does NOT** rerank, retrieve, score semantic relevance, inspect
embeddings or metadata, call an LLM, read files, or touch a store / pipeline.
Inputs are never mutated (cutoffs use a fresh prefix slice). Everything is
deterministic: no `random` / `uuid` / `time` / `hash()`, and `EvaluationResult`
now carries `mrr` / `map` / `ndcg` (three new defaulted fields appended after
the original five), which `to_dict()` and `summary()` report alongside the
Phase 8A rates.

**Implemented (Phase 8C)** -- MRR / MAP / NDCG and their `@K` variants via the
keyword-only `k`; full per-query math, macro aggregation, cutoff and `k`
validation, immutability, determinism, extended serialization, and an E2E test
proving real pipeline output yields exact ranking values (`mrr == 1/rank`,
`map == mrr` for single-relevant cases, `ndcg == 1/log2(rank + 1)`).

**Not implemented (Phase 8C)** -- no F1 / BLEU / ROUGE / BERTScore /
LLM-as-a-Judge; no graded / multi-level relevance (binary only); no
personalized or learned ranking metrics; no benchmark datasets; no external
evaluation framework. The public surface is unchanged -- no new classes, no new
exports; only `RetrievalEvaluator` gains the optional `k`, and
`EvaluationResult` gains three defaulted fields.

## Retrieval Evaluation Integration

Phase 9A closes the loop between a real pipeline and the Evaluation Harness:
`RetrievalEvaluationRunner` in `src/zmrag/evaluation_runner.py` (exported at the
package root) runs a list of `EvaluationCase`s through an injected
`RAGPipeline`, converts each query's real retrieval output into an
`EvaluationPrediction`, and hands the prepared pair to an injected
`EvaluationHarness`. It is the thin offline glue the earlier Evaluation layers
deliberately refused to provide: the Harness never retrieves, and Evaluation
never imports the pipeline, so *something* has to drive a real pipeline -- that
is this runner, and only this runner.

```
EvaluationCase[]  (query + relevant_chunk_ids)
   ↓  per case, exactly once
RAGPipeline.retrieve(query, *, top_k, candidate_k)      (the REAL retrieval path)
   ↓  list[SearchResult]  (order + scores verbatim)
EvaluationPrediction(query, results)
   ↓
EvaluationHarness.run(cases, predictions)  →  EvaluationResult   (unchanged)
```

**`retrieve()` is the shared retrieval path.** To feed real `SearchResult`s
without fabricating evidence, Phase 9A adds the evaluation-facing entry point
`RAGPipeline.retrieve(query, *, top_k, candidate_k) -> list[SearchResult]` to
the ABC. It is exactly the retrieve-then-rerank stage pair `query()` uses
(hybrid retriever asked for a `candidate_k`-wide pool, then reranked to
`top_k`) and returns the stages' own ranked results -- real scores, real order,
never position/hash/metadata guesses. `DeterministicRAGPipeline.query()` is
precisely `retrieve()` + `ContextBuilder.build()`, so there is exactly **one**
retrieval implementation: querying never diverges from what evaluation sees,
and no second retriever / reranker is ever created.

**Runner responsibilities (orchestration only).** Constructor-inject
`pipeline: RAGPipeline` and `harness: EvaluationHarness` (both type-checked;
nothing is constructed internally). `run(cases, *, top_k=5, candidate_k=20)`
validates `cases` is a list of `EvaluationCase` and both cutoffs are ints >= 1
(`candidate_k < top_k` is a `ValueError`) -- all *before* any retrieval -- then,
for each case in order, calls `pipeline.retrieve(case.query, top_k=...,
candidate_k=...)` **exactly once**, wraps the returned list in an
`EvaluationPrediction` with the query copied verbatim (no lower / strip /
normalize), and finally calls `harness.run(cases, predictions)` **exactly once**,
returning its `EvaluationResult` unchanged. The runner never computes a metric,
never sorts or truncates, never re-retrieves, never builds an `EvaluationResult`
of its own, and never calls `query()` or `ingest()`. Empty `cases` still flow
through `harness.run([], [])` so the evaluator's empty-run contract is
preserved.

**Dependency rules.** `evaluation_runner.py` imports only the evaluation,
harness, and pipeline modules; it depends on no retriever / reranker / store /
embedder / chunker / context implementation. The runner is deterministic
whenever the injected pipeline and evaluator are, and it mutates nothing --
cases, predictions, and the pipeline's returned `SearchResult`s are passed
through untouched.

**Not implemented (Phase 9A)** -- no metric computing or result assembly in the
runner, no query rewriting / expansion, no report files, no benchmark / dataset
drivers, no parallelism, and no persistent state between runs. The public
surface grows by exactly one class, `RetrievalEvaluationRunner`.

## Evaluation Case Set

Phase 9B (in `src/zmrag/evaluation_dataset.py`, exported at the package root)
adds the **case organization contract** of offline retrieval evaluation:
`EvaluationCaseSet` is an ordered, immutable collection of `EvaluationCase`s
that exists so a run's inputs are organized and reusable. It sits in front of
the existing evaluation chain and changes nothing downstream of it:

```
EvaluationCaseSet
   ↓  case_set.cases (tuple, input order preserved)
EvaluationCase[]  →  RetrievalEvaluationRunner
   ↓  per case: RAGPipeline.retrieve(query, top_k, candidate_k)
EvaluationPrediction[]
   ↓
EvaluationHarness.run(cases, predictions)
   ↓  RetrievalEvaluator
EvaluationResult
```

**`EvaluationCaseSet(cases)`** is a frozen dataclass holding a single
`cases: tuple[EvaluationCase, ...]` field -- the same immutable-tuple contract
`Context.chunks` uses, so a caller's mutable `list` is refused with `TypeError`
(convert with `tuple(cases)` when needed) and the stored sequence can never be
mutated through the object. A CaseSet **organizes** cases and nothing else:

- **only organizes** -- holds `cases` and provides `len()` / iteration /
  indexing; it never calls a pipeline, retriever, harness, or evaluator, never
  loads data, never retrieves, and never computes a metric.
- **keeps case order** -- the input order is preserved exactly; it is never
  sorted by query / chunk id / relevance count / hash / score, because
  `EvaluationCase[i]` aligns to `EvaluationPrediction[i]` by position.
- **immutable + deterministic** -- frozen, no ids / timestamps / hashes /
  filesystem / database / network; equivalent instances behave identically.
- **duplicates preserved** -- it is an ordered sequence, not a math set: no
  `set()` / `hash()` / deduplication is applied, and an empty tuple is a valid
  empty evaluation dataset (`len == 0`, `cases == ()`).
- **delegates case validation** -- query and `relevant_chunk_ids` rules stay
  where `EvaluationCase` already owns them (including its canonical sorted
  relevant-id tuple); CaseSet only asserts a well-formed, ordered sequence of
  `EvaluationCase`.
- **serializes minimally** -- `to_dict()` returns
  `{"cases": [{"query": ..., "relevant_chunk_ids": [...]}, ...]}` in exact
  case order (relevant ids reproduced from the case's canonical tuple, never
  re-sorted); JSON-friendly, fixed keys, deterministic across calls and
  equivalent instances, no reprs / addresses / timestamps / paths / hashes.

Phase 9B introduces **no benchmark framework and no benchmark dataset** -- no
BEIR / MS MARCO / Natural Questions / HotpotQA, no dataset loaders or
downloaders, no HuggingFace `datasets` integration, no dataset persistence, no
`dataset_id` / timestamp / version fields. `evaluation_dataset.py` (named for
future growth) implements only `EvaluationCaseSet`; no `Dataset` /
`DatasetLoader` / `Benchmark` abstraction is added. The public surface grows by
exactly one class; `EvaluationHarness` and `RetrievalEvaluationRunner` keep
their Phase 8B / 9A signatures and responsibilities unchanged (the runner still
takes a `list[EvaluationCase]`; a CaseSet feeds it via `list(case_set.cases)`).

## Evaluation Dataset Loader

Phase 9C (in `src/zmrag/evaluation_loader.py`, exported at the package root)
adds the **input adapter** of offline retrieval evaluation: a local,
JSON-compatible evaluation-case file is loaded into an `EvaluationCaseSet`, so
a reusable case set lives on disk and re-enters the Phase 9A / 9B chain
unchanged:

```
JSON file
   ↓  JsonEvaluationDatasetLoader.load(path)   (UTF-8, re-read each call)
EvaluationCaseSet
   ↓  list(case_set.cases)
RetrievalEvaluationRunner  →  EvaluationHarness  →  RetrievalEvaluator
   ↓
EvaluationResult
```

**`EvaluationDatasetLoader`** is an abstract contract with a single method
`load(path: str | Path) -> EvaluationCaseSet`. **`JsonEvaluationDatasetLoader`**
is the only shipped implementation: constructor-free (no `__init__` arguments),
stateless, and uncached -- every `load` re-reads the file, so two loads (same
loader or different instances) over the same content return equal CaseSets.

**Strict, minimal JSON schema.** Top level must be a JSON object carrying
exactly the `"cases"` key; each case must be an object carrying exactly
`"query"` and `"relevant_chunk_ids"` (an array of strings). Unknown fields at
either level are rejected, never silently ignored. The loader validates
**structure** only, then delegates to the value layer:

```
JSON raw fields → EvaluationCase(query, relevant_chunk_ids) → EvaluationCaseSet(cases=tuple(...))
```

`EvaluationCase` therefore keeps owning every business rule -- non-empty /
non-whitespace query (`TypeError`), non-empty relevant ids, no blank or
duplicate id, canonical sorted relevant-id tuple (`ValueError`) -- with no
second copy of that logic in the loader, and no second ordering rule on top of
it. Schema violations raise `ValueError`; `EvaluationCase` constraints
propagate untouched (never caught and rewrapped).

**Ordering / duplicates.** The JSON array order is preserved exactly into
`case_set.cases` -- never sorted, deduplicated, or hash-ordered. Duplicate
cases are legal and preserved. An empty `{"cases": []}` loads to
`EvaluationCaseSet(cases=())` without error.

**Strict file handling.** `path` must be a `str` or `pathlib.Path` (anything
else -> `TypeError`). A directory -> `IsADirectoryError`, a missing file ->
`FileNotFoundError`, a non-UTF-8 file -> `UnicodeDecodeError` (explicit UTF-8
read), malformed JSON -> `json.JSONDecodeError`. No exception is swallowed and
no empty CaseSet is ever fabricated for a bad file.

**Implemented (Phase 9C)** -- local JSON loading, strict schema validation,
deterministic ordering, `EvaluationCase` delegation, immutable
`EvaluationCaseSet` output, and an E2E test proving JSON -> loader -> CaseSet
-> real `DeterministicRAGPipeline` via `RetrievalEvaluationRunner` yields a
correct `EvaluationResult` (Hit Rate / Recall / MRR / MAP / NDCG from the
existing evaluator, never from the loader).

**Not implemented (Phase 9C)** -- BEIR / MS MARCO / Natural Questions /
HotpotQA; HuggingFace `datasets` integration; remote dataset download
(`requests` / `urllib`); dataset registry / versioning / metadata; benchmark
framework; database / sqlite; caching; CLI; web UI; query expansion; any
`Dataset` / `Benchmark` abstraction. `evaluation_loader.py` imports only the
stdlib (`json`, `pathlib`, `abc`) and the evaluation value layer
(`evaluation_dataset` -> `evaluation`); it never imports a pipeline, retriever,
store, embedder, chunker, reranker, or context module. No benchmark scores,
SOTA, or performance claims are made anywhere.

## Benchmark Runner

Phase 9D (in `src/zmrag/benchmark.py`, exported at the package root) adds the
**offline execution layer**: a thin runner that composes the existing stages
into one deterministic entry point, `EvaluationCaseSet -> EvaluationResult`.

```
EvaluationCaseSet
   ↓  BenchmarkRunner.run(cases, *, top_k=5, candidate_k=20)
   ↓    list(cases.cases)  ->  RetrievalEvaluationRunner.run(...)   (exactly once)
   ↓      per case: Pipeline.retrieve(query, top_k, candidate_k) -> EvaluationPrediction[]
   ↓        EvaluationHarness.run(cases, predictions) -> RetrievalEvaluator
EvaluationResult   (returned verbatim -- never rebuilt, never recomputed)
```

**`BenchmarkRunner`** is an abstract contract with a single method
`run(cases: EvaluationCaseSet, *, top_k: int = 5, candidate_k: int = 20) ->
EvaluationResult`. **`DeterministicBenchmarkRunner`** is the only shipped
implementation. It is **offline execution orchestration** and nothing else: it
receives an `EvaluationCaseSet`, validates its arguments *before* delegating,
converts the immutable `cases` tuple into the `list` the Phase 9A runner
already requires (order preserved, contents untouched), calls the injected
`RetrievalEvaluationRunner.run(list(cases.cases), top_k=..., candidate_k=...)`
**exactly once**, and returns that `EvaluationResult` unchanged.

The benchmark runner does **not** implement retrieval, reranking, or any metric;
does **not** load a dataset, save results, touch a database or the network; and
does **not** construct a pipeline, harness, evaluator, or runner. All component
wiring happens in the caller and is folded into the injected evaluation runner:
`DeterministicBenchmarkRunner(*, evaluation_runner: RetrievalEvaluationRunner)`
type-validates the single collaborator and keeps it; no pipeline / harness /
evaluator is ever built internally, and no environment variable, API key, or
file is read. Validation rejects a non-`EvaluationCaseSet` (`TypeError`),
non-int / bool / `None` cutoffs (`TypeError`), cutoffs `< 1` (`ValueError`),
and `candidate_k < top_k` (`ValueError`) -- arguments are never auto-corrected.

**Dataset loading is a separate concern, kept out of the benchmark runner.**
The boundary is exact: `JsonEvaluationDatasetLoader` maps a JSON file to an
`EvaluationCaseSet` *before* the benchmark runner is involved; the benchmark
runner only ever evaluates an already-built `EvaluationCaseSet`. No
`BenchmarkDatasetLoader` / `DatasetRegistry` / dataset version / metadata /
name / id / cache / checksum abstraction is added. An empty `EvaluationCaseSet`
is delegated to the evaluation runner (`list([])`), so the evaluator's empty-run
contract holds; the benchmark runner never fabricates its own `EvaluationResult`.

Deterministic and pure: no `random` / `uuid` / `time` / `hash()`, no hidden
state, no global cache, no mutation of the input CaseSet, cases, or result;
equal inputs over equal injected runners always yield equal results. Tests
prove exactly-once delegation with a recording fake runner, identity passthrough
of a sentinel `EvaluationResult`, input immutability, determinism across calls
and instances, and a real E2E through the full deterministic pipeline (hit +
miss -> Hit Rate / Recall / MRR / MAP / NDCG from the existing evaluator).

**Implemented (Phase 9D)** -- `EvaluationCaseSet -> EvaluationResult`
execution orchestration; constructor-injected `RetrievalEvaluationRunner`;
argument validation before delegation; exactly-once delegation; identity result
passthrough; empty-CaseSet delegation.

**Not implemented (Phase 9D)** -- any metric math (MRR / MAP / NDCG / F1 /
recall@K / precision@K / NDCG@K / BLEU / ROUGE / BERTScore / LLM-as-a-Judge);
retrieval / reranking; dataset loading or persistence; result file saving; CLI;
web UI; database; remote datasets; BEIR / MS MARCO / HuggingFace datasets;
benchmark registry; dataset versioning; caching; parallel / async evaluation;
LLM judge; generation / RAG-answer evaluation; query expansion;
observability / logging. `benchmark.py` imports only `evaluation_dataset`,
`evaluation_runner`, and `evaluation` (plus stdlib `abc`) -- never a retriever,
store, embedder, chunker, reranker, fusion, context, or schema module.

## Benchmark Dataset & Corpus

Phase 10A ships the *data inputs* for deterministic local experiments. They
live under ``benchmark/`` and are consumed by the existing chain unchanged:
``JsonEvaluationDatasetLoader`` reads ``benchmark/datasets/retrieval_small.json``
into an ``EvaluationCaseSet``, and ``BenchmarkRunner`` drives a real pipeline
over the corpus via ``RetrievalEvaluationRunner``.

Layout:

```
benchmark/
├── corpus/            # 6 hand-written synthetic .txt documents (doc_a..doc_f)
└── datasets/
    └── retrieval_small.json   # 8 evaluation cases (7 real + 1 deliberate miss)
```

Document ids are the file stems (``doc_a``, ...) -- explicit and stable, never
a UUID, timestamp, or absolute path. Ground-truth chunk ids in the JSON are
*real* output of the canonical chunker (``FixedSizeChunker(chunk_size=120,
overlap=20)``) over this corpus, captured from actual ingestion, never guessed.
Each document is 5-6 chunks; the corpus is topical (database indexes / cache /
message queue / vector search / hybrid retrieval / text chunking) with partial
lexical overlap and phrase-level ground truth. Queries cover lexical hits, a
two-chunk hybrid case, a different-wording vector case, and one unindexed
``missing::0`` miss.

**Implemented** -- local synthetic retrieval corpus; local JSON evaluation
cases; deterministic corpus; deterministic case -> chunk mapping; real
ingestion/evaluation E2E (FileIngestor -> Document -> Pipeline.ingest ->
retrieval -> EvaluationCaseSet -> RetrievalEvaluationRunner -> EvaluationHarness
-> RetrievalEvaluator -> EvaluationResult).

**Not implemented** -- BEIR; MS MARCO; HuggingFace datasets; remote datasets;
benchmark registry; dataset versioning; dataset metadata framework; automatic
dataset generation; external benchmark framework; performance comparison.
Corpus and dataset are data files only; no new loader, registry, config, or
production abstraction was added.

> This corpus is a small synthetic validation dataset for deterministic local
> experiments, not a claim of real-world benchmark performance.

## Benchmark Execution & Reproducible Baseline

Phase 10B makes the Phase 10A local retrieval benchmark a *reproducible
experiment baseline*: a single canonical entry point re-runs the whole chain
from scratch and writes one machine-readable evidence artifact that any later
run can be checked against.

```
benchmark/corpus (6 .txt)
   → PlainTextFileIngestor
   → Document
   → DeterministicRAGPipeline.ingest        (FixedSizeChunker 120/20 →
     DeterministicEmbedder → InMemoryVectorStore + InMemoryLexicalIndex →
     DenseRetriever / LexicalRetriever / WeightedScoreFusion / HybridRetriever)
   → deterministic retrieval
benchmark/datasets/retrieval_small.json
   → JsonEvaluationDatasetLoader → EvaluationCaseSet
   → DeterministicBenchmarkRunner → RetrievalEvaluationRunner →
     EvaluationHarness → RetrievalEvaluator → EvaluationResult
   → reproducible evidence  (benchmark/results/retrieval_small_baseline.json)
```

`benchmark/run_retrieval_small_baseline.py` is the canonical execution script.
It rebuilds every component on each run (no cache, no singleton, no reading of
a previous result), verifies the canonical boundaries (6 documents, 33 chunks,
``doc_<letter>::<integer>`` ids with no absolute paths, 8 cases) *before* any
evidence is produced, and then runs the real benchmark twice on one pipeline
instance and once on a fully fresh instance. Evidence is written only when
same-instance and cross-instance results agree on every view (``result``,
``to_dict``, ``summary``). All metrics in the evidence are copied verbatim from
``EvaluationResult.to_dict()`` -- nothing is hand-computed. The evidence schema
is fixed: dataset id, corpus counts, ``top_k`` / ``candidate_k`` (33/33), the
six retrieval metrics, the case counts, and ``deterministic``; it carries no
timestamp, path, hostname, username, or environment information, so a rerun can
be compared to it byte-for-byte by tests and by the runner itself.

**Implemented** -- local synthetic corpus; deterministic execution script; real
end-to-end retrieval evaluation; reproducibility checks (same-instance and
cross-instance); machine-readable baseline evidence; committed evidence that
tests re-derive from fresh real runs rather than asserting against itself.

**Not implemented** -- BEIR; MS MARCO; HuggingFace datasets; remote datasets;
external benchmark comparison; SOTA comparison; statistical significance
testing; experiment registry; result database; dashboard; web UI; CLI
framework; parallel execution; distributed benchmark; LLM judge; generation
evaluation. The evidence is a static JSON artifact written by a script; no
benchmark framework, result store, or production abstraction was added, and
``src/zmrag/`` is untouched by this phase.

> This baseline is a deterministic local validation result on a small synthetic
> corpus and is not a claim of real-world benchmark performance.

## Controlled Retrieval Experiments

Phase 10C runs a controlled, offline retrieval experiment matrix over the exact
Phase 10B fixed environment -- same corpus (6 documents / 33 chunks), same
dataset (8 cases), same chunking (``FixedSizeChunker(120, 20)``), same embedding
(``DeterministicEmbedder``), same ``top_k = candidate_k = 33``, and the same
``EvaluationHarness`` -> ``RetrievalEvaluator`` metric contract. Only the
retrieval configuration changes:

- E1 Dense-only (``DenseRetriever``)
- E2 Lexical-only (``LexicalRetriever``)
- E3 Hybrid + Weighted (``HybridRetriever`` + ``WeightedScoreFusion``)
- E4 Hybrid + RRF (``HybridRetriever`` + ``RRFFusion``)
- E5 Hybrid + Weighted + DeterministicReranker
- E6 Hybrid + RRF + DeterministicReranker

The experiment matrix is an offline controlled composition of existing public
retrieval/evaluation components. Each configuration is rebuilt from scratch on
every run, evaluated per case, and scored by the unchanged evaluator; every
metric in the evidence comes verbatim from ``EvaluationResult.to_dict()``.
Results are written only after same-instance and cross-instance runs are
identical (``result`` / ``to_dict`` / ``summary``), so every row is marked
``deterministic: true``. E5 is the Phase 10B baseline configuration and acts as
a recompute guard: it must reproduce the frozen baseline metrics exactly.

The experiments do not modify retrieval algorithms or production pipeline
semantics. ``DeterministicRAGPipeline`` is deliberately not used because its
``retrieve`` always asks the retriever for ``candidate_k`` and always reranks,
which cannot express single-source (Dense-only / Lexical-only) or no-reranker
configurations; the script composes those same public components directly
instead. The frozen Phase 10B baseline remains untouched.

Results are local synthetic-corpus evidence, not claims of real-world benchmark
performance.

``benchmark/run_retrieval_experiments.py`` is the canonical execution script;
``benchmark/results/retrieval_small_experiments.json`` is its committed,
machine-readable evidence artifact (fixed key set and order, int/float metrics
only, no timestamp / hostname / username / absolute path / environment dump).

**Implemented** -- controlled E1-E6 matrix; single-source and hybrid
configurations; Weighted and RRF fusion; DeterministicReranker on/off; real
direct-composition execution over the canonical corpus; per-configuration
same-instance and cross-instance determinism; E5 recompute guard against the
frozen baseline; machine-readable evidence; tests that re-derive every config
from a fresh real run rather than asserting against itself.

**Not implemented** -- BEIR; MS MARCO; HuggingFace datasets; remote datasets;
statistical significance framework; hyperparameter sweep; automated tuning;
parallel execution; async execution; experiment registry; experiment database;
MLflow; Weights & Biases; LLM judge; generation evaluation. The results are a
static JSON artifact written by a script; no experiment framework, result
store, or production abstraction was added, and ``src/zmrag/`` is untouched by
this phase.

> Results are deterministic, reproducible measurements of the existing system on
> a small synthetic corpus -- a measurement, not an optimization, and not a
> claim of real-world benchmark performance.

## Retrieval Truncation Study

Phase 10D measures how ``top_k`` / ``candidate_k`` truncation changes recall and
ranking for each retrieval configuration, over the exact Phase 10C fixed
environment -- same corpus (6 documents / 33 chunks), same dataset (8 cases),
same chunking (``FixedSizeChunker(120, 20)``), same embedding
(``DeterministicEmbedder``), and the same ``EvaluationHarness`` ->
``RetrievalEvaluator`` metric contract. Only the two cutoff knobs change.

### Motivation

Phase 10C ran every configuration at ``top_k = candidate_k = 33``, which is
essentially a full 33-chunk corpus ranking: each configuration already returned
(almost) the whole pool, so hit_rate / recall saturated at 7/8 and truncation
effects were unobservable. Phase 10D narrows the two knobs so candidate
*selection* (whether a ground-truth chunk survives into the fused candidate
list at all) can be told apart from final *ranking depth* (the chunk is
retrieved but cut by ``top_k``). It is a measurement of the existing system
under cutoffs -- not parameter tuning and not a search for "best" production
settings.

### Experimental matrix

18 cells, fixed in advance and never tuned against results. No reranker and no
``DeterministicRAGPipeline``: single-source and no-reranker cells are not
expressible through the pipeline (its ``retrieve`` always asks for a
``candidate_k`` pool and always reranks), so each cell is composed directly from
existing public components, exactly as Phase 10C did.

- ``D1`` Dense-only -- 2 cells (``top_k`` in {4, 8}; ``candidate_k`` null, the
  ``DenseRetriever`` API has no such knob)
- ``D2`` Lexical-only -- 2 cells (``top_k`` in {4, 8}; ``candidate_k`` null)
- ``D3`` Hybrid + Weighted -- 7 cells (``candidate_k`` in {4, 8, 16, 33} x
  ``top_k`` in {4, 8}, with ``top_k <= candidate_k``)
- ``D4`` Hybrid + RRF -- the same 7 cells

For every cell the script rebuilds all components from scratch (no cache, no
singleton, no reading of a prior result), runs the real chain twice on one
instance and once on a fully fresh instance, and refuses to write evidence
unless same-instance and cross-instance runs are identical in every view
(``result`` / ``to_dict`` / ``summary``). Results are used verbatim -- never
padded, never re-sorted, never capped beyond a stage's own "at most k". The
full-pool control (33/33) is *not* regenerated as new rows: Phase 10C E1-E4
already provide it. Instead an E3 cross-phase guard recomputes Hybrid + Weighted
@ 33/33 through this script's own path and requires it to reproduce the frozen
Phase 10C E3 row exactly before any evidence is written.

### ``candidate_k`` / ``top_k`` semantics

- ``candidate_k`` widens each *source* fetch inside ``HybridRetriever`` only
  (dense and lexical are each asked for ``candidate_k``); it is not a hard pool
  size -- the deduplicated fused union can exceed ``candidate_k`` -- and the
  fusion still truncates to the final ``top_k``. Single-source retrievers have no
  ``candidate_k``.
- ``top_k`` is the post-fusion depth actually returned and scored. The evaluator
  applies no ``@K`` of its own (see Evaluation), so ranking metrics are computed
  over the actual returned list and are **not comparable as same-depth metrics
  across different ``top_k``**.
- Precision's denominator is the number retrieved, so precision moves
  mechanically with ``top_k`` and is only comparable within one ``top_k``.

### Observed findings (on this fixed corpus / dataset only)

- **``candidate_k``**: no cell with ``candidate_k >= 8`` lost ground truth to
  candidate selection -- each such configuration reaches the 7/8 ceiling at
  ``top_k = 8``. Every recall dip observed at ``top_k = 4`` (e.g. D3-c16-t4
  hit_rate / recall 0.75; D4-c16-t4 0.625; D4-c33-t4 hit_rate 0.625 / recall
  0.5625) recovered to 0.875 at ``top_k = 8`` with the *same* ``candidate_k``,
  which locates those dips at the final depth (ranking truncation of an
  already-retrieved chunk), not at candidate-pool exclusion. RRF at
  ``candidate_k = 4`` (D4-c4-t4, recall 0.8125) is the one cell whose loss this
  evidence cannot separate between exclusion and ranking truncation.
- **``top_k``**: within these cells ``top_k`` is the dominant recall lever;
  raising it 4 -> 8 restored the ceiling in every cell that had dipped
  (D3-c8 recall .8125 -> .875; D3-c16 .75 -> .875; D4-c8 .75 -> .875; D4-c16
  .625 -> .875; D4-c33 .5625 -> .875). This is an observed property of these
  cells -- it is **not** a recommendation of ``top_k = 8`` as a production
  parameter.

### Weighted vs RRF

At the same ``candidate_k`` and ``top_k``, D3 and D4 receive identical candidate
lists from the same sources (both fusions run over the same dense + lexical
outputs and deduplicate the same union), so any difference between them is
produced by the fusion *ranking layer*, not by the source candidate-retrieval
layer:

- at ``top_k = 8`` recall converges to 0.875 for both fusions at every
  ``candidate_k >= 8``, while MRR / MAP / NDCG still differ at every matched
  cell (e.g. D3-c33-t8 mrr 0.75 vs D4-c33-t8 mrr 0.34791666666666665);
- at ``top_k = 4`` that ranking difference surfaces as a recall difference at the
  depth boundary (e.g. D3-c33-t4 recall 0.875 vs D4-c33-t4 recall 0.5625);
- RRF's ``candidate_k = 4`` cell carries an exclusion-vs-ranking ambiguity this
  evidence cannot resolve (above).

This is an observation on this corpus, not a claim that Weighted is better than
RRF in general.

### Dense-only / Lexical-only limitations

- ``D1`` dense-only is the most truncation-sensitive configuration here:
  ``top_k = 4`` returns zero relevant results across all eight cases (all metrics
  0.0) and ``top_k = 8`` recovers only hit_rate / recall 0.25 (two of the eight
  cases). This is the joint behavior of the current ``DeterministicEmbedder`` --
  a hash-derived deterministic *synthetic* embedding stand-in, not a semantic
  model -- and this corpus / dataset. It must **not** be read as "dense retrieval
  is weak", and it supports no general dense-vs-BM25 conclusion.
- ``D2`` lexical-only reaches 0.875 at both depths; its ranking metrics are
  identical at ``top_k`` 4 and 8 (mrr 0.8125, map 0.8229166666666666, ndcg
  0.8366783004521589) while precision differs mechanically (0.26041666666666663
  @4 vs 0.19166666666666665 @8), consistent with short BM25 lists whose relevant
  ranks sit at the top. The same reverse caution applies: this is not a
  "BM25 beats dense" claim.

### Missing-ground-truth ceiling

The dataset's eighth case (``quantum submarine agriculture``) references
``missing::0``, a chunk id that does not exist in the 33-chunk corpus. Only
seven of the eight cases therefore have ground truth that can actually be
retrieved, so the observed hit_rate / recall ceiling of every configuration is
7/8 = 0.875. 0.875 is not "best recall" in any production sense; it is the
ceiling imposed by this fixed dataset's deliberate missing-ground-truth control.
No configuration ever fabricates the missing id.

### Reproducibility and evidence

``benchmark/run_retrieval_truncation.py`` is the canonical execution script;
``benchmark/results/retrieval_truncation.json`` is its committed,
machine-readable evidence artifact (fixed schema and row order, 18 rows, all
marked ``deterministic: true``, no timestamp / hostname / username / absolute
path / environment dump); ``tests/test_benchmark_truncation.py`` pins the
schema, the 18-cell matrix, per-cell determinism, and the E3 cross-phase guard,
and re-runs the runner to a fresh output path to prove byte-for-byte
reproduction. Phase 10B / Phase 10C evidence files are read but never
overwritten, and ``src/zmrag/`` is untouched by this phase. The full test suite
passes (832 tests).

**Implemented** -- D1-D4 18-cell truncation matrix; per-cell same-instance and
cross-instance determinism; E3 cross-phase recompute guard against the frozen
Phase 10C E3 row; machine-readable evidence; tests that re-derive every
committed row from a fresh real run rather than asserting against itself.

**Not implemented** -- BEIR; MS MARCO; HuggingFace datasets; remote datasets;
statistical significance; hyperparameter sweep; automated tuning of cutoffs,
fusion weights, or ``rrf_k``; learned / semantic reranking under truncation;
scalability / large-corpus behavior; any recommendation of a "best"
``candidate_k`` / ``top_k``; any production dense-vs-BM25 or Weighted-vs-RRF
claim. The evidence is a static JSON artifact written by a script; no benchmark
framework, result store, or production abstraction was added, and ``src/zmrag/``
is untouched by this phase.

> Results are deterministic, reproducible measurements of the existing system
> on a small synthetic corpus under ``top_k`` / ``candidate_k`` cutoffs -- a
> measurement of truncation behavior, not an optimization, and not a claim of
> real-world benchmark performance.

## Guiding constraints (development process)

- Build in small phases; each phase is runnable and tested before the next.
- API/interface first, implementation second.
- No speculative features, no fake benchmarks, no ZMAI/ZMForge coupling yet.
- Low coupling between modules.
- Python stdlib + mature, necessary third-party deps only.

## Planned module layout (later phases)

> Superseded early-phase plan. Every phase shipped as flat modules directly
> under `src/zmrag/` (`schema.py`, `chunkers.py`, …, `benchmark.py`) — the
> sections above reference them by file. No sub-package restructure is planned.

```
src/zmrag/
├── schema/        # canonical document/chunk/result types
├── ingest/        # ingestion + normalization
├── chunk/         # chunkers
├── embed/         # embedder interface + providers
├── index/         # vector-store interface + backends
├── retrieve/      # retrievers (dense/sparse/hybrid)
├── rerank/        # rerankers
├── context/       # context construction
└── evaluate/      # evaluation
```
