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

## Query Transformation

```
User Query
   ↓
QueryRewriter.rewrite(query)          (optional stage: skipped when none is injected)
   ↓  retrieval query
Retriever (dense / lexical / hybrid)

User Query ──────────────────────────► Generator (the original, never the retrieval query)
```

`QueryRewriter` (abstract: `rewrite(query: str) -> str`, in
`src/zmrag/rewriters.py`) maps the query the user is asking with into the query
the retrieval path searches with. One method, one direction, no state carried
between queries: it never retrieves, embeds, fuses, reranks, packs context,
generates, or attributes, and there is no route from a rewriter to a store,
index, or model.

**The original query is never overwritten.** Three responsibilities, three
queries:

| Stage | Sees |
| --- | --- |
| QueryRewriter | the caller's original query (its own input) |
| Retriever, Reranker | the *retrieval* query (the rewriter's output) |
| Generator, Citations | the caller's original query, verbatim |

The reranker scores against the retrieval query because the candidate pool was
recalled with it -- the whole retrieval path works on one query. The Generator
is handed the original because rewriting decides *how a question is looked up*,
never *what was asked*: a rewriter that reworded the question could otherwise
change the subject of the answer. Citations are unaffected either way, since
they project the Context retrieval actually produced.

Rewriting is **optional and off by default** (`query_rewriter: QueryRewriter |
None = None`); with `None` the query reaches the retriever verbatim and the
rewriter is never called. The stage runs inside `retrieve()`, after argument
validation, exactly once per call -- `query()`, `answer()`, and
`answer_with_sources()` all reach it by way of `retrieve()`, so one public call
is one rewrite.

**Implemented** -- `DeterministicQueryRewriter`: trim, plus collapsing each run
of internal whitespace to a single space (`str.split` splits on all Unicode
whitespace). Case, punctuation, accents, CJK, and word order are left exactly as
typed. Stateless, therefore identical across calls, instances, and processes.

**Not implemented** -- anything that changes the information need: no synonyms,
no query expansion, no HyDE, no multi-query, no LLM rewriting, no model of any
kind. The deterministic rewriter is a *normalization baseline*, not a semantic
rewriter, and it makes no claim to improve retrieval -- on some queries it
cannot. Semantic rewriting belongs behind this same contract, carrying its own
optional dependency the way `semantic.py` carries sentence-transformers.

**Why normalization stops before the tokenizer** -- `SimpleTokenizer`
lowercases, drops punctuation, and splits CJK into single characters, so
tokenize-then-rejoin would *replace* the query rather than normalize it. The
retrieval query is also what the dense embedder sees, while the lexical side
tokenizes for itself further down.

**Failure posture** -- a configured rewriter's contract is enforced, never
repaired: a rewrite that raises, returns a non-`str`, or returns an empty /
whitespace-only string fails the call. The pipeline never silently falls back to
the original query, which would hide the bug behind plausible-looking results.

**Evaluation** -- because the rewriter is a pipeline collaborator, a
baseline-vs-query-rewritten comparison is just two pipelines injected into
`RetrievalEvaluationRunner`; no metric, harness, runner, or dataset changes.

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
   Generator (optional stage)
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

**Current limitations** — no summarization / compression / semantic filtering,
no token budget. End-to-end pipeline implemented through Context:

```
Raw Text -> TextIngestor -> Document -> Chunker -> Chunk[]
        --(dense / lexical)--> HybridRetriever
        -> Reranker -> ContextBuilder -> Context -> (Generator)
```

## Generation

```
Query + Context
        ↓
   Generator.generate(query, context) -> str
        ↓
      Answer
```

`Generator` (abstract: `generate(query, context) -> str`) is the last stage and
the only one that produces user-facing text. Its inputs are the user's query
and the packed `Context`; its output is the answer string. It never chunks,
embeds, indexes, retrieves, fuses, reranks, packs, evaluates, or mutates the
Context it is given — answering is the whole responsibility. `build_prompt(
query, context)` is the minimal dependency-free prompt helper (explicitly
delimited context and question sections, no template engine, no prompt
registry); a real adapter reuses or replaces it.

**Two implementations ship, one contract** — the Generator is the pluggable
boundary of this stage:

```
                         ┌─ DeterministicGenerator        (extractive, no model)
Context ─────────────────┤
                         └─ OpenAICompatibleGenerator     (real LLM over HTTP)
                                    │
                                    ▼
                            OpenAI-compatible provider
                            POST {base_url}/chat/completions
```

`DeterministicGenerator(tokenizer)` is the **test-only / local-development**
stand-in. It is *not* a language model: generation is **extractive** — it
returns the `chunk.text` of the context chunk with the highest count of
distinct query tokens (the same interpretable lexical signal
`DeterministicReranker` orders by, used to select rather than to sort), ties
keeping the earliest/best-ranked chunk. No model, no network, no randomness, no
temperature/max-tokens knobs, so it never pretends to be an LLM; being
copy-only it cannot hallucinate. `Context(text="", chunks=())` yields `""` — no
evidence, no fabricated answer.

`OpenAICompatibleGenerator` (in `src/zmrag/llm_generators.py`) is the real
adapter: it builds one request from the query and the Context (a fixed system
instruction plus `build_prompt(query, context)` as the user message), sends it
through an injected transport, and returns `choices[0].message.content`.

**Provider-specific HTTP logic stays inside the Generator adapter. The Pipeline
remains provider-agnostic** — it hands over a query and a `Context`, receives a
`str`, and never sees a URL, a header, a JSON body, or a vendor name. One
adapter reaches every server exposing the OpenAI-compatible chat completions
shape (OpenAI, DeepSeek, Groq, Together, vLLM, llama.cpp, Ollama, LM Studio,
LiteLLM, ...); no vendor SDK and no HTTP client library is involved — the
transport is `urllib.request` + `json`, so the core stays stdlib-only.

**The HTTP boundary is injectable** — `Transport = Callable[[HttpRequest],
HttpResponse]`, defaulting to the stdlib `urllib_transport`. That is the only
thing tests need to replace: the whole adapter (request shape, headers, auth,
parsing, failure modes) is exercised with no socket, no key, no cost.

**Failure posture** — enforced, never repaired. A non-2xx status, a provider
`error` payload, invalid JSON, missing/empty `choices`, a missing `message`, or
a non-string `content` all raise `GeneratorResponseError`; an unreachable host,
DNS/TLS failure, or timeout raises `GeneratorTransportError` (with the
underlying `urllib`/socket exception chained). There is no fallback answer, no
default string, and no retry loop — `GeneratorError` is the shared base.

**Credentials** — the key comes from the `api_key` argument or `ZMRAG_API_KEY`
(one generic variable; no vendor-specific variable is read). It is resolved
once at construction, sent only as a bearer header, and never rendered:
`OpenAICompatibleGenerator.__repr__` reports only whether a key is set, and
`HttpRequest.__repr__` masks credential headers, because a repr reaches logs
and tracebacks.

**Optional stage** — `generator=None` (the default) disables generation: the
pipeline still ingests, retrieves, and packs Context exactly as before, and
`answer()` reports that generation is unavailable instead of returning the raw
Context as if it were an answer.

**Not implemented here, and not claimed** — streaming, retries/backoff,
provider-specific parameters, function calling, multi-provider routing, token
counting, multi-turn memory, agents, and citation generation (see Source
Attribution: citations are projected from the Context, never produced by the
model). Real *semantic* query rewriting is likewise still absent — the shipped
rewriter normalizes whitespace only.

## Source Attribution

```
Query → Retrieval → Rerank → Context → Generation → Answer + Sources
```

```
Context.chunks (the packing record)
        ↓
   citations_from_context
        ↓
   tuple[Citation, ...]   →   AnswerResult(answer, sources)
```

**Sources are derived from retrieved/context chunks, not generated or
hallucinated by the Generator.** The Generator's contract is unchanged —
`generate(query, context) -> str` — and it is never asked to report its own
sources: a model can invent a citation, a packing record cannot. Attribution
is a *projection* of `Context.chunks`, which are the exact `Chunk` objects whose
`[n]` entries make up `Context.text` (see Context Building). No assembled
context text is parsed to guess where a passage came from, no chunk is
re-retrieved, and a chunk that never reached the Context cannot be cited.

**This holds for the real LLM adapter too.** `OpenAICompatibleGenerator`
returns a plain answer string and is never asked for a citations payload; the
model's prose is never parsed for source ids, so a model that writes
"according to `doc_a::0`" produces exactly that text and nothing else — the
citations still come from the Context, unaffected. Keeping the answer a `str`
is what makes attribution deterministic even when generation is not.

`Citation` (frozen dataclass, exported) is flat by design — it copies both
identity levels out of the chunk rather than nesting or referencing it:
`document_id` (source-level identity), `chunk_id` + `position` (chunk-level
identity), `metadata` (shallow-copied, like every other metadata in the core),
and `text` (the chunk text **verbatim** — never truncated, summarized, or
rewritten, because a citation that does not quote its source is worse than
none). Many chunks may share one `document_id`; grouping by document is a
caller-side `groupby` on a field that is already present, so no nested
source-tree type is invented, and nothing resembling a citation DSL exists.
`Citation.from_chunk(chunk)` reads a Chunk and never mutates it.

`AnswerResult` (frozen dataclass, exported) is `answer: str` (the Generator's
return value, passed through verbatim — never edited, prefixed, or
post-processed) plus `sources: tuple[Citation, ...]`, in Context order
(best-ranked first) and deduplicated by `chunk_id`, so the same chunk is never
cited twice. An empty context yields `AnswerResult(answer="", sources=())`:
no fabricated source is ever added to justify an answer.

`answer_with_sources(query, *, top_k, candidate_k, max_chars) -> AnswerResult`
is `query(...)` followed by generation and attribution — one retrieval path and
one packing path, so the Context cited is exactly the Context generated from.
`answer(...) -> str` is unchanged and is now literally
`answer_with_sources(...).answer`, so the two can never diverge. Like `answer`,
the new entry point is concrete on `RAGPipeline` (raising `NotImplementedError`
by default) so pipelines predating this phase stay valid, and it raises
`ValueError` — not an answer without sources — when no generator is configured.
Generator errors still propagate unchanged.

**Current limitations** — attribution is chunk-level provenance, nothing more:
no span-level offsets inside a chunk, no per-sentence citation, no answer
verification / faithfulness scoring, no dedup across near-identical chunks (only
exact `chunk_id`), no citation formatting for a UI, and no LLM-generated
citations (deliberately, per above).

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
        user Query ───────────────────────────────────────────┐
        ▼                                                     │  (the original query)
QueryRewriter.rewrite(query)                                  │  optional stage: skipped when no rewriter is injected
        ▼  retrieval query                                    │
HybridRetriever.retrieve(retrieval_query, top_k, candidate_k) │  recall: candidate_k-wide pool
        ▼  list[SearchResult]                                 │
Reranker.rerank(retrieval_query, candidates, top_k)           │  optional stage: narrow to best top_k
        ▼  list[SearchResult]                                 │
ContextBuilder.build(reranked, max_chars)                     │  pack: budget-bound
        ▼                                                     │
Context                                                       │
        ▼                                                     │
Generator.generate(query, context)                    ◄───────┘  optional stage: generate
        ▼
Answer  (+ Sources, via citations_from_context — see Source Attribution)
```

The rail on the right is the caller's query, untouched: it bypasses rewriting
entirely and arrives at the Generator as it was passed in.

`query(query, *, top_k=5, candidate_k=20, max_chars=2000) -> Context`.
Retrieval is asked for a `candidate_k`-wide pool; the reranker -- when one is
injected -- narrows it to the best `top_k`; ContextBuilder packs whatever fits
`max_chars`. Without a reranker the retriever's own ranking stands, cut to
`top_k`. Every stage returns "at most k", so `top_k > candidate_k` simply
cannot be filled. All scores, ordering, and selection come from the injected
stages; the pipeline never sorts, filters, re-scores, or re-packs results
itself. The return value is the existing `Context` (text + chunks); no
`Answer`/`RAGResponse` wrapper is invented.

A configured `query_rewriter` runs first, inside `retrieve()`, exactly once per
public call: its output is what the retriever *and* the reranker see, while the
caller's own query is what reaches `Generator.generate` (see Query
Transformation above). The rewriter never sees the retriever, and neither ever
sees the other's query by accident.

`answer(query, *, top_k=5, candidate_k=20, max_chars=2000) -> str` is exactly
`query(...)` followed by the injected Generator -- one retrieval path and one
packing path, so the Context a generator sees is the one `query` returns. The
answer is the generator's own return value, passed through verbatim; generator
errors propagate unchanged (nothing is caught, retried, or turned into a
fallback answer), the same posture indexing failures already have. With
`generator=None` (the default) `answer` raises `ValueError` rather than
returning raw Context where an answer was expected. `RAGPipeline.answer` is
concrete and raises `NotImplementedError` by default, so pipeline
implementations predating this phase stay valid.

`answer_with_sources(...) -> AnswerResult` adds the attribution for that same
Context; `answer()` is now `answer_with_sources().answer` (see Source
Attribution below).

### Dependency injection

Six collaborators are **required keyword arguments** and three are optional,
validated against their ABCs at construction and never constructed internally:
`chunker` (`Chunker`), `embedder` (`EmbeddingProvider`), `vector_store`
(`VectorStore`), `lexical_index` (`LexicalIndex`), `retriever` (`Retriever`,
hybrid-shaped `retrieve(query, top_k, candidate_k)` -- `HybridRetriever` in
practice), `context_builder` (`ContextBuilder`), and the optional `reranker`
(`Reranker | None = None`), `generator` (`Generator | None = None`), and
`query_rewriter` (`QueryRewriter | None = None`); `None` disables that stage,
which is the default for all three. The wiring contract is that the retriever
searches the same `vector_store` / `lexical_index` that `ingest` writes to; the
caller owns that wiring.

### Responsibilities

Pipeline is responsible for: orchestration, dependency wiring, indexing
coordination (chunk -> embed -> both stores), query coordination (rewrite ->
retrieve -> rerank -> build), generation coordination (query -> generate), and
attribution coordination (Context -> sources). It also owns the query boundary
itself: which query each stage is handed, and that a rewriter runs once per
call rather than per stage.

Pipeline is NOT responsible for (all delegated to injected stages): chunking
algorithm, embedding algorithm, vector similarity, BM25, fusion algorithm,
reranking algorithm, query rewriting algorithm, tokenization, context packing,
prompt construction, answer generation, and the *content* of a citation (== the
injected stages' own Chunk objects -- the pipeline only projects them, and never
asks a model for them).

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

**Current limitations** -- generation can call a real LLM
(`OpenAICompatibleGenerator`, above), but the *bundled* default remains the
extractive `DeterministicGenerator`, and query rewriting is whitespace
normalization only: no semantic query rewriting / expansion / HyDE /
multi-query, no agent, no streaming, no persistence / external vector DB.
`query` still ends at `Context` and `answer` at a `str`; Evaluation (below) is a
separate, offline consumer.

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

## Answer Evaluation

```
Query
 ↓
Retrieval → Reranking → Context → Generation → Answer + Sources
 ↓                                                  ↓
Retrieval Evaluation                        Answer Evaluation
(did we retrieve the right context?)   (did the answer cover the reference
                                        and stay grounded in the context?)
```

Two evaluation layers, deliberately separate, sharing no metric and no code path:

| Layer | Module | Question |
| --- | --- | --- |
| Retrieval Evaluation | `evaluation.py` (+ harness / runner) | Did we retrieve the right context? |
| Answer Evaluation | `answer_evaluation.py` | Did the generated answer cover the expected answer, and stay grounded in the context it was given? |

`AnswerEvaluator.evaluate(query_id, query, answer, context, reference_answer,
sources) -> AnswerEvaluation` scores **one already-produced answer**. It never
retrieves, reranks, packs context, generates, cites, or reads a store, and the
pipeline knows nothing about it: evaluation stays a consumer that runs *after*
`answer_with_sources`, so the same evaluator can score a deterministic answer or
a real LLM's answer without either side changing.

**Three deterministic lexical baselines ship. None of them is a judge.**

- **lexical reference coverage** = `matched / |unique reference tokens|`, where a
  reference token is matched when the answer contains it. Unique tokens only: a
  word repeated in the reference is one fact. An answer that paraphrases
  correctly but shares no vocabulary scores 0 -- the documented ceiling.
- **lexical groundedness** = `grounded sentences / answer sentences`. A sentence
  is grounded when at least `groundedness_threshold` (default 0.5, fixed, never
  tuned) of its unique tokens appear in the context. An empty answer has zero
  sentences and scores 0.0; an empty context grounds nothing; no denominator is
  ever zero, so no NaN can be produced.
- **source-level citation coverage** = the same sentence measure computed against
  the concatenated `Citation.text` of the `AnswerResult`'s sources. It is
  **source-level, not claim-level**: a `Citation` carries no mapping from a
  sentence to the source that supports it, so this layer never attributes one
  sentence to one source, and never parses an answer for citation markers.

Token identity comes from the existing `SimpleTokenizer` after NFC
normalization, so case, whitespace, and punctuation never count as differences
and CJK matches at character granularity.

**The baselines cannot see meaning, and the profile records that rather than
hiding it.** A sentence and its negation score identically (both fully covered,
both fully grounded), because every token is present either way. Likewise a
perfectly grounded sentence that answers a different question is grounded and
uncovered at once. This is why the naming is exact: *lexical* coverage,
*lexical* groundedness, *source-level* citation coverage -- never "semantic
correctness", "faithfulness", or an LLM judge. A real judge belongs behind the
same value type, carrying its own optional dependency, the way `semantic.py`
carries sentence-transformers.

**Evidence** -- `benchmark/run_answer_evaluation.py` runs the deterministic
pipeline over `benchmark/datasets/answer_small.json` (12 cases: the eight answer
shapes plus the negative probes) and writes
`benchmark/results/answer_evaluation_profile.json` with aggregates, per-kind
means, per-query rows, and a retrieval-outcome join. Cases may carry
`answer_override` / `drop_sources` to construct probes the pipeline would never
produce; those rows are labelled `answer_source: "override"` and are excluded
from the retrieval join. The profile is offline, key-free, model-free, and
refuses to write evidence unless two fresh runs agree exactly.

**Not implemented here** -- no LLM judge, no semantic answer similarity, no
embedding-based scoring, no claim extraction, no automated fact checking, no
RAGAS / DeepEval-style framework, and no faithfulness claim of any kind.

## Claim-Level Attribution

```
Answer
 ↓
DeterministicClaimExtractor      (one sentence = one claim)
 ↓  Claim[]
DeterministicClaimAttributor     (token overlap, per claim)
 ↓
AttributionResult
   supported  -- the context covers the claim
   attributed -- a cited source covers the claim
```

The finest-grained evaluation layer, and the one that answers the question
source-level coverage cannot: *which* source carries *which* claim.

**Two questions, two rates, never merged.**

- ``supported`` -- the claim's unique tokens are covered (>= ``threshold``,
  default 0.5) by the **Context** text. The context is therefore a required
  input: support is a property of the claim against the evidence retrieval
  actually produced.
- ``attributed`` -- at least one **Citation** covers the claim the same way.
  ``supporting_citation_ids`` lists every citation that qualifies, in citation
  input order (never re-sorted by score), duplicates collapsed to the first
  occurrence, exactly as ``citations_from_context`` collapses chunk ids.

An answer with no citations can be fully supported and fully unattributed; an
answer can cite a source that does not cover it. Those are the cells the two
rates exist to keep apart, and the profile counts them
(``claims_supported_but_uncited``).

**Deterministic and lexical, at every step.** Claim extraction is sentence
splitting on the ASCII and CJK terminators -- no clause decomposition, no
subject/predicate analysis, no coreference. Support is ``|unique claim tokens ∩
support tokens| / |unique claim tokens|`` over the same NFC +
``SimpleTokenizer`` path answer evaluation uses (``text.py``), so the two layers
cannot drift into measuring different things. A claim with no tokens scores 0.0
and an answer with no sentences produces no claims: no denominator is ever zero
and no NaN is reachable.

**Nothing is parsed out of the answer.** Attribution reads ``Citation.text`` and
``Citation.chunk_id`` and nothing else -- no ``[1]`` markers, no chunk ids
scanned in prose, no citation grammar, and no regex import at all. If the
structured citation is not there, the answer is simply unattributed.

**Known limitations, recorded rather than fixed.** A negation reuses every token
of the sentence it contradicts, so it scores as supported *and* is attributed to
that source; a correct paraphrase with different words scores 0.0. Both are in
the evidence as limitation probes. Entailment, NLI, embeddings, and an LLM judge
are all out of scope here; a real attribution model would sit behind this same
value type with its own optional dependency.

**Evidence** -- ``benchmark/run_attribution_profile.py`` reuses the answer
dataset, the answer harness's pipeline wiring, and the retrieval runner's corpus
ingestion, and writes ``benchmark/results/answer_attribution_profile.json`` with
pooled aggregates, a claim funnel, and one row per claim. The runner executes
the profile twice and writes only if the two runs are byte-identical.

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

## Semantic Retrieval Profile

Phase 11 adds an *optional*, real-semantic-embedding channel on top of the
unchanged deterministic retrieval / evaluation stack. It is a separate evidence
channel only -- it changes no core module, adds no dependency to the default or
dev install, and leaves the frozen deterministic profiles (Phases 10B / 10C /
10D) byte-identical.

### Motivation

Every embedding ZMRAG shipped up to Phase 10D came from
``DeterministicEmbedder`` -- a stdlib, sha256-derived *synthetic* vector that is
byte-stable but has no semantic content. That made the deterministic local
benchmarks reproducible, but it is deliberately unsuitable for any semantic-
quality observation: dense retrieval on the synthetic floor cannot answer
"does a real semantic embedding drive this stack?" Phase 11 adds a real provider
without disturbing the deterministic path, so the two profiles are comparable
only in how they exercise the same retrieval components -- never as competing
claims.

### Architecture

The existing ``EmbeddingProvider`` abstraction (``embed`` / ``embed_many`` /
``dimension`` -> ``Embedding``) is the integration seam and is reused unchanged:
``SemanticEmbeddingProvider`` is a new *implementation* of it, and every stage
downstream is the same public component the deterministic profile already used.
No ABC, retriever, store, fusion, or evaluator was modified.

```
SemanticEmbeddingProvider
        ↓
Embedding
        ↓
InMemoryVectorStore
        ↓
DenseRetriever / HybridRetriever
        ↓
EvaluationHarness
        ↓
RetrievalEvaluator
```

``SemanticEmbeddingProvider`` (``src/zmrag/semantic.py``) wraps
sentence-transformers behind a narrow backend adapter. Exactly one model source
is required (``model_name`` or ``local_model_path``, validated before any
backend work); results are converted to the library ``Embedding`` value object
via ``tolist()``, so numpy / torch objects never leak into ZMRAG and never need
to be imported. ``dimension`` is read from the loaded model at construction.

### Optional dependency boundary

- ``sentence-transformers`` is an **optional** extra only:
  ``[project.optional-dependencies] semantic = ["sentence-transformers>=2.0"]``.
- It is **not** in ``dependencies = []`` and **not** in the ``dev`` extra
  (pytest only), so a default install and the CI test suite never import torch.
- The heavy import happens inside ``_create_backend`` at instantiation time,
  never at module scope -- importing ``zmrag`` or ``zmrag.semantic`` is always
  safe, and constructing the provider without the dependency raises an
  actionable ``ImportError`` naming ``pip install "zmrag[semantic]"``.
- The deterministic profile therefore remains fully dependency-free and
  byte-reproducible across environments.

### Model

``benchmark/run_semantic_profile.py`` pins exactly one model and never
substitutes another:

```
sentence-transformers/all-MiniLM-L6-v2
384 dimensions
normalize_embeddings=True
```

Weights are downloaded on first use unless already cached by
sentence-transformers. The runner records the installed ``python`` /
``sentence_transformers`` / ``torch`` / ``numpy`` versions in the evidence (the
facts that explain why a different environment may differ) -- never a hostname,
username, path, timestamp, or credential.

### Semantic benchmark

The runner feeds the *same* canonical corpus / dataset / chunking the frozen
deterministic profiles used (6 documents / 33 chunks, the 8-case
``retrieval_small.json``, ``FixedSizeChunker(120, 20)``) through the real model
and the unchanged public components. Only retrieval configuration varies, in a
3-cell matrix locked before any run and never tuned: ``S1`` Dense-only,
``S2`` Hybrid + ``WeightedScoreFusion``, ``S3`` Hybrid + ``RRFFusion``, at
unified ``candidate_k`` 33 / ``top_k`` 8, no reranker. Metrics come verbatim
from ``EvaluationResult.to_dict()``:

```
S1 dense             hit_rate 0.875  recall 0.875  precision 0.125  MRR 0.708  MAP 0.719  NDCG 0.756
S2 hybrid+weighted   hit_rate 0.875  recall 0.875  precision 0.125  MRR 0.8125 MAP 0.823  NDCG 0.837
S3 hybrid+rrf        hit_rate 0.875  recall 0.875  precision 0.125  MRR 0.729  MAP 0.740  NDCG 0.774
```

These are **bounded evidence on the fixed Phase 10 corpus/dataset** -- they
observe that a real semantic embedding reaches and drives the existing retrieval
stack, nothing more.

### Evidence separation and reproducibility

Semantic evidence lives in
``benchmark/results/retrieval_semantic_profile.json`` and is labelled
``deterministic: false`` at the top level and on every config row. The three
frozen Phase 10 files (``retrieval_small_baseline.json`` /
``retrieval_small_experiments.json`` / ``retrieval_truncation.json``) are never
written by the semantic runner and stay byte-identical to HEAD. Semantic
inference is *same-environment reproducible* -- the runner executes the whole
profile twice on freshly constructed providers and records
``same_environment_reproducible`` only when the two passes are exactly equal --
but model output depends on the model / torch / dependency versions, so no
byte-identical cross-machine reproduction is claimed and the evidence must never
be merged or labelled as deterministic evidence.

### Synthetic vs semantic

On the same 33-chunk corpus at the same ``top_k = 8``:

```
DeterministicEmbedder dense@8   hit_rate/recall ≈ 0.25   MRR ≈ 0.036   (frozen Phase 10D D1-t8 cell)
Semantic dense@8 (S1)           hit_rate/recall = 0.875  MRR = 0.708
```

This shows the real provider integrates correctly and moves dense retrieval off
the synthetic ``DeterministicEmbedder`` floor. It is **not** general superiority
over BM25 -- the lexical profile reaches the same 0.875 ceiling here, and this
corpus cannot discriminate the two strategies.

### Benchmark interpretation limits

The evidence cannot support general claims:

- only **6 documents / 33 chunks / 8 cases**;
- case 8 references the deliberately unindexed ``missing::0`` chunk, so the
  maximum observable recall of any configuration is 7/8 = 0.875;
- most positive queries have strong lexical / topical overlap, so dense and
  lexical retrieval largely agree at the top;

this benchmark therefore cannot establish dense-vs-BM25 (or fusion) superiority
in general, and makes no real-world performance claim.

### Deterministic profile protection

Phase 11 touches exactly one source file (``src/zmrag/semantic.py``, additive
and never imported by the default graph) and adds optional ``pyproject.toml``
extras plus tests, a benchmark runner, and evidence. ``embedders.py``,
``retrievers.py``, ``stores.py``, ``fusion.py``, ``rerankers.py``,
``pipeline.py``, ``evaluation.py``, ``evaluation_harness.py``, ``__init__.py``
and the Phase 10 benchmark runners are unchanged; the public API is unchanged
(``SemanticEmbeddingProvider`` is *not* exported at the package root).

### Non-goals

Phase 11 does **not** implement generation, answer / generated-answer
evaluation, agents, tool calling, MCP, memory, semantic reranking, persistence,
a new fusion strategy, new evaluation metrics, or a new corpus / dataset. The
retrieval/evaluation stack still ends at ranked retrieval results and an
offline metric result; ZMRAG remains a retrieval layer, not a generated-answer
RAG or Agent system.

## Semantic Cross-Encoder Reranking Profile (Phase 12)

Phase 12 adds an *optional* two-stage retrieval profile on top of Phase 11's
semantic channel: a real cross-encoder re-scores the first-stage candidate pool
before evaluation. Like Phase 11 it is a separate, non-deterministic evidence
channel only -- it changes no core module, adds no new dependency or extra (it
reuses Phase 11's `semantic` extra, since `CrossEncoder` ships in the same
`sentence-transformers` distribution), and leaves the frozen deterministic
profiles (Phases 10B / 10C / 10D) and the Phase 11 semantic evidence
byte-identical.

### Architecture

The existing `Reranker` abstraction (`rerank(query, candidates, top_k)`) is the
integration seam and is reused unchanged: `CrossEncoderReranker`
(`src/zmrag/semantic_rerank.py`) is a new optional *implementation* of it. The
two-stage shape:

```
retrieval -> candidate pool (up to candidate_k=33)
CrossEncoderReranker.rerank -> CrossEncoder scoring of (query, chunk.text)
                               for every candidate in the pool
                            -> top-k SearchResult (top_k=8)
```

`CrossEncoderReranker` wraps sentence-transformers behind a narrow
``score_pairs(query, texts) -> list[float]`` backend adapter (lazy
`_create_backend`, exactly like `semantic.py`). Exactly one model source
(`model_name` or `local_model_path`) is validated before any backend work;
scores are coerced to finite floats (NaN / Inf and count mismatches are
rejected). It honours the `Reranker` contract exactly: it is **subset-only** --
it re-ranks the candidate pool it is given and never recalls, never queries a
store / retriever / index, never generates candidates -- emits fresh
`SearchResult`s, and sorts by `(-score, chunk.id)`.

Nothing else changed: `retrievers.py`, `pipeline.py`, `evaluation.py`,
`rerankers.py`, `stores.py`, `fusion.py`, `embedders.py`, `__init__.py`, and
`pyproject.toml` are untouched. No core module imports `CrossEncoderReranker`,
and it is *not* exported at the package root -- the public API is unchanged. The
deterministic components are unchanged, and the Phase 10 and Phase 11 evidence
files stay frozen (byte-identical to their prior commits).

### Optional dependency and lazy model loading

The reranker uses the same optional `semantic` extra already declared in Phase
11 (`pip install "zmrag[semantic]"`, `sentence-transformers>=2.0`); no new extra
or dependency is added. `CrossEncoder` is imported only inside `_create_backend`
at construction time, never at module scope, so `import zmrag` /
`import zmrag.semantic_rerank` never loads torch / transformers /
sentence-transformers. Constructing the reranker without the dependency raises
an actionable `ImportError`.

### Locked benchmark

`benchmark/run_semantic_rerank_profile.py` keeps the Phase 11 environment -- the
same canonical corpus / dataset / chunking (6 documents / 33 chunks, the 8-case
`retrieval_small.json`, `FixedSizeChunker(120, 20)`) and the same locked
embedding (`sentence-transformers/all-MiniLM-L6-v2`, 384-dim) -- and reranks
with one locked CrossEncoder model, `cross-encoder/ms-marco-MiniLM-L-6-v2`.
First stage recalls a candidate pool of up to `candidate_k=33` (dense directly;
hybrid via `top_k=candidate_k=33`, so fusion never truncates before the
reranker), then the reranker narrows the pool to `top_k=8`. Matrix:

```
R1 dense                      -> CrossEncoder -> 8
R2 hybrid + WeightedScoreFusion -> CrossEncoder -> 8
R3 hybrid + RRFFusion         -> CrossEncoder -> 8
```

Metrics come verbatim from `EvaluationResult.to_dict()`:

```
R1 dense              hit_rate 0.875  recall 0.875  precision 0.125  MRR 0.8125  MAP 0.7916666666666666  NDCG 0.8188313178399557
R2 hybrid + weighted   hit_rate 0.875  recall 0.875  precision 0.125  MRR 0.8125  MAP 0.7916666666666666  NDCG 0.8188313178399557
R3 hybrid + rrf        hit_rate 0.875  recall 0.875  precision 0.125  MRR 0.8125  MAP 0.7916666666666666  NDCG 0.8188313178399557
```

Each row also records `delta_vs_phase11` versus the matching Phase 11 row
(R1 -> S1, R2 -> S2, R3 -> S3) for MRR / MAP / NDCG:

```
R1  MRR +0.10417  MAP +0.07292  NDCG +0.06263
R2  MRR 0.0       MAP -0.03125  NDCG -0.01785
R3  MRR +0.08333  MAP +0.05208  NDCG +0.04465
```

These are **observations on the fixed corpus, not a superiority claim**: the
corpus is only 6 documents / 33 chunks / 8 cases and is lexical-overlap-heavy,
and recall is capped at 7/8 = 0.875 because one case references the deliberately
unindexed `missing::0` chunk. The evidence demonstrates that the two-stage
semantic reranking path (33 -> CrossEncoder -> 8) works end to end on the real
model -- it does not show that CrossEncoder reranking universally improves
retrieval quality.

### Evidence separation and reproducibility

Rerank evidence lives in
`benchmark/results/retrieval_semantic_rerank_profile.json`, labelled
`deterministic: false` (top level and every row), with an honest
`same_environment_reproducible` flag recorded by running the whole matrix twice
on freshly constructed providers / rerankers. The four prior evidence files
(Phase 10 baseline / experiments / truncation and Phase 11 semantic) are never
written by the runner and stay byte-identical to HEAD. Real-model integration
was executed successfully (gated behind `ZMRAG_SEMANTIC_INTEGRATION=1`).

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
