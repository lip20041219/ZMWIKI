# Semantic Retrieval Benchmark (optional): real Dense vs Hybrid

Runner: `benchmark/run_semantic_profile.py` (Phase 11).
Evidence: `benchmark/results/retrieval_semantic_profile.json`.
**Current environment: `available: false`** -- see §4.

This is the *second*, separate evidence channel. It must never be merged into
the deterministic table: the two answer different questions and are not
comparable row for row. See §5.

## 1. Two embedding profiles, two experiments

| | Deterministic profile | Semantic profile (this document) |
| --- | --- | --- |
| Embedder | `DeterministicEmbedder(dimension=8)` | `SentenceTransformerEmbedder` via `SemanticEmbeddingProvider` |
| What it is | sha256 hash stand-in -- **no semantic content** | real `sentence-transformers/all-MiniLM-L6-v2`, 384-d |
| Reproducibility | byte-identical across calls / instances / processes | identical within one pinned model + dependency set; **not** promised across model / torch versions |
| Dependency | none (stdlib only) | `pip install "zmrag[semantic]"` (opt-in) |
| Network | never | model fetch on first use unless cached; never in the test path |
| Used for | deterministic regression + architecture testing | observing real semantic retrieval behaviour |
| Evidence | `retrieval_small_*.json`, `retrieval_truncation.json`, `retrieval_strategy_comparison.json` | `retrieval_semantic_profile.json`, `retrieval_semantic_rerank_profile.json`, `retrieval_strategy_semantic_profile.json` |

`DeterministicEmbedder` is a hash: two texts that mean the same thing get
unrelated vectors, and two texts that share no meaning can score as
"similar". It is exactly what a regression test needs (stable, offline,
free) and exactly what a retrieval-quality claim must not rest on. In the
deterministic strategy table T1 Dense scores `0.0000` across every metric --
that is the hash stand-in failing to rank, **not** a measurement of dense
retrieval. The semantic profile is where dense retrieval is actually observed.

## 2. What the semantic profile measures

Same corpus, same chunking, same queries, same ground truth as the frozen
Phase 10 runs -- only the embedder and the strategy vary. The runner asserts
this rather than assuming it, and indexes exactly the expected chunk count
before any query is issued.

| | |
| --- | --- |
| Corpus | `benchmark/corpus/doc_a.txt` … `doc_f.txt` -- 6 documents, 33 chunks |
| Chunking | `FixedSizeChunker(chunk_size=120, overlap=20)` |
| Queries / ground truth | `benchmark/datasets/retrieval_small.json` -- 8 cases (one has ground truth `missing::0`, so observable recall caps at 7/8 = 0.875) |
| Embedding | `sentence-transformers/all-MiniLM-L6-v2`, 384-d, `normalize_embeddings=True` |
| BM25 | `InMemoryLexicalIndex` + `SimpleTokenizer` (unchanged from the deterministic runs) |
| Serving depth | `top_k = 8` |
| Recall pool | `candidate_k = 33` (the whole corpus), held constant across every cutoff |
| Evaluator | `EvaluationHarness` → `RetrievalEvaluator`; every number is read from `EvaluationResult.to_dict()` |

Strategy matrix, locked before any result was seen:

| ID | Strategy |
| --- | --- |
| S1 | Semantic Dense |
| S2 | Semantic Dense + BM25 → `WeightedScoreFusion` |
| S3 | Semantic Dense + BM25 → `RRFFusion` |
| R1–R3 / T6 | the same three, plus a real cross-encoder -- separate evidence: `retrieval_semantic_rerank_profile.json` (Phase 12) |

Each row carries both its serving-depth metrics (`hit_rate`, `recall`,
`precision`, `mrr`, `map`, `ndcg` at `top_k=8`) and the `@K` cutoff columns
(`hit_rate_at_1`, `recall_at_1`, `recall_at_3`, `recall_at_5`,
`precision_at_5`, `mrr_at_5`, `map_at_5`, `ndcg_at_5`). Every `@K` value comes
from an evaluator run *at that depth*, with `candidate_k` still 33 -- no metric
is re-implemented in the runner, and only the serving depth changes between
cutoff runs.

## 3. Historical results (committed evidence)

> **These numbers were produced in an earlier environment** -- Python 3.11.9,
> `sentence_transformers` 6.0.1, `torch` 2.14.0+cpu, `numpy` 2.4.6, per the
> `environment` block in the evidence file. **They were not re-executed for
> this document**, and the environment that produced them is no longer present
> on this machine (§4). They are reproduced verbatim from
> `benchmark/results/retrieval_semantic_profile.json`.

`top_k = 8`, `candidate_k = 33`, 8 queries, `same_environment_reproducible: true`:

| Strategy | HitRate | Recall | Precision | MRR | MAP | NDCG |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| S1 Semantic Dense | 0.8750 | 0.8750 | 0.1250 | 0.7083 | 0.7188 | 0.7562 |
| S2 Hybrid + Weighted | 0.8750 | 0.8750 | 0.1250 | 0.8125 | 0.8229 | 0.8367 |
| S3 Hybrid + RRF | 0.8750 | 0.8750 | 0.1250 | 0.7292 | 0.7396 | 0.7742 |

`precision` is 0.125 = 1/8 on every row because exactly one relevant chunk per
query is retrieved into 8 slots; it is a property of the corpus
(one gold chunk per query), not of the strategies.

**The `@K` columns are absent from this file.** The committed evidence predates
the cutoff extension (§6); the runner now emits them, but they cannot be
back-filled without re-running the model, and inventing them is not an option.
The file is SHA-256 pinned by `tests/test_semantic_rerank_profile.py` at
`fb4e6f4a…` and is left byte-identical.

### Reading these numbers honestly

The corpus is 6 documents and 8 queries, and one case has no corpus-grounded
answer, so **recall is capped at 0.875 for every strategy**. The remaining
queries are heavily lexical-overlap, which is why S1 and S2 tie on hit
rate / recall and separate only on ranking quality (MRR / MAP / NDCG). At this
size:

- all three strategies tie on hit rate / recall / precision -- the ceiling is
  reached by each, so those columns cannot discriminate between them;
- S2 (weighted fusion) ranks the correct chunk highest most often here;
- **nothing here establishes that hybrid beats dense, or that RRF beats
  weighted fusion.** A 3-way tie at a 7/8 ceiling on 8 queries carries no
  such claim. Re-run on a larger corpus before drawing any conclusion.

## 4. Current status: `available: false`

`sentence-transformers` is not installed in this repository's `.venv` (which
holds only `pytest`, keeping `dependencies = []` and the default test suite
ML-free). The model weights *are* present in the local HF cache
(`models--sentence-transformers--all-MiniLM-L6-v2`), but the library that reads
them is not.

Running the profile therefore reports, exits 0, and **writes nothing**:

```
$ python benchmark/run_semantic_profile.py
semantic profile unavailable: ImportError: SemanticEmbeddingProvider requires
the 'semantic' optional dependency (sentence-transformers + torch).
Install it with: pip install "zmrag[semantic]"
install with: pip install "zmrag[semantic]"
```

`main()` returns

```json
{"available": false,
 "reason": "ImportError: SemanticEmbeddingProvider requires …",
 "install_hint": "pip install \"zmrag[semantic]\""}
```

Rules the runner holds to, and which `tests/test_semantic_profile.py` pins:

- a missing model is a **normal outcome, not a failure** -- exit code 0;
- it **never** writes evidence, and never substitutes zeros or any other
  placeholder for a metric that was not measured;
- it never downloads a model to make the benchmark runnable. Restoring the
  model is a deliberate operator action;
- an absent `available` key in written evidence means the model loaded and the
  numbers are real.

Note the distinction the code enforces: only dependency / model loading
failures degrade to `available: false`. A wrong corpus or dataset still raises,
so a genuine bug can never be silently reported as "the model is missing".

## 5. Why this is not merged into the deterministic table

The deterministic strategy benchmark (`docs/retrieval_strategy_benchmark.md`)
uses `DeterministicEmbedder`, a hash with no semantic content. Its T1 Dense row
is `0.0000` across every metric. Placing that next to the semantic S1 row would
suggest a controlled comparison of dense retrieval methods, when in fact the
deterministic row is measuring the absence of a model.

The two therefore stay in separate files, separate tables, and separate
documentation, with the embedder named on every row. A reader must be able to
tell which channel a number came from without cross-referencing anything.

## 6. Reproducing

```bash
pip install -e ".[semantic]"          # opt-in; never a runtime dependency
python benchmark/run_semantic_profile.py
```

Optional positional argument overrides the output path (the tests use this so
they never overwrite committed evidence). The runner performs the whole profile
twice on two freshly constructed providers and records
`same_environment_reproducible` -- `true` only when both passes are exactly
equal. It never asserts or writes the frozen deterministic files.

Re-running **will** change `benchmark/results/retrieval_semantic_profile.json`,
which is SHA-256 pinned, and will add the `@K` columns missing from §3. That
in turn requires re-checking the Phase 12 cross-phase guard in
`run_retrieval_strategy_profile.py` (`main_semantic`), which compares against
the `R2` row of the rerank evidence. Plan for it rather than being surprised by
it.
