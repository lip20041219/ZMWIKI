# Retrieval Strategy Benchmark: Dense vs Hybrid vs Hybrid + Reranker

Evidence: `benchmark/results/retrieval_strategy_comparison.json` (deterministic),
`benchmark/results/retrieval_strategy_semantic_profile.json` (opt-in).
Runner: `benchmark/run_retrieval_strategy_profile.py`.

## 1. Experimental Setup

Everything except the strategy is frozen and identical across rows. The corpus
and dataset are the Phase 10A ones, untuned and untouched.

| | |
| --- | --- |
| Corpus | `benchmark/corpus/doc_a.txt` … `doc_f.txt` -- 6 documents, 33 chunks |
| Chunking | `FixedSizeChunker(chunk_size=120, overlap=20)` |
| Queries / ground truth | `benchmark/datasets/retrieval_small.json` -- 8 cases (one has ground truth `missing::0`, so observable recall caps at 7/8 = 0.875) |
| Dense embedding | `DeterministicEmbedder(dimension=8)` -- **a sha256 stand-in, not a model** (see Conclusions) |
| BM25 | `InMemoryLexicalIndex` + `SimpleTokenizer` |
| Serving depth | `top_k = 5` |
| Recall pool | `candidate_k = 33` (the whole corpus) for every fused strategy |
| Evaluator | `EvaluationHarness` → `RetrievalEvaluator`; every number below is read from `EvaluationResult.to_dict()` |

Fairness is verified, not asserted. Each strategy is built from scratch by one
builder over the same six documents; `_assert_only_variable_differs` compares
the full configuration and fails unless the only differing keys are the ones the
matrix varies. Each strategy's stores are then read back chunk by chunk against
the frozen corpus, so "same corpus" is a measurement:

```
T1  compared to T1: differs in (nothing)
T2  compared to T1: differs in retriever, fusion, candidate_k     <- all intended
T3  compared to T1: differs in retriever, fusion, candidate_k     <- all intended
T4  compared to T1: differs in retriever, fusion, reranker, candidate_k
T5  compared to T1: differs in retriever, fusion, reranker, candidate_k
```

## 2. Retrieval Strategies

| ID | Strategy | What it does |
| --- | --- | --- |
| T1 | Dense | Embed the query, cosine-search the vector store, return `top_k`. No recall pool. |
| T2 | Hybrid + Weighted | Dense and BM25 each recall `candidate_k`; `WeightedScoreFusion` merges both scored lists; cut to `top_k`. |
| T3 | Hybrid + RRF | Same two sources, merged by `RRFFusion` (reciprocal rank, `rrf_k=60`), which uses ranks only -- no score scale to reconcile. |
| T4 | Hybrid + Weighted + DeterministicReranker | T2's `candidate_k`-wide pool is re-scored by the reranker (lexical-overlap count) and cut to `top_k`. The pool is never pre-cut. |
| T5 | Hybrid + RRF + DeterministicReranker | T3's pool, same reranking stage. |
| T6 | Hybrid + Weighted + semantic embeddings + CrossEncoder *(opt-in)* | Phase 12's locked two-stage configuration, `candidate_k=33 → top_k=8`, real `sentence-transformers/all-MiniLM-L6-v2` + `cross-encoder/ms-marco-MiniLM-L-6-v2`. |

`DeterministicReranker` is **not** a semantic reranker: it counts how many
distinct query tokens a candidate contains. It is deterministic, so it can be
measured and reproduced, but it cannot judge relevance the way a cross-encoder
can.

## 3. Overall Results

`top_k = 5`, `candidate_k = 33`, 8 queries. Every column comes from an evaluator
run at that depth, so all rows are directly comparable:

| Strategy | HitRate@1 | Recall@1 | Recall@3 | Recall@5 | Precision@5 | MRR@5 | MAP@5 | NDCG@5 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| T1 Dense | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| T2 Hybrid + Weighted | 0.6250 | 0.6250 | 0.8125 | 0.8750 | 0.2000 | 0.7500 | 0.7500 | 0.7852 |
| T3 Hybrid + RRF | 0.1250 | 0.1250 | 0.5000 | 0.7500 | 0.1750 | 0.3271 | 0.3365 | 0.4399 |
| T4 Hybrid + Weighted + Rerank | 0.6250 | 0.6250 | 0.8750 | 0.8750 | 0.2000 | 0.7500 | 0.7604 | 0.7905 |
| T5 Hybrid + RRF + Rerank | 0.6250 | 0.6250 | 0.8750 | 0.8750 | 0.2000 | 0.7500 | 0.7604 | 0.7905 |
| T6 Semantic + CrossEncoder *(opt-in, `top_k=8`)* | -- | -- | -- | -- | -- | 0.8125 | 0.7917 | 0.8188 |

T6 is scored at its own locked depth (8) rather than 5, so its `MRR` / `MAP` /
`NDCG` are shown full-depth and are **not** in the same column comparison as
T1–T5; it is listed for reference only.

The same five strategies at the frozen full-pool configuration
(`top_k = candidate_k = 33`), which is what the historical guards compare:

| Strategy | hit_rate | recall | mrr | map | ndcg |
| --- | ---: | ---: | ---: | ---: | ---: |
| T1 Dense | 0.8750 | 0.8750 | 0.0660 | 0.0682 | 0.2294 |
| T2 Hybrid + Weighted | 0.8750 | 0.8750 | 0.7500 | 0.7500 | 0.7852 |
| T3 Hybrid + RRF | 0.8750 | 0.8750 | 0.3479 | 0.3573 | 0.4844 |
| T4 Hybrid + Weighted + Rerank | 0.8750 | 0.8750 | 0.7500 | 0.7604 | 0.7905 |
| T5 Hybrid + RRF + Rerank | 0.8750 | 0.8750 | 0.7500 | 0.7604 | 0.7905 |

All six cross-phase guards pass (`historical_baseline_match: true`): the
full-pool arm reproduces the frozen Phase 10B baseline row and the frozen Phase
10C E1/E3/E4/E5/E6 rows field by field. The run refuses to write evidence
otherwise, and never writes to the old files.

## 4. Per-query Analysis

Rank of the first relevant chunk per strategy (`0` = no relevant chunk in the
top 5; `hit@1/@3/@5` as Y/n):

| # | Query | T1 Dense | T2 Weighted | T3 RRF | T4 W+Rerank | T5 RRF+Rerank |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 1 | `database B-tree index query` | 0 nnn | 1 YYY | 3 YYn | 1 YYY | 1 YYY |
| 2 | `cache eviction memory` | 0 nnn | 1 YYY | 2 YYn | **2 YYn** | 2 YYn |
| 3 | `producer consumer message queue` | 0 nnn | 2 YYn | 5 Ynn | **1 YYY** | 1 YYY |
| 4 | `embedding vector similarity retrieval` | 0 nnn | 1 YYY | 0 nnn | 1 YYY | 1 YYY |
| 5 | `BM25 lexical vector ranking fusion` | 0 nnn | 2 YYn | 4 Ynn | 2 YYn | 2 YYn |
| 6 | `document chunk context` | 0 nnn | 1 YYY | 3 YYn | 1 YYY | 1 YYY |
| 7 | `nearest neighbor search representation` | 0 nnn | 1 YYY | 1 YYY | 1 YYY | 1 YYY |
| 8 | `quantum submarine agriculture` | 0 nnn | 0 nnn | 0 nnn | 0 nnn | 0 nnn |

Rows vs the Dense baseline:

| Strategy | queries with a different ranking | queries with a different top-1 |
| --- | ---: | ---: |
| T1 Dense | 0 | 0 |
| T2 Hybrid + Weighted | 7 | 7 |
| T3 Hybrid + RRF | 7 | 6 |
| T4 Hybrid + Weighted + Rerank | 8 | 8 |
| T5 Hybrid + RRF + Rerank | 8 | 8 |

**What the reranker actually changed** (T2 → T4, the only pair where the
reranker is the sole difference):

- **Improved one query**: `producer consumer message queue` moved the correct
  chunk from rank 2 to rank 1 (its `hit@1` flips n → Y). This is the source of
  the `Recall@3` gain 0.8125 → 0.8750.
- **Degraded one query**: `cache eviction memory` moved the correct chunk from
  rank 1 to rank 2 (its `hit@1` flips Y → n). Net `HitRate@1` is unchanged at
  0.6250 -- one gain, one loss.
- **No change to the first relevant rank** on the other five answerable queries,
  although the rest of each top-5 differs (the reranker pulls in more chunks of
  the same document, because it scores by lexical overlap).
- `quantum submarine agriculture` cannot be hit by any strategy: its ground
  truth is `missing::0`, which is not in the corpus.

So the deterministic reranker is a **wash at the top** (MRR@5 unchanged at
0.7500) and a **small win in the tail** (NDCG@5 0.7852 → 0.7905, MAP@5 0.7500 →
0.7604). It is not a uniform improvement, and it is not a semantic improvement
either -- it reorders by token overlap.

## 5. Candidate Pool Analysis

`top_k = 5` fixed; the reranked strategies get a wider recall pool:

| Strategy | candidate_k | Recall@5 | MRR@5 | NDCG@5 |
| --- | ---: | ---: | ---: | ---: |
| T4 Hybrid + Weighted + Rerank | 5 | 0.8750 | 0.7500 | 0.7905 |
| T4 Hybrid + Weighted + Rerank | 10 | 0.8750 | 0.7500 | 0.7905 |
| T4 Hybrid + Weighted + Rerank | 20 | 0.8750 | 0.7500 | 0.7905 |
| T4 Hybrid + Weighted + Rerank | 33 | 0.8750 | 0.7500 | 0.7905 |
| T5 Hybrid + RRF + Rerank | 5 | 0.8125 | 0.7500 | 0.7522 |
| T5 Hybrid + RRF + Rerank | 10 | 0.8750 | 0.7500 | 0.7905 |
| T5 Hybrid + RRF + Rerank | 20 | 0.8750 | 0.7500 | 0.7905 |
| T5 Hybrid + RRF + Rerank | 33 | 0.8750 | 0.7500 | 0.7905 |

Widening the pool beyond 10 changes **nothing** for either strategy. It is not a
"more candidates → more correct candidates" story on this corpus: the fused
first stage already surfaces every answerable relevant chunk early, so once the
pool is wide enough to contain them, the reranker has nothing more to find. The
only observable effect is at `candidate_k = 5`, where RRF's ordering is narrow
enough to lose the `doc_c::0` chunk the reranker would otherwise promote
(`Recall@5` 0.8125 vs 0.8750). Bigger is not better here -- it is simply
saturated.

## 6. Conclusions

All of the following are what this run measured; none of it is a general claim.

1. **Hybrid beats dense on this corpus, but that comparison is not
   informative.** Dense scores 0.0000 at `top_k = 5` across all eight queries,
   and its full-pool MRR is 0.0660 -- essentially an arbitrary order. The reason
   is the embedder: `DeterministicEmbedder` maps text to a sha256-derived vector,
   which carries no semantic similarity at all. The dense row measures the
   *plumbing*, not dense retrieval. **This benchmark cannot support "hybrid >
   dense"**, and no such claim is made.
2. **Weighted fusion clearly beats RRF on this corpus** (MRR@5 0.7500 vs 0.3271;
   NDCG@5 0.7852 vs 0.4399). RRF discards scores and keeps only ranks, so the
   noisy dense ranking (see 1) dilutes the strong BM25 signal. The frozen Phase
   10C matrix says the same thing (E3 0.75 vs E4 0.3479).
3. **The deterministic reranker gives a small, mixed improvement, not a
   uniform one.** Over the same pool, Weighted → Weighted+Rerank moves NDCG@5
   0.7852 → 0.7905 and MAP@5 0.7500 → 0.7604, leaves MRR@5 flat at 0.7500, and
   swaps one `hit@1` win for one `hit@1` loss. It is worth having as a stage,
   but "reranking improves retrieval" is not what these numbers say.
4. **Reranking erases the fusion choice.** T4 and T5 are identical to four
   decimals, because the reranker re-scores the *whole* pool by its own signal,
   discarding the incoming order. That is also why RRF's deficit disappears:
   the reranker repairs an ordering it never uses.
5. **A wider candidate pool does not help** beyond `candidate_k = 10` here; the
   metric is saturated, not improved. `candidate_k` is a recall knob whose
   benefit is corpus-dependent, and on a 33-chunk corpus it runs out of work.
6. **With a real model the picture changes** (T6): semantic embeddings plus a
   cross-encoder reach MRR 0.8125 / NDCG 0.8188 at depth 8, the best ranking
   quality in this profile -- measured on a 6-document synthetic corpus, on one
   locked model pair, and not comparable column-for-column with T1–T5 because
   its depth differs. It is a plumbing-and-ranking check, not a SOTA claim.

## Reproduce

```bash
python benchmark/run_retrieval_strategy_profile.py                              # T1-T5, no model, no network
HF_HUB_OFFLINE=1 python benchmark/run_retrieval_strategy_profile.py --semantic  # T6
python -m pytest -q
```

The semantic arm reports `available: false` with a reason and writes nothing if
the `semantic` extra or the local model cache is missing; it never downloads.
