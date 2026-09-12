# Query Rewriter Benchmark: Baseline vs DeterministicQueryRewriter

Evidence: `benchmark/results/retrieval_query_rewriter_profile.json` (deterministic)
and `benchmark/results/retrieval_query_rewriter_semantic_profile.json` (opt-in).
Runner: `benchmark/run_query_rewriter_profile.py`.

## Objective

Answer one question on the existing retrieval benchmark: **does the shipped
`DeterministicQueryRewriter` (whitespace normalization only) actually change
anything?**

No feature was added and no metric was re-implemented: the experiment runs the
canonical Phase 10B evaluation chain twice, with and without the rewriter
injected, and reads every number from `EvaluationResult.to_dict()`.

## Dataset

Unchanged and reused verbatim -- nothing was regenerated, edited, or tuned:

| | |
| --- | --- |
| Corpus | `benchmark/corpus/doc_a.txt` … `doc_f.txt` -- 6 documents, 33 chunks (`FixedSizeChunker(chunk_size=120, overlap=20)`) |
| Queries / ground truth | `benchmark/datasets/retrieval_small.json` -- 8 cases, one of which (`quantum submarine agriculture`) has ground truth `missing::0`, capping observable recall at 7/8 = 0.875 |
| Documents / chunks / cases | 6 / 33 / 8, asserted before any run |

This is the same local, synthetic corpus the frozen Phase 10B–12 evidence uses.
It is not BEIR, not MS MARCO, and not a large-scale result.

## Protocol

> The only variable is whether the Pipeline uses `DeterministicQueryRewriter`.

```
Baseline          Query ─────────────────────────────► Retrieval
Query rewritten   Query ──► DeterministicQueryRewriter ──► Retrieval
```

Both pipelines are built by the *same* function in the runner, ingest the same
documents, and share every other setting: chunker, embedder, stores, retriever,
fusion, reranker, context builder, dataset, ground truth, `top_k`,
`candidate_k = 33`. The runner asserts this (`_assert_only_variable_differs`):
the baseline pipeline's `query_rewriter` is `None`, the other's is a
`DeterministicQueryRewriter`, and every other collaborator has the same type.

Scenarios run:

| ID | Retrieval | Reranker | candidate_k | Depth sweep |
| --- | --- | --- | --- | --- |
| Q1 | Hybrid + Weighted fusion | DeterministicReranker | 33 | `top_k` ∈ {1, 3, 5, 33} |
| Q2 | Hybrid + RRF fusion | DeterministicReranker | 33 | `top_k` ∈ {1, 3, 5, 33} |
| Q3 (opt-in) | Hybrid + Weighted fusion (semantic embeddings) | CrossEncoderReranker | 33 | `top_k` = 8 (Phase 12's locked depth) |

Because the shipped reranker only ever re-scores and cuts the candidates it is
given, a run at depth K produces the framework's own Recall / MRR / MAP / NDCG
over exactly K results -- i.e. the `@K` values, computed by the evaluator over
results the pipeline really returned. The recall pool (`candidate_k`) is never
widened or narrowed between variants.

Guards, all stop-and-report -- evidence is written only if every one passes:
corpus/dataset boundary checks (Phase 10B's own, reused), same-instance and
cross-instance determinism on the real chain, and two cross-phase guards
requiring this run's baselines to reproduce the *frozen* Phase 10B and Phase 12
R2 rows exactly.

`--semantic` (Q3) needs the optional `semantic` extra and a model; it was run
with `HF_HUB_OFFLINE=1` against the local model cache, so that run made no
network request either. The default run loads no model and touches no network.

## Results

Every row below is printed twice -- once per variant. The two variants produced
**identical metrics in every scenario and at every depth**
(`variants_identical: true` in the evidence).

Q1 -- Hybrid + Weighted + DeterministicReranker (`candidate_k=33`):

| Pipeline | Recall@1 | Recall@3 | Recall@5 | MRR | MAP | NDCG@5 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Baseline | 0.6250 | 0.8750 | 0.8750 | 0.7500 | 0.7604 | 0.7905 |
| Query Rewrite | 0.6250 | 0.8750 | 0.8750 | 0.7500 | 0.7604 | 0.7905 |

MRR / MAP are at full depth (33) and NDCG@5 at depth 5, matching the frozen
Phase 10B configuration. The full depth sweep, unchanged between variants:

| Row | hit_rate | recall | precision | mrr | map | ndcg |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Q1-baseline-t1 | 0.6250 | 0.6250 | 0.6250 | 0.6250 | 0.6250 | 0.6250 |
| Q1-query_rewritten-t1 | 0.6250 | 0.6250 | 0.6250 | 0.6250 | 0.6250 | 0.6250 |
| Q1-baseline-t3 | 0.8750 | 0.8750 | 0.3333 | 0.7500 | 0.7604 | 0.7905 |
| Q1-query_rewritten-t3 | 0.8750 | 0.8750 | 0.3333 | 0.7500 | 0.7604 | 0.7905 |
| Q1-baseline-t5 | 0.8750 | 0.8750 | 0.2000 | 0.7500 | 0.7604 | 0.7905 |
| Q1-query_rewritten-t5 | 0.8750 | 0.8750 | 0.2000 | 0.7500 | 0.7604 | 0.7905 |
| Q1-baseline-t33 | 0.8750 | 0.8750 | 0.0303 | 0.7500 | 0.7604 | 0.7905 |
| Q1-query_rewritten-t33 | 0.8750 | 0.8750 | 0.0303 | 0.7500 | 0.7604 | 0.7905 |

The Q1 baseline at depth 33 (`hit_rate` 0.875, `recall` 0.875, `precision`
0.0303, `mrr` 0.75, `map` 0.7604, `ndcg` 0.7905) is byte-equal to the frozen
Phase 10B evidence -- the cross-phase guard enforces that before writing.

Q2 -- Hybrid + RRF + DeterministicReranker (`candidate_k=33`) produced the same
rows numerically. That collapse is not new here and is not caused by the
rewriter: the frozen Phase 10C matrix shows the same thing (E5 weighted and E6
RRF both reach `mrr` 0.75 / `ndcg` 0.7905 once the deterministic reranker is
applied over the full 33-chunk pool, while without a reranker E3 weighted 0.75
and E4 RRF 0.3479 differ sharply). At full pool depth the subset-only reranker's
own token-overlap score decides the final order, so the fusion order that
reached it is erased -- identically for both variants.

Q3 -- Hybrid + Weighted + CrossEncoderReranker, 33 → 8 (semantic embeddings):

| Pipeline | hit_rate | recall | precision | MRR | MAP | NDCG |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Baseline | 0.8750 | 0.8750 | 0.1250 | 0.8125 | 0.7917 | 0.8188 |
| Query Rewrite | 0.8750 | 0.8750 | 0.1250 | 0.8125 | 0.7917 | 0.8188 |

The baseline row reproduces the frozen Phase 12 R2 row exactly (cross-phase
guard). This scenario is recorded `deterministic: false`, like all model-backed
Phase 11/12 evidence.

## Per-query Analysis

`per_query` in the evidence records, for each of the 8 benchmark cases in each
deterministic scenario: the original query, the retrieval query the rewriter
produced, the top-5 ids of both variants, whether the full 33-deep rankings are
identical, the rank of the first relevant chunk, and the `hit@1/3/5` flags.

| # | Query | Retrieval query | Rewrite identical | Rank of first relevant | hit@1 | hit@3 | hit@5 |
| --- | --- | --- | :---: | ---: | :---: | :---: | :---: |
| 1 | `database B-tree index query` | same | yes | 1 | yes | yes | yes |
| 2 | `cache eviction memory` | same | yes | 2 | no | yes | yes |
| 3 | `producer consumer message queue` | same | yes | 1 | yes | yes | yes |
| 4 | `embedding vector similarity retrieval` | same | yes | 1 | yes | yes | yes |
| 5 | `BM25 lexical vector ranking fusion` | same | yes | 2 | no | yes | yes |
| 6 | `document chunk context` | same | yes | 1 | yes | yes | yes |
| 7 | `nearest neighbor search representation` | same | yes | 1 | yes | yes | yes |
| 8 | `quantum submarine agriculture` | same | yes | 0 (none) | no | no | no |

- Queries with an identical result: **all 8** -- every query, in both variants,
  at every depth.
- Queries whose ranking changed: **0** (`ranking_identical: true` for all 16
  records across Q1 and Q2, compared 33 deep).
- Queries whose `hit@K` changed: **0** for K ∈ {1, 3, 5}.
- Regressions caused by the rewrite: **none observed** -- and none possible in
  this run, since the retrieval query was byte-identical to the original for
  every case (`queries_changed_by_rewriter: 0`).

The per-query `hit@K` flags are diagnostics read off the ranked ids the
pipeline returned; they are not a metric implementation -- nothing is averaged
or scored there, and every number in the tables above comes from the evaluator.

## Query transformation sanity check

For every benchmark query the recorded `retrieval_query` is byte-identical to
the original, so the transformation is trivially whitespace-only here: no case
change, no punctuation removed, no CJK rewriting, no token expansion, no
synonyms, no second query. This matches the rewriter's contract
(`" ".join(query.split())`) and is asserted at the pipeline level by the unit
tests.

To check what the rewriter does when it *does* fire, the evidence carries a
separate `mechanism_probe`: each benchmark query re-typed in a deliberately
de-normalized form (padded, doubled internal spaces). Its queries are
diagnostics only -- they never replace a dataset query and never enter any
metric.

| Measurement | Result |
| --- | ---: |
| Probes where the rewrite changed the string | 8 / 8 |
| … of those, canonical pipeline ranking changed | **0 / 8** |
| … of those, lexical (BM25) ranking changed | **0 / 8** |
| … of those, dense (deterministic embedder) ranking changed | 8 / 8 |

So even when the rewriter acts, BM25 is unaffected -- `SimpleTokenizer` drops
whitespace outright, so a whitespace difference cannot reach a term. The dense
side *does* reorder, because `DeterministicEmbedder` hashes the raw text, so a
padded query is a different vector; that effect is noise rather than signal
(dense-only retrieval scores `mrr` 0.066 in the frozen Phase 10C E1 row), and
in the canonical pipeline it is erased by the reranker over the full pool.

## Conclusion

**No improvement, and none expected -- and now measured rather than assumed.**

`DeterministicQueryRewriter` performs whitespace normalization only, and the
benchmark's queries are already normalized, so it returns the identical string
for all 8 cases (`queries_changed_by_rewriter: 0`). Identical retrieval queries
produce identical candidates, rankings, and metrics: all three scenarios, at
every depth, return byte-equal metrics for both variants, and no per-query
ranking changed.

This is the honest outcome, not a failure: the stage exists so that
`query_rewriter=None` has a deterministic baseline to be compared against, and
this experiment establishes that baseline as exactly "no change on this corpus".
It says nothing about whether a *semantic* rewriter (an LLM rewriter, HyDE,
multi-query expansion) would help -- that is a different stage behind the same
contract, still unimplemented.

What would change the result: a plain `DeterministicQueryRewriter` cannot alter
these metrics for any configuration, because its output is the same string for
already-normalized input, and because the canonical pipeline's final order is
decided by token-based scores that ignore whitespace. What could change is the
dense-only ordering under the hash-based stand-in embedder -- a difference with
no relevance signal behind it, and one that a real embedding model (which does
respect whitespace as text) would treat differently. That has not been measured
here.

## Reproduce

```bash
python benchmark/run_query_rewriter_profile.py                 # Q1 + Q2, no model, no network
HF_HUB_OFFLINE=1 python benchmark/run_query_rewriter_profile.py --semantic   # Q3
python -m pytest -q
```
