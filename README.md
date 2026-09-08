# ZMRAG

An independent, engineering-grade **Knowledge / Retrieval Layer** — a reusable, testable, model-agnostic RAG foundation component.

> ZMRAG = Knowledge / Retrieval
> (ZMAI = Reasoning / Planning; ZMForge = Execution / Tools / Lifecycle)
> ZMRAG is intentionally **decoupled** from ZMAI and ZMForge for now.

## Pipeline

```
Document/Data
→ Ingestion
→ Chunking
→ Embedding
→ Indexing
→ Retrieval
→ Reranking
→ Context Construction
→ Evaluation
```

## Status

Phase 10D — Retrieval Truncation Study. Over the same fixed Phase 10C
environment (local synthetic corpus, dataset, chunking, deterministic
embedding, unchanged evaluator), a controlled study varies `candidate_k` in
{4, 8, 16, 33} and `top_k` in {4, 8} to separate candidate selection from final
ranking depth across an 18-cell matrix (D1 Dense-only, D2 Lexical-only, D3
Hybrid + Weighted, D4 Hybrid + RRF; no reranker, no pipeline). Phase 10C ran the
same environment at the full-pool depth `top_k`/`candidate_k` 33/33 (E1
Dense-only, E2 Lexical-only, E3 Hybrid + Weighted, E4 Hybrid + RRF, E5 Hybrid +
Weighted + DeterministicReranker, E6 Hybrid + RRF + DeterministicReranker).
`benchmark/run_retrieval_experiments.py` composes each configuration directly
from existing public retrieval/evaluation components — never through the
pipeline, which cannot express single-source or no-reranker configurations —
verifies same-instance and cross-instance determinism plus an E5 recompute
guard against the frozen Phase 10B baseline, and writes machine-readable
evidence (`benchmark/results/retrieval_small_experiments.json`) whose metrics
come verbatim from `EvaluationResult.to_dict()`. Phase 10D's
`benchmark/run_retrieval_truncation.py` composes each cell the same way, verifies
same-instance and cross-instance determinism plus an E3 cross-phase guard
against the frozen Phase 10C row, and writes machine-readable evidence
(`benchmark/results/retrieval_truncation.json`) whose metrics come verbatim from
`EvaluationResult.to_dict()`. The Phase 10B reproducible
baseline (`benchmark/run_retrieval_small_baseline.py` →
`benchmark/results/retrieval_small_baseline.json`) remains frozen and is never
overwritten. Results are deterministic, reproducible local validation numbers
on a small synthetic corpus; they are not a claim of real-world benchmark
performance. The data-to-context pipeline is
implemented end-to-end and tested: ingestion (text / `.txt` file / single
directory) → Document → chunking → embedding → vector + lexical indexing →
dense / lexical / hybrid retrieval → fusion → reranking → context building →
`Context`, exposed both as `query()` (→ `Context`) and the evaluation-facing
`retrieve()` (→ real ranked `SearchResult[]`, the single shared retrieval path).
An offline, deterministic retrieval Evaluation layer (Hit Rate / Recall /
Precision plus the ranking metrics MRR / MAP / NDCG with optional `@K` cutoffs,
macro-averaged over queries) consumes pipeline output with no reverse coupling,
wrapped by an `EvaluationHarness` that validates and runs prepared (cases,
predictions) pairs against an injected evaluator. A `RetrievalEvaluationRunner`
drives real pipelines through that harness: per `EvaluationCase` it calls
`pipeline.retrieve` exactly once and hands the resulting predictions to
`harness.run`, reporting results (`EvaluationResult.to_dict` / `summary`). An
`EvaluationCaseSet` organizes cases as an ordered, immutable, JSON-friendly
input contract, a `JsonEvaluationDatasetLoader` loads that set from a local,
strictly-schema-checked JSON file, and a `BenchmarkRunner` now composes it all
into one offline execution entry point (`EvaluationCaseSet → EvaluationResult`
through an injected `RetrievalEvaluationRunner`, exactly-once, result passed
through unchanged). A small, hand-written, fully local synthetic corpus (six
`.txt` documents) and JSON evaluation cases now live under `benchmark/` and run
through that same chain unchanged (`benchmark/datasets/retrieval_small.json`
loads through `JsonEvaluationDatasetLoader`). No external benchmark datasets
(BEIR / MS MARCO / HuggingFace), no Dataset framework, no remote download. All
stages are deterministic, dependency-injected, network-free, and unit-tested.
No LLM generation, benchmark scores, external evaluation frameworks,
persistence, or external services yet.

## Development

```bash
pip install -e ".[dev]"
pytest
```

See [docs/architecture.md](docs/architecture.md) for design intent.
