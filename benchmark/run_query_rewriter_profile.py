"""Query rewriter profile: baseline vs DeterministicQueryRewriter, E2E.

Answers one question: does the shipped :class:`DeterministicQueryRewriter`
(whitespace normalization only) change anything on the existing retrieval
benchmark? The experiment is a *controlled* one -- the only variable between
the two runs is whether the pipeline was built with ``query_rewriter=None`` or
``query_rewriter=DeterministicQueryRewriter()``:

    Baseline         Query -> Retrieval
    Query rewritten  Query -> DeterministicQueryRewriter -> Retrieval

Everything else is identical by construction and asserted below: corpus,
chunking, embedding, stores, retriever, fusion, reranker, dataset, ground
truth, ``top_k``, ``candidate_k``, and the evaluation chain. No query is
edited, no ground truth is touched, no cutoff is widened for the rewritten
run, and both variants are built by the *same* function, so a difference can
only come from the rewriter.

Two deterministic retrieval scenarios are run (Phase 10B's canonical
pipeline and the same pipeline with RRF fusion), each at four depths
(``top_k`` in 1/3/5/33 with the recall pool held at ``candidate_k=33``).
Because the pipeline's reranker is a subset-only cut, those depths give the
framework's own Recall / MRR / MAP / NDCG at depth K -- no metric is
re-implemented here, every number comes verbatim from
``EvaluationResult.to_dict()``.

The third scenario the project supports -- Hybrid + Cross-Encoder reranking
(Phase 12) -- needs the optional ``semantic`` extra and a model, so it is a
separate, explicitly opt-in run (``--semantic``) writing its own evidence file,
exactly as Phase 11/12 split model-backed evidence from deterministic evidence.
The default run stays dependency-free and byte-reproducible.

Guards, all stop-and-report (nothing is written unless every one passes):

- corpus / dataset boundary checks (Phase 10B's own, reused);
- same-instance and cross-instance determinism on the real chain;
- Q1 baseline @33/33 must reproduce the *frozen* Phase 10B evidence file
  exactly (read-only cross-phase guard: this path equals the proven one);
- every benchmark query must be byte-identical before and after rewriting
  (recorded as ``queries_changed_by_rewriter``, honest either way).

Run from the repository root:

    python benchmark/run_query_rewriter_profile.py
    python benchmark/run_query_rewriter_profile.py --semantic   # needs the extra

For a guaranteed network-free semantic run (cached models), prefix
``HF_HUB_OFFLINE=1``; without it, sentence-transformers still checks the Hub for
model metadata. The default run never loads a model and never touches a network.

An optional positional output path overrides the default evidence location (so
tests never overwrite the committed evidence file).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Bootstrapping for the uninstalled src/ layout (mirrors the pyproject
# ``pythonpath = ["src"]`` pytest setting), so the script runs as-is from the
# repo root without ``pip install``. Stdlib only.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from zmrag import (
    DenseRetriever,
    DeterministicBenchmarkRunner,
    DeterministicContextBuilder,
    DeterministicEmbedder,
    DeterministicQueryRewriter,
    DeterministicRAGPipeline,
    DeterministicReranker,
    EvaluationHarness,
    FixedSizeChunker,
    HybridRetriever,
    InMemoryLexicalIndex,
    InMemoryVectorStore,
    JsonEvaluationDatasetLoader,
    LexicalRetriever,
    RetrievalEvaluationRunner,
    RetrievalEvaluator,
    RRFFusion,
    WeightedScoreFusion,
)

#: Sibling Phase 10B runner: the canonical corpus ingestion + boundary checks.
import run_retrieval_small_baseline as _baseline

#: Repo root, resolved from this file so the script runs from any cwd.
ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = ROOT / "benchmark" / "datasets" / "retrieval_small.json"
BASELINE_PATH = ROOT / "benchmark" / "results" / "retrieval_small_baseline.json"
RERANK_PATH = ROOT / "benchmark" / "results" / "retrieval_semantic_rerank_profile.json"
DEFAULT_EVIDENCE_PATH = ROOT / "benchmark" / "results" / "retrieval_query_rewriter_profile.json"
DEFAULT_SEMANTIC_PATH = (
    ROOT / "benchmark" / "results" / "retrieval_query_rewriter_semantic_profile.json"
)

#: Stable identifiers used in the evidence (never an absolute path).
DATASET_ID = "benchmark/datasets/retrieval_small.json"
EXPERIMENT_ID = "retrieval_query_rewriter_profile"
SEMANTIC_EXPERIMENT_ID = "retrieval_query_rewriter_semantic_profile"
REWRITER_ID = "DeterministicQueryRewriter"
REFERENCE_EXPERIMENT = "retrieval_small_baseline"
SEMANTIC_REFERENCE_EXPERIMENT = "retrieval_semantic_rerank_profile"

#: Fixed experiment environment -- identical to Phase 10B / 10C / 10D.
CHUNK_SIZE = 120
OVERLAP = 20
EXPECTED_DOCUMENTS = 6
EXPECTED_TOTAL_CHUNKS = 33
EXPECTED_CASES = 8

#: The recall pool is never widened or narrowed for either variant.
CANDIDATE_K = 33

#: Depths the deterministic scenarios are run at. With a subset-only reranker,
#: depth K yields the framework's Recall / MRR / MAP / NDCG over exactly K
#: results -- i.e. the @K metrics, computed by the evaluator, not here.
DEPTHS = (1, 3, 5, 33)

#: The six metrics every evidence row carries (counts live at the top level).
METRIC_FIELDS = ("hit_rate", "recall", "precision", "mrr", "map", "ndcg")

#: Metric/count keys shared by EvaluationResult.to_dict and the evidence.
RESULT_FIELDS = (
    "total_cases",
    "evaluated_cases",
    "hit_rate",
    "recall",
    "precision",
    "mrr",
    "map",
    "ndcg",
)

#: The two variants. The ONLY difference between them is the rewriter.
BASELINE_VARIANT = "baseline"
REWRITTEN_VARIANT = "query_rewritten"

#: The locked deterministic scenarios -- fixed here, never tuned against results.
CONFIGS = (
    {"id": "Q1", "fusion": "weighted"},
    {"id": "Q2", "fusion": "rrf"},
)

#: The Phase 12 two-stage configuration, reused verbatim for the opt-in run:
#: recall 33 -> CrossEncoder -> 8. The two models are Phase 12's locked ones --
#: never substituted, and checked against the frozen evidence before running.
SEMANTIC_CONFIG = {"id": "Q3", "fusion": "weighted", "candidate_k": 33, "top_k": 8}
EMBEDDING_MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
RERANKER_MODEL_ID = "cross-encoder/ms-marco-MiniLM-L-6-v2"

#: Deterministic per-query diagnostics are recorded at the full pool depth.
PER_QUERY_DEPTH = 33
#: How many ranked ids each per-query record keeps (full-depth comparison still
#: uses every id; this is only what the evidence stores).
PER_QUERY_HEAD = 5


def _fusion(name: str):
    if name == "weighted":
        return WeightedScoreFusion()
    if name == "rrf":
        return RRFFusion()  # default rrf_k = 60 -- never tuned
    raise ValueError(f"unknown fusion: {name!r}")


def build_pipeline(documents: list, *, fusion: str, rewriter):
    """One freshly built pipeline + evaluation chain, then ingest ``documents``.

    ``rewriter`` is the experiment's single variable. Everything else -- corpus,
    chunking, embedder, stores, retriever, fusion, reranker, context builder --
    is constructed identically for both variants by this one function.
    """
    chunker = FixedSizeChunker(chunk_size=CHUNK_SIZE, overlap=OVERLAP)
    embedder = DeterministicEmbedder()
    vector_store = InMemoryVectorStore()
    lexical_index = InMemoryLexicalIndex()
    hybrid = HybridRetriever(
        DenseRetriever(embedder, vector_store),
        LexicalRetriever(lexical_index),
        _fusion(fusion),
    )
    pipeline = DeterministicRAGPipeline(
        chunker=chunker,
        embedder=embedder,
        vector_store=vector_store,
        lexical_index=lexical_index,
        retriever=hybrid,
        reranker=DeterministicReranker(),
        context_builder=DeterministicContextBuilder(),
        query_rewriter=rewriter,
    )
    for document in documents:
        pipeline.ingest(document)
    runner = RetrievalEvaluationRunner(
        pipeline=pipeline, harness=EvaluationHarness(RetrievalEvaluator())
    )
    return pipeline, DeterministicBenchmarkRunner(evaluation_runner=runner)


def _assert_equal(label: str, a, b) -> None:
    """Stop-and-report guard: two runs must be identical in every view."""
    checks = {
        "result equality": a == b,
        "to_dict equality": a.to_dict() == b.to_dict(),
        "summary equality": a.summary() == b.summary(),
    }
    failed = [name for name, ok in checks.items() if not ok]
    if failed:
        raise RuntimeError(f"{label}: determinism failed on {', '.join(failed)}")


def _assert_only_variable_differs(baseline_pipe, rewritten_pipe) -> None:
    """The two pipelines must differ in the rewriter and in nothing else."""
    if baseline_pipe.query_rewriter is not None:
        raise RuntimeError("baseline pipeline must have query_rewriter=None")
    if not isinstance(rewritten_pipe.query_rewriter, DeterministicQueryRewriter):
        raise RuntimeError(
            "rewritten pipeline must carry a DeterministicQueryRewriter, got "
            f"{type(rewritten_pipe.query_rewriter).__name__}"
        )
    for name in (
        "chunker",
        "embedder",
        "vector_store",
        "lexical_index",
        "retriever",
        "reranker",
        "context_builder",
    ):
        left, right = type(getattr(baseline_pipe, name)), type(getattr(rewritten_pipe, name))
        if left is not right:
            raise RuntimeError(f"collaborator {name!r} differs: {left} != {right}")


def _load_cases():
    case_set = JsonEvaluationDatasetLoader().load(DATASET_PATH)
    if len(case_set) != EXPECTED_CASES:
        raise RuntimeError(f"dataset has {len(case_set)} cases, expected {EXPECTED_CASES}")
    return case_set


def _run_depth(chain, case_set, top_k: int):
    """One real benchmark run through the canonical evaluation chain."""
    return chain.run(case_set, top_k=top_k, candidate_k=CANDIDATE_K)


def _run_variant(documents: list, case_set, *, fusion: str, rewriter, depths):
    """Run one variant at every depth, twice, and require identical results.

    Returns ``{(top_k, "baseline"|"query_rewritten"): EvaluationResult}`` for a
    freshly built pipeline, with a same-instance determinism guard per depth.
    """
    variant = REWRITTEN_VARIANT if rewriter is not None else BASELINE_VARIANT
    pipeline, chain = build_pipeline(documents, fusion=fusion, rewriter=rewriter)
    results = {}
    for top_k in depths:
        first = _run_depth(chain, case_set, top_k)
        _assert_equal(f"same-instance {fusion} {variant} top_k={top_k}", first, _run_depth(chain, case_set, top_k))
        results[(top_k, variant)] = first
    return pipeline, results


def _per_query_record(pipeline_b, pipeline_r, case) -> dict:
    """Per-query diagnostics: the transform, both rankings, and the hit@K flags.

    Rankings come from the same ``pipeline.retrieve`` call the evaluation runner
    makes (deterministic, so identical to what the metrics were computed over).
    The hit@K flags are diagnostics derived from those ranked ids -- not a
    metric implementation: nothing is averaged, scored, or re-derived here.
    """
    relevant = set(case.relevant_chunk_ids)
    retrieval_query = DeterministicQueryRewriter().rewrite(case.query)
    base_ids = [r.chunk.id for r in pipeline_b.retrieve(case.query, top_k=PER_QUERY_DEPTH, candidate_k=CANDIDATE_K)]
    rew_ids = [r.chunk.id for r in pipeline_r.retrieve(case.query, top_k=PER_QUERY_DEPTH, candidate_k=CANDIDATE_K)]
    hits = {k: any(cid in relevant for cid in base_ids[:k]) for k in DEPTHS if k != PER_QUERY_DEPTH}
    rew_hits = {k: any(cid in relevant for cid in rew_ids[:k]) for k in hits}
    rank = next((i for i, cid in enumerate(base_ids, start=1) if cid in relevant), 0)
    return {
        "query": case.query,
        "retrieval_query": retrieval_query,
        "rewrite_identical": retrieval_query == case.query,
        "relevant_chunk_ids": list(case.relevant_chunk_ids),
        "baseline_top_5": base_ids[:PER_QUERY_HEAD],
        "query_rewritten_top_5": rew_ids[:PER_QUERY_HEAD],
        "ranking_identical": base_ids == rew_ids,
        "rank_of_first_relevant": rank,
        "hit_at_1": hits[1],
        "hit_at_3": hits[3],
        "hit_at_5": hits[5],
        "hit_at_k_changed": hits != rew_hits,
    }


def _mechanism_probe(documents: list, case_set) -> dict:
    """Does whitespace normalization ever change a ranking? (diagnostic only)

    The benchmark queries are already normalized, so the main experiment cannot
    observe the rewriter doing anything. This probe feeds each query in a
    deliberately *de-normalized* form (padding + doubled internal spaces) to
    both variants and compares the resulting rankings. The probe queries are
    derived diagnostics -- they are **not** part of any metric, no ground truth
    is touched, and they never enter the metric tables.

    It exists to separate two different claims: "the rewriter did not fire"
    (true, but uninformative) from "the rewriter cannot change this retrieval"
    (a statement about the retriever, which the probe can show).
    """
    pipeline_b, _ = build_pipeline(documents, fusion=CONFIGS[0]["fusion"], rewriter=None)
    pipeline_r, _ = build_pipeline(
        documents, fusion=CONFIGS[0]["fusion"], rewriter=DeterministicQueryRewriter()
    )
    # Raw single-source retrievers over the *same* ingested stores, to see where
    # the normalization is visible at all.
    dense = DenseRetriever(pipeline_b.embedder, pipeline_b.vector_store)
    lexical = LexicalRetriever(pipeline_b.lexical_index)

    def ids(pipeline, query):
        return [
            r.chunk.id
            for r in pipeline.retrieve(query, top_k=PER_QUERY_DEPTH, candidate_k=CANDIDATE_K)
        ]

    records = []
    for case in case_set:
        dirty = f"  {case.query.replace(' ', '  ')}  "
        rewritten = DeterministicQueryRewriter().rewrite(dirty)
        records.append(
            {
                "query": case.query,
                "denormalized_query": dirty,
                "retrieval_query": rewritten,
                "rewrite_changed_the_string": rewritten != dirty,
                "pipeline_ranking_identical": ids(pipeline_b, dirty) == ids(pipeline_r, dirty),
                "dense_top_5_dirty": [r.chunk.id for r in dense.retrieve(dirty, top_k=5)],
                "dense_top_5_normalized": [
                    r.chunk.id for r in dense.retrieve(rewritten, top_k=5)
                ],
                "lexical_top_5_dirty": [r.chunk.id for r in lexical.retrieve(dirty, top_k=5)],
                "lexical_top_5_normalized": [
                    r.chunk.id for r in lexical.retrieve(rewritten, top_k=5)
                ],
            }
        )
    return {
        "note": (
            "Diagnostic only: de-normalized queries derived from the benchmark "
            "queries, never used for any metric and never replacing a dataset "
            "query. Shows what the rewriter does when it actually fires, and "
            "where a whitespace difference is visible at all."
        ),
        "queries": records,
        "probes": len(records),
        "probes_where_rewrite_changed_the_string": sum(
            1 for r in records if r["rewrite_changed_the_string"]
        ),
        "probes_where_pipeline_ranking_changed": sum(
            1 for r in records if not r["pipeline_ranking_identical"]
        ),
        "probes_where_lexical_ranking_changed": sum(
            1 for r in records if r["lexical_top_5_dirty"] != r["lexical_top_5_normalized"]
        ),
        "probes_where_dense_ranking_changed": sum(
            1 for r in records if r["dense_top_5_dirty"] != r["dense_top_5_normalized"]
        ),
    }


def _build_rows(results: dict, config_id: str, fusion: str) -> list:
    """Flat evidence rows: one per (variant, depth), metrics verbatim."""
    rows = []
    for (top_k, variant), result in results.items():
        metrics = result.to_dict()
        row = {
            "id": f"{config_id}-{variant}-t{top_k}",
            "variant": variant,
            "retriever": "hybrid",
            "fusion": fusion,
            "reranker": "deterministic",
            "candidate_k": CANDIDATE_K,
            "top_k": top_k,
        }
        for field in METRIC_FIELDS:
            row[field] = metrics[field]
        row["deterministic"] = True
        rows.append(row)
    return rows


def _check_q1_against_phase10b(result) -> None:
    """Cross-phase guard: this path's baseline @33/33 == the frozen Phase 10B row."""
    frozen = json.loads(BASELINE_PATH.read_text(encoding="utf-8"))
    got = result.to_dict()
    mismatches = [f for f in RESULT_FIELDS if got[f] != frozen[f]]
    if mismatches:
        raise RuntimeError(
            "BLOCKED -- Phase 10B cross-phase mismatch on: "
            + ", ".join(f"{f} ({got[f]} != {frozen[f]})" for f in mismatches)
        )


def main(out_path: str | Path = DEFAULT_EVIDENCE_PATH) -> dict:
    """Run both deterministic scenarios for both variants and write evidence."""
    output = Path(out_path)

    documents = _baseline.ingest_corpus()
    _baseline.verify_corpus(documents)
    case_set = _load_cases()

    rows: list[dict] = []
    per_query: list[dict] = []
    q1_baseline = None
    changed_queries = 0
    variants_identical = True

    for config in CONFIGS:
        fusion = config["fusion"]
        pipe_b, results_b = _run_variant(
            documents, case_set, fusion=fusion, rewriter=None, depths=DEPTHS
        )
        pipe_r, results_r = _run_variant(
            documents,
            case_set,
            fusion=fusion,
            rewriter=DeterministicQueryRewriter(),
            depths=DEPTHS,
        )
        _assert_only_variable_differs(pipe_b, pipe_r)

        # The experiment's whole point: same metrics for both variants?
        for top_k in DEPTHS:
            a = results_b[(top_k, BASELINE_VARIANT)]
            b = results_r[(top_k, REWRITTEN_VARIANT)]
            if a.to_dict() != b.to_dict():
                variants_identical = False

        # Cross-instance determinism: a fully fresh build must agree.
        fresh_documents = _baseline.ingest_corpus()
        _baseline.verify_corpus(fresh_documents)
        fresh_cases = _load_cases()
        _, fresh_b = _run_variant(
            fresh_documents, fresh_cases, fusion=fusion, rewriter=None, depths=DEPTHS
        )
        _, fresh_r = _run_variant(
            fresh_documents,
            fresh_cases,
            fusion=fusion,
            rewriter=DeterministicQueryRewriter(),
            depths=DEPTHS,
        )
        for top_k in DEPTHS:
            _assert_equal(
                f"cross-instance {fusion} {BASELINE_VARIANT} top_k={top_k}",
                results_b[(top_k, BASELINE_VARIANT)],
                fresh_b[(top_k, BASELINE_VARIANT)],
            )
            _assert_equal(
                f"cross-instance {fusion} {REWRITTEN_VARIANT} top_k={top_k}",
                results_r[(top_k, REWRITTEN_VARIANT)],
                fresh_r[(top_k, REWRITTEN_VARIANT)],
            )

        for case in case_set:
            record = _per_query_record(pipe_b, pipe_r, case)
            record["config"] = config["id"]
            if not record["rewrite_identical"]:
                changed_queries += 1
            per_query.append(record)

        rows.extend(_build_rows(results_b, config["id"], fusion))
        rows.extend(_build_rows(results_r, config["id"], fusion))
        if q1_baseline is None:
            q1_baseline = results_b[(PER_QUERY_DEPTH, BASELINE_VARIANT)]

    _check_q1_against_phase10b(q1_baseline)

    evidence = {
        "experiment": EXPERIMENT_ID,
        "reference": REFERENCE_EXPERIMENT,
        "dataset": DATASET_ID,
        "rewriter": REWRITER_ID,
        "only_variable": "query_rewriter",
        "corpus_documents": EXPECTED_DOCUMENTS,
        "corpus_chunks": EXPECTED_TOTAL_CHUNKS,
        "candidate_k": CANDIDATE_K,
        "total_cases": EXPECTED_CASES,
        "evaluated_cases": EXPECTED_CASES,
        "queries_changed_by_rewriter": changed_queries,
        "variants_identical": variants_identical,
        "deterministic": True,
        "results": rows,
        "per_query": per_query,
        "mechanism_probe": _mechanism_probe(documents, case_set),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(evidence, indent=2) + "\n", encoding="utf-8")
    return evidence


def main_semantic(out_path: str | Path = DEFAULT_SEMANTIC_PATH) -> dict:
    """Opt-in Phase 12 scenario: Hybrid + CrossEncoder, baseline vs rewritten.

    Requires the ``semantic`` extra and a model cache (or network); the default
    run never imports it. Same corpus / dataset / candidate pool (33) and the
    same locked rerank depth (``top_k=8``) as the frozen Phase 12 profile -- the
    rewriter is again the only variable.
    """
    from zmrag.semantic import SemanticEmbeddingProvider
    from zmrag.semantic_rerank import CrossEncoderReranker

    output = Path(out_path)
    documents = _baseline.ingest_corpus()
    _baseline.verify_corpus(documents)
    case_set = _load_cases()

    # Read-only: the frozen Phase 12 evidence fixes the models and the row this
    # run's baseline must reproduce. It is never regenerated or overwritten.
    frozen = json.loads(RERANK_PATH.read_text(encoding="utf-8"))
    r2 = next(c for c in frozen["configs"] if c["id"] == "R2")
    if (r2["retriever"], r2["fusion"], r2["reranker"]) != (
        "hybrid",
        SEMANTIC_CONFIG["fusion"],
        "cross_encoder",
    ):
        raise RuntimeError(f"frozen Phase 12 R2 configuration is unexpected: {r2!r}")
    if (frozen["embedding_model_id"], frozen["reranker_model_id"]) != (
        EMBEDDING_MODEL_ID,
        RERANKER_MODEL_ID,
    ):
        raise RuntimeError("frozen Phase 12 model ids differ from the locked ones")

    def run(rewriter):
        embedder = SemanticEmbeddingProvider(model_name=EMBEDDING_MODEL_ID)
        vector_store = InMemoryVectorStore()
        lexical_index = InMemoryLexicalIndex()
        hybrid = HybridRetriever(
            DenseRetriever(embedder, vector_store),
            LexicalRetriever(lexical_index),
            _fusion(SEMANTIC_CONFIG["fusion"]),
        )
        pipeline = DeterministicRAGPipeline(
            chunker=FixedSizeChunker(chunk_size=CHUNK_SIZE, overlap=OVERLAP),
            embedder=embedder,
            vector_store=vector_store,
            lexical_index=lexical_index,
            retriever=hybrid,
            reranker=CrossEncoderReranker(model_name=RERANKER_MODEL_ID),
            context_builder=DeterministicContextBuilder(),
            query_rewriter=rewriter,
        )
        for document in documents:
            pipeline.ingest(document)
        runner = RetrievalEvaluationRunner(
            pipeline=pipeline, harness=EvaluationHarness(RetrievalEvaluator())
        )
        return DeterministicBenchmarkRunner(evaluation_runner=runner).run(
            case_set, top_k=SEMANTIC_CONFIG["top_k"], candidate_k=CANDIDATE_K
        )

    def row_for(variant, result):
        metrics = result.to_dict()
        row = {
            "id": f"{SEMANTIC_CONFIG['id']}-{variant}",
            "variant": variant,
            "retriever": "hybrid",
            "fusion": SEMANTIC_CONFIG["fusion"],
            "reranker": "cross_encoder",
            "candidate_k": CANDIDATE_K,
            "top_k": SEMANTIC_CONFIG["top_k"],
        }
        for field in METRIC_FIELDS:
            row[field] = metrics[field]
        row["deterministic"] = False
        return row

    baseline_result = run(None)
    rewritten_result = run(DeterministicQueryRewriter())
    baseline_row = row_for(BASELINE_VARIANT, baseline_result)
    rewritten_row = row_for(REWRITTEN_VARIANT, rewritten_result)
    rows = [baseline_row, rewritten_row]
    changed = sum(
        1
        for case in case_set
        if DeterministicQueryRewriter().rewrite(case.query) != case.query
    )
    metric_identical = all(
        baseline_row[f] == rewritten_row[f] for f in METRIC_FIELDS
    )

    # Cross-phase guard: the baseline run must reproduce the frozen Phase 12 R2 row.
    got = baseline_result.to_dict()
    mismatches = [f for f in RESULT_FIELDS if got[f] != r2[f]]
    if mismatches:
        raise RuntimeError(
            "BLOCKED -- Phase 12 cross-phase mismatch on: "
            + ", ".join(f"{f} ({got[f]} != {r2[f]})" for f in mismatches)
        )

    evidence = {
        "experiment": SEMANTIC_EXPERIMENT_ID,
        "reference": SEMANTIC_REFERENCE_EXPERIMENT,
        "dataset": DATASET_ID,
        "rewriter": REWRITER_ID,
        "only_variable": "query_rewriter",
        "embedding_model_id": frozen["embedding_model_id"],
        "reranker_model_id": frozen["reranker_model_id"],
        "corpus_documents": EXPECTED_DOCUMENTS,
        "corpus_chunks": EXPECTED_TOTAL_CHUNKS,
        "candidate_k": CANDIDATE_K,
        "top_k": SEMANTIC_CONFIG["top_k"],
        "total_cases": EXPECTED_CASES,
        "evaluated_cases": EXPECTED_CASES,
        "queries_changed_by_rewriter": changed,
        "variants_identical": metric_identical,
        "deterministic": False,
        "results": rows,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(evidence, indent=2) + "\n", encoding="utf-8")
    return evidence


if __name__ == "__main__":
    args = sys.argv[1:]
    flags = {a for a in args if a.startswith("--")}
    paths = [a for a in args if not a.startswith("--")]
    if flags - {"--semantic"}:
        raise SystemExit(f"unknown flag(s): {sorted(flags - {'--semantic'})}")
    if "--semantic" in flags:
        target = paths[0] if paths else DEFAULT_SEMANTIC_PATH
        evidence = main_semantic(target)
    else:
        target = paths[0] if paths else DEFAULT_EVIDENCE_PATH
        evidence = main(target)
    print(f"Evidence written to: {target}")
    print(json.dumps(evidence, indent=2))
