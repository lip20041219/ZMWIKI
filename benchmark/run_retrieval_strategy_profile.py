"""Retrieval strategy profile: Dense vs Hybrid vs Hybrid + Reranker, E2E.

One controlled, reproducible experiment over the frozen Phase 10A corpus and
dataset, answering what the retrieval *strategy* is worth -- not by adding a
retriever, but by holding everything else still and varying only the strategy.

The locked matrix (fixed here, never tuned against results):

    T1  Dense                                     -> top_k
    T2  Hybrid + Weighted fusion                  -> top_k
    T3  Hybrid + RRF fusion                       -> top_k
    T4  Hybrid + Weighted -> candidate_k -> DeterministicReranker -> top_k
    T5  Hybrid + RRF      -> candidate_k -> DeterministicReranker -> top_k
    T6  Hybrid + semantic embeddings + CrossEncoder (opt-in, --semantic)

Arms, all over the same corpus / chunking / embedding / queries / ground truth:

- **headline** (``top_k=5``, ``candidate_k=33``) -- the strategy table. Depths
  1 / 3 / 5 are run so ``HitRate@1``, ``Recall@1`` and ``Recall@3`` come from the
  evaluator too: every metric is read from ``EvaluationResult.to_dict()`` over
  results the strategy really returned. No metric is re-implemented here, and
  every strategy is scored at the *same* depth, so the columns are comparable.
- **candidate pool** -- ``candidate_k`` in 5 / 10 / 20 / 33 at ``top_k=5`` for
  both reranked strategies: does a wider recall pool give the reranker more
  correct candidates to reorder?
- **full pool** (``top_k=candidate_k=33``) -- the same five strategies at the
  *frozen* Phase 10B / 10C configuration, used as a cross-phase guard (below)
  and to report full-depth ranking quality.
- **semantic** (opt-in) -- Phase 12's locked two-stage configuration, kept in
  its own evidence file so the default run stays model-free and byte-stable.

Fairness is checked, not assumed: every strategy is built from scratch by one
builder, over the same six documents and the same chunker, and
``_assert_only_variable_differs`` compares the full configuration (embedder,
tokenizer, stores, retriever, fusion, reranker, cutoffs) and fails unless the
only differing keys are the ones the matrix means to vary. Each strategy's
stores are additionally read back and compared chunk by chunk against the
expected corpus, so "same corpus" is a measurement rather than a claim.

Historical evidence is read-only and guarded: the full-pool arm must reproduce
the frozen Phase 10B row and the frozen Phase 10C E1/E3/E4/E5/E6 rows field by
field. On any mismatch the run stops without writing evidence; on a match the
new evidence records ``historical_baseline_match: true``. The old evidence files
are never written to.

Run from the repository root:

    python benchmark/run_retrieval_strategy_profile.py
    HF_HUB_OFFLINE=1 python benchmark/run_retrieval_strategy_profile.py --semantic

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
    DeterministicEmbedder,
    DeterministicReranker,
    EvaluationHarness,
    EvaluationPrediction,
    FixedSizeChunker,
    HybridRetriever,
    InMemoryLexicalIndex,
    InMemoryVectorStore,
    JsonEvaluationDatasetLoader,
    LexicalRetriever,
    RetrievalEvaluator,
    RRFFusion,
    WeightedScoreFusion,
)

#: Sibling frozen runner: the canonical corpus ingestion + boundary checks.
#: Reused as-is -- never modified, never re-implemented.
import run_retrieval_small_baseline as _baseline

#: Repo root, resolved from this file so the script runs from any cwd.
ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = ROOT / "benchmark" / "datasets" / "retrieval_small.json"
EXPERIMENTS_PATH = ROOT / "benchmark" / "results" / "retrieval_small_experiments.json"
BASELINE_PATH = ROOT / "benchmark" / "results" / "retrieval_small_baseline.json"
RERANK_PATH = ROOT / "benchmark" / "results" / "retrieval_semantic_rerank_profile.json"
DEFAULT_EVIDENCE_PATH = ROOT / "benchmark" / "results" / "retrieval_strategy_comparison.json"
DEFAULT_SEMANTIC_PATH = (
    ROOT / "benchmark" / "results" / "retrieval_strategy_semantic_profile.json"
)

#: Stable identifiers used in the evidence (never an absolute path).
DATASET_ID = "benchmark/datasets/retrieval_small.json"
EXPERIMENT_ID = "retrieval_strategy_comparison"
SEMANTIC_EXPERIMENT_ID = "retrieval_strategy_semantic_profile"
REFERENCE_EXPERIMENT = "retrieval_small_baseline"

#: Fixed experiment environment -- identical to Phase 10B / 10C / 10D.
CHUNK_SIZE = 120
OVERLAP = 20
EXPECTED_DOCUMENTS = 6
EXPECTED_TOTAL_CHUNKS = 33
EXPECTED_CASES = 8
CHUNKING_ID = f"FixedSizeChunker(chunk_size={CHUNK_SIZE}, overlap={OVERLAP})"
EMBEDDING_ID = "DeterministicEmbedder(dimension=8)"

#: Serving depth of the headline table, and the recall pool feeding it.
TOP_K = 5
CANDIDATE_K = 33

#: Depths whose evaluator run supplies the @K columns of the headline table.
HEADLINE_DEPTHS = (1, 3, TOP_K)

#: The candidate-pool arm: how wide a recall pool the reranker is handed.
CANDIDATE_K_MATRIX = (5, 10, 20, 33)

#: The locked strategy matrix. ``candidate_k`` is ``None`` exactly where the
#: strategy has no recall pool (single-source retrieval -- Phase 10D's rule).
STRATEGIES = (
    {"id": "T1", "name": "dense", "retriever": "dense", "fusion": None, "reranker": None},
    {"id": "T2", "name": "hybrid_weighted", "retriever": "hybrid", "fusion": "weighted", "reranker": None},
    {"id": "T3", "name": "hybrid_rrf", "retriever": "hybrid", "fusion": "rrf", "reranker": None},
    {"id": "T4", "name": "hybrid_weighted_reranked", "retriever": "hybrid", "fusion": "weighted", "reranker": "deterministic"},
    {"id": "T5", "name": "hybrid_rrf_reranked", "retriever": "hybrid", "fusion": "rrf", "reranker": "deterministic"},
)

#: The strategy the per-query analysis compares everything else against.
BASELINE_STRATEGY_ID = "T1"

#: Strategies whose recall pool is varied in the candidate-pool arm.
RERANKED_STRATEGY_IDS = ("T4", "T5")

#: EvaluationResult.to_dict keys every cross-phase comparison uses.
METRIC_FIELDS = ("hit_rate", "recall", "precision", "mrr", "map", "ndcg")

#: The headline table's fields, each read from the evaluator at its own depth.
HEADLINE_FIELDS = (
    "hit_rate_at_1",
    "recall_at_1",
    "recall_at_3",
    "recall_at_5",
    "precision_at_5",
    "mrr_at_5",
    "map_at_5",
    "ndcg_at_5",
)

#: Cross-phase guards: a full-pool strategy must reproduce a frozen row exactly.
#: ``config_id`` is the row id inside that frozen file; ``None`` means the file
#: is a flat single row (Phase 10B).
HISTORICAL_GUARDS = (
    {"strategy": "T1", "file": "retrieval_small_experiments.json", "config_id": "E1"},
    {"strategy": "T2", "file": "retrieval_small_experiments.json", "config_id": "E3"},
    {"strategy": "T3", "file": "retrieval_small_experiments.json", "config_id": "E4"},
    {"strategy": "T4", "file": "retrieval_small_experiments.json", "config_id": "E5"},
    {"strategy": "T5", "file": "retrieval_small_experiments.json", "config_id": "E6"},
    {"strategy": "T4", "file": "retrieval_small_baseline.json", "config_id": None},
)

#: Phase 12's locked two-stage configuration, reused verbatim for the opt-in
#: arm: recall 33 -> CrossEncoder -> 8.
SEMANTIC_STRATEGY = {
    "id": "T6",
    "name": "hybrid_weighted_semantic_cross_encoder",
    "retriever": "hybrid",
    "fusion": "weighted",
    "reranker": "cross_encoder",
}
SEMANTIC_CANDIDATE_K = 33
SEMANTIC_TOP_K = 8
EMBEDDING_MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
RERANKER_MODEL_ID = "cross-encoder/ms-marco-MiniLM-L-6-v2"


def _fusion(name: str | None):
    if name is None:
        return None
    if name == "weighted":
        return WeightedScoreFusion()
    if name == "rrf":
        return RRFFusion()  # default rrf_k = 60 -- never tuned
    raise ValueError(f"unknown fusion: {name!r}")


class StrategyPath:
    """One retrieval strategy: its own stores, its own retriever, optional rerank.

    Built from scratch every time -- nothing is shared between instances, so a
    strategy can never observe another strategy's index. The composition mirrors
    the frozen Phase 10D ``RetrievalPath`` (dense / lexical / hybrid + fusion)
    but is kept local because the opt-in semantic arm has to swap the embedder,
    which that class fixes to the deterministic one.
    """

    def __init__(self, strategy: dict, *, reranker=None, embedder=None) -> None:
        self.strategy = dict(strategy)
        self.retriever = strategy["retriever"]
        self.fusion = strategy["fusion"]
        self._reranker = reranker
        self._embedder = embedder if embedder is not None else DeterministicEmbedder()
        self.vector_store = InMemoryVectorStore()
        self.lexical_index = InMemoryLexicalIndex()
        self._dense = DenseRetriever(self._embedder, self.vector_store)
        self._lexical = LexicalRetriever(self.lexical_index)
        self._hybrid = None
        if self.retriever == "hybrid":
            self._hybrid = HybridRetriever(self._dense, self._lexical, _fusion(self.fusion))

    def ingest(self, documents: list) -> None:
        """Chunk + embed + index the corpus into this strategy's own stores."""
        chunker = FixedSizeChunker(chunk_size=CHUNK_SIZE, overlap=OVERLAP)
        for document in documents:
            for chunk in chunker.chunk(document):
                self.vector_store.upsert(chunk, self._embedder.embed(chunk.text))
                self.lexical_index.upsert(chunk)

    def retrieve(self, query: str, *, top_k: int, candidate_k: int | None) -> list:
        """Ordered ``SearchResult`` list for one query under this strategy.

        Single-source retrieval is asked for ``top_k`` directly (its API has no
        recall pool); a fused strategy recalls a ``candidate_k``-wide pool and
        fuses it. A reranker -- when the strategy has one -- then re-scores that
        whole pool and cuts to ``top_k``: the pool is never pre-truncated to
        ``top_k`` first, or there would be no reranking stage left to measure.
        """
        if self.retriever == "dense":
            pool = self._dense.retrieve(query, top_k=top_k)
        else:
            pool = self._hybrid.retrieve(query, top_k=candidate_k, candidate_k=candidate_k)
        if self._reranker is None:
            return pool[:top_k]
        return self._reranker.rerank(query, pool, top_k=top_k)


def build_strategy(strategy: dict, *, embedder=None) -> StrategyPath:
    """Build one strategy (reranker included when it names one) and ingest."""
    reranker = None
    if strategy["reranker"] == "deterministic":
        reranker = DeterministicReranker()
    path = StrategyPath(strategy, reranker=reranker, embedder=embedder)
    path.ingest(_baseline.ingest_corpus())
    return path


def _load_cases():
    case_set = JsonEvaluationDatasetLoader().load(DATASET_PATH)
    if len(case_set) != EXPECTED_CASES:
        raise RuntimeError(f"dataset has {len(case_set)} cases, expected {EXPECTED_CASES}")
    return case_set


def _expected_chunks(documents: list) -> list:
    """The corpus as the frozen chunker cuts it -- what every store must hold."""
    chunker = FixedSizeChunker(chunk_size=CHUNK_SIZE, overlap=OVERLAP)
    chunks = []
    for document in documents:
        chunks.extend(chunker.chunk(document))
    if len(chunks) != EXPECTED_TOTAL_CHUNKS:
        raise RuntimeError(
            f"corpus chunks {len(chunks)} != frozen {EXPECTED_TOTAL_CHUNKS}"
        )
    return chunks


def _verify_stores(path: StrategyPath, expected: list) -> tuple:
    """Read each expected chunk back out of the strategy's own stores.

    This is the fairness anchor: two strategies are only comparable if they were
    built over the same chunks, so it is measured per strategy rather than
    assumed once for all of them.
    """
    if path.vector_store.count() != EXPECTED_TOTAL_CHUNKS:
        raise RuntimeError(f"vector store holds {path.vector_store.count()} chunks")
    if path.lexical_index.count() != EXPECTED_TOTAL_CHUNKS:
        raise RuntimeError(f"lexical index holds {path.lexical_index.count()} chunks")
    fingerprint = []
    for chunk in expected:
        stored = path.vector_store.get(chunk.id)
        if stored is None:
            raise RuntimeError(f"chunk {chunk.id!r} missing from the vector store")
        stored_chunk, _embedding = stored
        if stored_chunk.text != chunk.text or stored_chunk.position != chunk.position:
            raise RuntimeError(f"chunk {chunk.id!r} differs from the frozen corpus")
        if path.lexical_index.get(chunk.id) is None:
            raise RuntimeError(f"chunk {chunk.id!r} missing from the lexical index")
        fingerprint.append((chunk.id, chunk.position, chunk.text))
    return tuple(fingerprint)


def _config_of(path: StrategyPath, *, top_k: int, candidate_k: int | None) -> dict:
    """Everything that must be held still, plus the one thing being varied."""
    return {
        "embedder": type(path._embedder).__name__,
        "embedding_dimension": path._embedder.dimension,
        "bm25_tokenizer": type(path.lexical_index.tokenizer).__name__,
        "vector_store": type(path.vector_store).__name__,
        "lexical_index": type(path.lexical_index).__name__,
        "retriever": path.retriever,
        "fusion": None if path._hybrid is None else type(path._hybrid.fusion).__name__,
        "reranker": None if path._reranker is None else type(path._reranker).__name__,
        "top_k": top_k,
        "candidate_k": candidate_k,
    }


def _assert_only_variable_differs(
    reference: StrategyPath,
    other: StrategyPath,
    *,
    reference_args: dict,
    other_args: dict,
    allowed: tuple[str, ...],
) -> dict:
    """Fail unless ``other`` differs from ``reference`` only in ``allowed`` keys."""
    left = _config_of(reference, **reference_args)
    right = _config_of(other, **other_args)
    differing = tuple(key for key in left if left[key] != right[key])
    unexpected = tuple(key for key in differing if key not in allowed)
    if unexpected:
        raise RuntimeError(
            "BLOCKED -- strategy comparison is not controlled; unexpected "
            f"difference(s) in {unexpected}: "
            + "; ".join(f"{key}: {left[key]!r} != {right[key]!r}" for key in unexpected)
        )
    return {"differing_keys": list(differing), "allowed_keys": list(allowed)}


def _run(path: StrategyPath, case_set, *, top_k: int, candidate_k: int | None):
    """One real evaluation: the strategy's ranked ids plus the evaluator's metrics."""
    predictions = []
    retrieved: list[list[str]] = []
    for case in case_set:
        results = path.retrieve(case.query, top_k=top_k, candidate_k=candidate_k)
        retrieved.append([result.chunk.id for result in results])
        predictions.append(EvaluationPrediction(query=case.query, results=results))
    harness = EvaluationHarness(RetrievalEvaluator())
    return harness.run(list(case_set.cases), predictions), retrieved


def _headline(path: StrategyPath, case_set, *, candidate_k: int | None):
    """The @K columns, each read from the evaluator run at that depth."""
    at_depth = {}
    retrieved = None
    for depth in HEADLINE_DEPTHS:
        result, ids = _run(path, case_set, top_k=depth, candidate_k=candidate_k)
        at_depth[depth] = result.to_dict()
        if depth == TOP_K:
            retrieved = ids
    one, three, five = at_depth[1], at_depth[3], at_depth[TOP_K]
    return (
        {
            "hit_rate_at_1": one["hit_rate"],
            "recall_at_1": one["recall"],
            "recall_at_3": three["recall"],
            "recall_at_5": five["recall"],
            "precision_at_5": five["precision"],
            "mrr_at_5": five["mrr"],
            "map_at_5": five["map"],
            "ndcg_at_5": five["ndcg"],
        },
        retrieved,
    )


def _strategy_row(strategy: dict, candidate_k: int | None, headline: dict, full_pool: dict) -> dict:
    """One evidence row: the varied configuration and every real metric."""
    row = {
        "id": strategy["id"],
        "name": strategy["name"],
        "retriever": strategy["retriever"],
        "fusion": strategy["fusion"],
        "reranker": strategy["reranker"],
        "candidate_k": candidate_k,
        "top_k": TOP_K,
    }
    row.update({field: headline[field] for field in HEADLINE_FIELDS})
    row["full_pool_candidate_k"] = CANDIDATE_K
    row["full_pool_top_k"] = CANDIDATE_K
    for field in METRIC_FIELDS:
        row[f"full_pool_{field}"] = full_pool[field]
    row["deterministic"] = True
    return row


def _per_query(rows: list, case_set, retrievals: dict) -> list:
    """Per-query evidence: what each strategy retrieved, and where it landed.

    ``rank_of_first_relevant`` and the ``hit@K`` flags are diagnostics read off
    the ranked ids the strategy returned -- not a metric implementation (nothing
    is averaged or scored here). Every number in the strategy table comes from
    the evaluator.
    """
    records = []
    for index, case in enumerate(case_set):
        relevant = set(case.relevant_chunk_ids)
        per_strategy = {}
        for row in rows:
            ids = retrievals[row["id"]][index]
            per_strategy[row["id"]] = {
                "retrieved_chunk_ids": ids,
                "rank_of_first_relevant": next(
                    (i for i, cid in enumerate(ids, start=1) if cid in relevant), 0
                ),
                "hit_at_1": any(cid in relevant for cid in ids[:1]),
                "hit_at_3": any(cid in relevant for cid in ids[:3]),
                "hit_at_5": any(cid in relevant for cid in ids[:5]),
            }
        records.append(
            {
                "query": case.query,
                "relevant_chunk_ids": list(case.relevant_chunk_ids),
                "strategies": per_strategy,
            }
        )
    return records


def _rank_changes(rows: list, retrievals: dict) -> list:
    """How many queries each strategy ranks differently from the dense baseline."""
    baseline = retrievals[BASELINE_STRATEGY_ID]
    changes = []
    for row in rows:
        ids = retrievals[row["id"]]
        changes.append(
            {
                "id": row["id"],
                "queries_with_changed_ranking": sum(
                    1 for a, b in zip(baseline, ids) if a != b
                ),
                "queries_with_changed_top_1": sum(
                    1 for a, b in zip(baseline, ids) if a[:1] != b[:1]
                ),
            }
        )
    return changes


def _candidate_pool_arm(case_set) -> list:
    """Would a wider recall pool give the reranker more correct candidates?"""
    arm = []
    for strategy in STRATEGIES:
        if strategy["id"] not in RERANKED_STRATEGY_IDS:
            continue
        for candidate_k in CANDIDATE_K_MATRIX:
            path = build_strategy(strategy)
            headline, _ = _headline(path, case_set, candidate_k=candidate_k)
            arm.append(
                {
                    "id": strategy["id"],
                    "fusion": strategy["fusion"],
                    "reranker": strategy["reranker"],
                    "top_k": TOP_K,
                    "candidate_k": candidate_k,
                    "recall_at_5": headline["recall_at_5"],
                    "mrr_at_5": headline["mrr_at_5"],
                    "ndcg_at_5": headline["ndcg_at_5"],
                }
            )
    return arm


def _frozen_row(entry: dict) -> dict:
    """Read one frozen historical row (read-only)."""
    if entry["config_id"] is None:
        return json.loads(BASELINE_PATH.read_text(encoding="utf-8"))
    frozen = json.loads(EXPERIMENTS_PATH.read_text(encoding="utf-8"))
    return next(row for row in frozen["configs"] if row["id"] == entry["config_id"])


def _historical_guard(full_pool: dict) -> list:
    """Compare the full-pool arm to the frozen evidence, field by field.

    Stops the run (no evidence written) on any mismatch: this experiment must
    not quietly disagree with Phase 10B / 10C about a configuration they share.
    """
    comparisons = []
    for entry in HISTORICAL_GUARDS:
        frozen = _frozen_row(entry)
        got = full_pool[entry["strategy"]].to_dict()
        mismatches = [field for field in METRIC_FIELDS if got[field] != frozen[field]]
        source = (
            "retrieval_small_baseline"
            if entry["config_id"] is None
            else f"retrieval_small_experiments:{entry['config_id']}"
        )
        if mismatches:
            raise RuntimeError(
                "BLOCKED -- historical baseline mismatch vs "
                f"{source} for {entry['strategy']}: "
                + ", ".join(f"{f} ({got[f]} != {frozen[f]})" for f in mismatches)
            )
        comparisons.append(
            {
                "strategy": entry["strategy"],
                "reference": source,
                "fields_compared": list(METRIC_FIELDS),
                "match": True,
            }
        )
    return comparisons


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


def _dataset_fingerprint(case_set) -> dict:
    return {
        "queries": [case.query for case in case_set],
        "relevant_chunk_ids": [list(case.relevant_chunk_ids) for case in case_set],
    }


def main(out_path: str | Path = DEFAULT_EVIDENCE_PATH) -> dict:
    """Run the strategy matrix + candidate-pool arm, verify, write evidence."""
    output = Path(out_path)

    documents = _baseline.ingest_corpus()
    _baseline.verify_corpus(documents)
    case_set = _load_cases()
    expected = _expected_chunks(documents)

    paths = {}
    fingerprints = set()
    for strategy in STRATEGIES:
        path = build_strategy(strategy)
        fingerprints.add(_verify_stores(path, expected))
        paths[strategy["id"]] = path
    if len(fingerprints) != 1:
        raise RuntimeError("strategies were not built over the same corpus")

    rows = []
    retrievals = {}
    full_pool = {}
    fairness = []
    for strategy in STRATEGIES:
        sid = strategy["id"]
        candidate_k = None if strategy["retriever"] == "dense" else CANDIDATE_K
        path = paths[sid]

        # Against the dense baseline, only the strategy itself may differ --
        # except that a single-source retriever has no recall pool to record.
        fairness.append(
            {
                "id": sid,
                "compared_against": BASELINE_STRATEGY_ID,
                **_assert_only_variable_differs(
                    paths[BASELINE_STRATEGY_ID],
                    path,
                    reference_args={"top_k": TOP_K, "candidate_k": None},
                    other_args={"top_k": TOP_K, "candidate_k": candidate_k},
                    allowed=("retriever", "fusion", "reranker", "candidate_k"),
                ),
            }
        )

        headline, retrieved = _headline(path, case_set, candidate_k=candidate_k)
        again, _ = _run(path, case_set, top_k=TOP_K, candidate_k=candidate_k)
        first, _ = _run(path, case_set, top_k=TOP_K, candidate_k=candidate_k)
        _assert_equal(f"same-instance {sid}", first, again)

        # A fully fresh build must agree too (fresh ingestion, fresh stores).
        fresh = build_strategy(strategy)
        _verify_stores(fresh, expected)
        fresh_result, _ = _run(fresh, case_set, top_k=TOP_K, candidate_k=candidate_k)
        _assert_equal(f"cross-instance {sid}", first, fresh_result)

        full, _ = _run(path, case_set, top_k=CANDIDATE_K, candidate_k=CANDIDATE_K)
        full_again, _ = _run(path, case_set, top_k=CANDIDATE_K, candidate_k=CANDIDATE_K)
        _assert_equal(f"full-pool {sid}", full, full_again)

        retrievals[sid] = retrieved
        full_pool[sid] = full
        rows.append(_strategy_row(strategy, candidate_k, headline, full.to_dict()))

    comparisons = _historical_guard(full_pool)

    evidence = {
        "experiment": EXPERIMENT_ID,
        "reference": REFERENCE_EXPERIMENT,
        "dataset": DATASET_ID,
        "corpus_documents": EXPECTED_DOCUMENTS,
        "corpus_chunks": EXPECTED_TOTAL_CHUNKS,
        "chunking": CHUNKING_ID,
        "embedding": EMBEDDING_ID,
        "bm25_tokenizer": type(paths[BASELINE_STRATEGY_ID].lexical_index.tokenizer).__name__,
        "total_cases": EXPECTED_CASES,
        "evaluated_cases": EXPECTED_CASES,
        "top_k": TOP_K,
        "candidate_k": CANDIDATE_K,
        "dataset_fingerprint": _dataset_fingerprint(case_set),
        "fairness": fairness,
        "historical_baseline_match": all(c["match"] for c in comparisons),
        "historical_comparisons": comparisons,
        "strategies": rows,
        "rank_changes_vs_baseline": _rank_changes(rows, retrievals),
        "per_query": _per_query(rows, case_set, retrievals),
        "candidate_pool": _candidate_pool_arm(case_set),
        "deterministic": True,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(evidence, indent=2) + "\n", encoding="utf-8")
    return evidence


def main_semantic(out_path: str | Path = DEFAULT_SEMANTIC_PATH) -> dict:
    """Opt-in Phase 12 configuration: semantic embeddings + CrossEncoder rerank.

    Returns ``{"available": False, "reason": ...}`` and writes nothing when the
    optional extra or the local model cache is missing. The default run and the
    default test suite never reach this function.
    """
    output = Path(out_path)
    try:
        from zmrag.semantic import SemanticEmbeddingProvider
        from zmrag.semantic_rerank import CrossEncoderReranker

        embedder = SemanticEmbeddingProvider(model_name=EMBEDDING_MODEL_ID)
        reranker = CrossEncoderReranker(model_name=RERANKER_MODEL_ID)
    except (ImportError, OSError, RuntimeError, ValueError) as error:
        return {"available": False, "reason": f"{type(error).__name__}: {error}"}

    documents = _baseline.ingest_corpus()
    _baseline.verify_corpus(documents)
    case_set = _load_cases()
    expected = _expected_chunks(documents)

    path = build_strategy(SEMANTIC_STRATEGY, embedder=embedder)
    path._reranker = reranker  # the arm's whole point: a real reranking stage
    _verify_stores(path, expected)

    result, retrieved = _run(
        path, case_set, top_k=SEMANTIC_TOP_K, candidate_k=SEMANTIC_CANDIDATE_K
    )
    metrics = result.to_dict()

    # Cross-phase guard: the same locked configuration as Phase 12's R2 row.
    frozen = json.loads(RERANK_PATH.read_text(encoding="utf-8"))
    reference = next(c for c in frozen["configs"] if c["id"] == "R2")
    mismatches = [field for field in METRIC_FIELDS if metrics[field] != reference[field]]
    if mismatches:
        raise RuntimeError(
            "BLOCKED -- Phase 12 cross-phase mismatch on: "
            + ", ".join(f"{f} ({metrics[f]} != {reference[f]})" for f in mismatches)
        )

    row = {
        "id": SEMANTIC_STRATEGY["id"],
        "name": SEMANTIC_STRATEGY["name"],
        "retriever": SEMANTIC_STRATEGY["retriever"],
        "fusion": SEMANTIC_STRATEGY["fusion"],
        "reranker": SEMANTIC_STRATEGY["reranker"],
        "candidate_k": SEMANTIC_CANDIDATE_K,
        "top_k": SEMANTIC_TOP_K,
    }
    row.update({field: metrics[field] for field in METRIC_FIELDS})
    row["deterministic"] = False

    evidence = {
        "experiment": SEMANTIC_EXPERIMENT_ID,
        "reference": "retrieval_semantic_rerank_profile",
        "dataset": DATASET_ID,
        "corpus_documents": EXPECTED_DOCUMENTS,
        "corpus_chunks": EXPECTED_TOTAL_CHUNKS,
        "chunking": CHUNKING_ID,
        "embedding_model_id": frozen["embedding_model_id"],
        "reranker_model_id": frozen["reranker_model_id"],
        "total_cases": EXPECTED_CASES,
        "evaluated_cases": EXPECTED_CASES,
        "candidate_k": SEMANTIC_CANDIDATE_K,
        "top_k": SEMANTIC_TOP_K,
        "historical_baseline_match": True,
        "historical_comparisons": [
            {
                "strategy": SEMANTIC_STRATEGY["id"],
                "reference": "retrieval_semantic_rerank_profile:R2",
                "fields_compared": list(METRIC_FIELDS),
                "match": True,
            }
        ],
        "strategies": [row],
        "per_query": [
            {
                "query": case.query,
                "relevant_chunk_ids": list(case.relevant_chunk_ids),
                "retrieved_chunk_ids": retrieved[index],
            }
            for index, case in enumerate(case_set)
        ],
        "deterministic": False,
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
        if not evidence.get("available", True):
            print(f"semantic arm unavailable: {evidence['reason']}")
            raise SystemExit(0)
    else:
        target = paths[0] if paths else DEFAULT_EVIDENCE_PATH
        evidence = main(target)
    print(f"Evidence written to: {target}")
    print(json.dumps(evidence, indent=2))
