"""Answer evaluation profile: pipeline -> AnswerResult -> AnswerEvaluator.

The answer-side counterpart of the retrieval baseline. It runs the canonical
Phase 10B pipeline over an answer dataset, scores each answer with the
deterministic lexical baselines in ``zmrag.answer_evaluation``, and writes
per-query evidence plus aggregates.

    answer_small.json
       ↓  per case: Pipeline.answer_with_sources -> AnswerResult
       ↓             Pipeline.query             -> the same Context (checked)
       ↓  AnswerEvaluator.evaluate(reference_answer, context, sources)
       ↓  AnswerEvaluation[]
    aggregate + per-query evidence

What this harness is careful about:

- **It does not touch retrieval evaluation.** No retrieval metric is computed
  here; the one retrieval number it records, ``retrieval_hit_at_5``, is a
  boolean diagnostic over the ranked ids the pipeline returned, used only to
  join retrieval outcome to answer quality (see the report). The retrieval
  benchmark and its evidence are separate and unchanged.
- **It does not judge with a model.** The generator is the deterministic
  extractive one, so the whole profile is offline, dependency-free, and
  reproducible; there is no LLM judge and no opt-in model path.
- **It does not fabricate answers.** A case may carry ``answer_override`` (and
  ``citation_chunk_ids``) to construct a negative probe -- a wrong answer, an
  uncited answer -- that the real pipeline would never produce. Such rows are
  labelled ``answer_source: "override"`` in the evidence; every other row is
  the pipeline's own output.
- **It evaluates exactly the Context the answer was grounded in.** The Context
  used for scoring is re-fetched through ``query()`` and required to match the
  AnswerResult's own sources, so the citation contract is verified rather than
  assumed.

Run from the repository root:

    python benchmark/run_answer_evaluation.py

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
    AnswerEvaluator,
    Citation,
    DenseRetriever,
    DeterministicContextBuilder,
    DeterministicEmbedder,
    DeterministicGenerator,
    DeterministicRAGPipeline,
    DeterministicReranker,
    FixedSizeChunker,
    HybridRetriever,
    InMemoryLexicalIndex,
    InMemoryVectorStore,
    LexicalRetriever,
    WeightedScoreFusion,
    aggregate_evaluations,
    citations_from_context,
)

#: Sibling frozen runner: the canonical corpus ingestion + boundary checks.
import run_retrieval_small_baseline as _baseline

#: Repo root, resolved from this file so the script runs from any cwd.
ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = ROOT / "benchmark" / "datasets" / "answer_small.json"
DEFAULT_EVIDENCE_PATH = ROOT / "benchmark" / "results" / "answer_evaluation_profile.json"

#: Stable identifiers used in the evidence (never an absolute path).
DATASET_ID = "benchmark/datasets/answer_small.json"
EXPERIMENT_ID = "answer_evaluation_profile"

#: Fixed environment -- the same corpus, chunking and embedding as the frozen
#: retrieval benchmark, so the two profiles describe one system.
CHUNK_SIZE = 120
OVERLAP = 20
EXPECTED_DOCUMENTS = 6
EXPECTED_TOTAL_CHUNKS = 33
EXPECTED_CASES = 16
EMBEDDING_ID = "DeterministicEmbedder(dimension=8)"
GENERATOR_ID = "DeterministicGenerator"
EVALUATOR_ID = "AnswerEvaluator"
RETRIEVER_ID = "HybridRetriever(dense + bm25, WeightedScoreFusion)"
RERANKER_ID = "DeterministicReranker"
TOKENIZER_ID = "SimpleTokenizer"

#: Serving depth, and the recall pool feeding it -- identical to the retrieval
#: strategy benchmark's headline configuration.
TOP_K = 5
CANDIDATE_K = 33
MAX_CHARS = 2000

#: Required dataset fields per case; ``answer_override`` / ``citation_chunk_ids`` /
#: ``note`` are optional probe controls (see the dataset's own ``probe_controls``).
REQUIRED_CASE_FIELDS = ("id", "kind", "query", "reference_answer", "relevant_chunk_ids")

#: The aggregate keys, in their fixed order.
AGGREGATE_FIELDS = (
    "mean_reference_coverage",
    "mean_groundedness",
    "mean_citation_coverage",
    "answer_rate",
    "source_presence_rate",
    "mean_source_count",
)


def _load_cases() -> list[dict]:
    """Load and validate the answer dataset; stop on anything malformed."""
    payload = json.loads(DATASET_PATH.read_text(encoding="utf-8"))
    cases = payload.get("cases")
    if not isinstance(cases, list) or not cases:
        raise RuntimeError(f"{DATASET_ID} has no cases list")
    if len(cases) != EXPECTED_CASES:
        raise RuntimeError(f"dataset has {len(cases)} cases, expected {EXPECTED_CASES}")
    seen = set()
    for case in cases:
        if not isinstance(case, dict):
            raise RuntimeError(f"each case must be an object, got {type(case).__name__}")
        missing = [field for field in REQUIRED_CASE_FIELDS if field not in case]
        if missing:
            raise RuntimeError(f"case {case.get('id')!r} is missing {missing}")
        for field in REQUIRED_CASE_FIELDS:
            value = case[field]
            if isinstance(value, str) and not value.strip():
                raise RuntimeError(f"case {case['id']!r}: {field} must be non-empty")
        if not case["relevant_chunk_ids"]:
            raise RuntimeError(f"case {case['id']!r}: relevant_chunk_ids must not be empty")
        if case["id"] in seen:
            raise RuntimeError(f"duplicate case id: {case['id']!r}")
        seen.add(case["id"])
    return cases


def build_pipeline(documents: list) -> DeterministicRAGPipeline:
    """The canonical deterministic pipeline over the frozen corpus."""
    chunker = FixedSizeChunker(chunk_size=CHUNK_SIZE, overlap=OVERLAP)
    embedder = DeterministicEmbedder()
    vector_store = InMemoryVectorStore()
    lexical_index = InMemoryLexicalIndex()
    pipeline = DeterministicRAGPipeline(
        chunker=chunker,
        embedder=embedder,
        vector_store=vector_store,
        lexical_index=lexical_index,
        retriever=HybridRetriever(
            DenseRetriever(embedder, vector_store),
            LexicalRetriever(lexical_index),
            WeightedScoreFusion(),
        ),
        reranker=DeterministicReranker(),
        context_builder=DeterministicContextBuilder(),
        generator=DeterministicGenerator(),
    )
    for document in documents:
        pipeline.ingest(document)
    return pipeline


def _resolve_sources(case: dict, context, answer_result):
    """Which citations this case carries: the Context's own, or pinned ones.

    Absent ``citation_chunk_ids`` means "every chunk of the Context", which is
    what ``citations_from_context`` produces and what the pipeline really does.
    A present list -- including an empty one -- pins the citations to exactly
    those context chunks, in that order, so a probe can construct a wrong
    citation or an uncited answer that the pipeline would never produce. A
    pinned id that is not in the Context is a malformed probe and stops the run.
    """
    pinned = case.get("citation_chunk_ids")
    if pinned is None:
        return answer_result.sources, "context"
    if not isinstance(pinned, list):
        raise RuntimeError(f"case {case['id']!r}: citation_chunk_ids must be a list")
    by_id = {chunk.id: chunk for chunk in context.chunks}
    unknown = [cid for cid in pinned if cid not in by_id]
    if unknown:
        raise RuntimeError(
            f"case {case['id']!r}: citation_chunk_ids {unknown} are not in the "
            "retrieved Context"
        )
    return tuple(Citation.from_chunk(by_id[cid]) for cid in pinned), "pinned"


def _evaluate_case(case: dict, pipeline, evaluator: AnswerEvaluator):
    """One case: real retrieval, real (or overridden) answer, real scores.

    Returns the evidence row together with the :class:`AnswerEvaluation` it was
    built from, so the aggregate is computed from the very same objects the
    per-query evidence reports.
    """
    query = case["query"]
    kwargs = {"top_k": TOP_K, "candidate_k": CANDIDATE_K, "max_chars": MAX_CHARS}

    answer_result = pipeline.answer_with_sources(query, **kwargs)
    context = pipeline.query(query, **kwargs)
    # The Context being scored must be the one the answer was grounded in --
    # the citation contract promises exactly this, so it is checked, not assumed.
    if answer_result.sources != citations_from_context(context):
        raise RuntimeError(
            f"case {case['id']!r}: the scored Context is not the one the answer "
            "was grounded in"
        )

    override = case.get("answer_override")
    answer = override if override is not None else answer_result.answer
    sources, citation_source = _resolve_sources(case, context, answer_result)

    evaluation = evaluator.evaluate(
        query_id=case["id"],
        query=query,
        answer=answer,
        context=context,
        reference_answer=case["reference_answer"],
        sources=sources,
    )

    retrieved = pipeline.retrieve(query, top_k=TOP_K, candidate_k=CANDIDATE_K)
    retrieved_ids = [result.chunk.id for result in retrieved]
    relevant = set(case["relevant_chunk_ids"])

    row = {
        "id": case["id"],
        "kind": case["kind"],
        "query": query,
        "answer_source": "override" if override is not None else "pipeline",
        "citation_source": citation_source,
        "answer": answer,
        "reference_answer": case["reference_answer"],
        "relevant_chunk_ids": list(case["relevant_chunk_ids"]),
        "retrieved_chunk_ids": retrieved_ids,
        # Boolean diagnostic over the ranked ids, used only to join retrieval
        # outcome to answer quality -- not a retrieval metric.
        "retrieval_hit_at_5": any(cid in relevant for cid in retrieved_ids),
    }
    row.update(evaluation.to_dict())
    return row, evaluation


def _by_kind(rows: list[dict]) -> dict:
    """Per-kind counts and means, so the negative cases can be read directly."""
    grouped: dict[str, list[dict]] = {}
    for row in rows:
        grouped.setdefault(row["kind"], []).append(row)
    summary = {}
    for kind in sorted(grouped):
        group = grouped[kind]
        count = float(len(group))
        summary[kind] = {
            "cases": len(group),
            "mean_reference_coverage": sum(r["reference_coverage"] for r in group) / count,
            "mean_groundedness": sum(r["groundedness"] for r in group) / count,
            "mean_citation_coverage": sum(r["citation_coverage"] for r in group) / count,
            "answer_rate": sum(1 for r in group if r["has_answer"]) / count,
        }
    return summary


def _retrieval_join(rows: list[dict]) -> dict:
    """Retrieval outcome vs answer quality -- the two existing outputs, joined.

    Answers the question the retrieval benchmark cannot: when retrieval misses,
    does answer quality follow? Means over real pipeline rows only: a row whose
    answer was overridden carries a synthetic answer, so it cannot say what the
    pipeline would have produced and is excluded (its count is recorded below).
    No metric is recomputed here.
    """
    real = [row for row in rows if row["answer_source"] == "pipeline"]
    hit = [row for row in real if row["retrieval_hit_at_5"]]
    miss = [row for row in real if not row["retrieval_hit_at_5"]]

    def _mean(group, field):
        return sum(row[field] for row in group) / len(group) if group else 0.0

    return {
        "cases_considered": len(real),
        "cases_excluded_as_probes": len(rows) - len(real),
        "cases_with_retrieval_hit": len(hit),
        "cases_with_retrieval_miss": len(miss),
        "hit_mean_reference_coverage": _mean(hit, "reference_coverage"),
        "miss_mean_reference_coverage": _mean(miss, "reference_coverage"),
        "hit_mean_groundedness": _mean(hit, "groundedness"),
        "miss_mean_groundedness": _mean(miss, "groundedness"),
    }


def _run_profile() -> dict:
    """Run every case through a freshly built pipeline; nothing is cached."""
    documents = _baseline.ingest_corpus()
    _baseline.verify_corpus(documents)
    cases = _load_cases()
    pipeline = build_pipeline(documents)
    evaluator = AnswerEvaluator()

    evaluated = [_evaluate_case(case, pipeline, evaluator) for case in cases]
    rows = [row for row, _evaluation in evaluated]
    aggregate = aggregate_evaluations(evaluation for _row, evaluation in evaluated)
    if list(aggregate) != list(AGGREGATE_FIELDS):
        raise RuntimeError("aggregate keys drifted from the locked field list")

    return {
        "experiment": EXPERIMENT_ID,
        "dataset": DATASET_ID,
        "corpus_documents": EXPECTED_DOCUMENTS,
        "corpus_chunks": EXPECTED_TOTAL_CHUNKS,
        "chunking": f"FixedSizeChunker(chunk_size={CHUNK_SIZE}, overlap={OVERLAP})",
        "embedding": EMBEDDING_ID,
        "retriever": RETRIEVER_ID,
        "reranker": RERANKER_ID,
        "generator": GENERATOR_ID,
        "evaluator": EVALUATOR_ID,
        "tokenizer": TOKENIZER_ID,
        "groundedness_threshold": evaluator.groundedness_threshold,
        "top_k": TOP_K,
        "candidate_k": CANDIDATE_K,
        "total_cases": len(rows),
        "evaluated_cases": len(rows),
        "aggregate": aggregate,
        "by_kind": _by_kind(rows),
        "retrieval_answer_join": _retrieval_join(rows),
        "per_query": rows,
        "deterministic": True,
    }


def main(out_path: str | Path = DEFAULT_EVIDENCE_PATH) -> dict:
    """Run the profile twice on fresh pipelines and write the evidence."""
    output = Path(out_path)
    first = _run_profile()
    second = _run_profile()
    if first != second:
        raise RuntimeError(
            "answer evaluation profile is not deterministic: two fresh runs differ"
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(first, indent=2) + "\n", encoding="utf-8")
    return first


if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_EVIDENCE_PATH
    evidence = main(target)
    print(f"Evidence written to: {target}")
    print(json.dumps(evidence, indent=2))
