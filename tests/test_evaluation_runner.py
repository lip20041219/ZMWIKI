"""Retrieval evaluation runner tests: Pipeline -> Prediction -> Harness.

Covers the Phase 9A RetrievalEvaluationRunner: constructor injection and type
validation, exactly-once per-case ``pipeline.retrieve`` delegation with query /
``top_k`` / ``candidate_k`` passed verbatim, predictions built from the real
returned SearchResult list (order and objects preserved, never sorted or
fabricated), exactly-once ``harness.run(cases, predictions)`` delegation whose
result is returned unchanged (the runner computes no metrics and never builds an
EvaluationResult itself), empty-run passthrough to ``harness.run([], [])``,
input validation ordering, determinism, immutability, and a real-pipeline E2E.
"""

import pytest

from zmrag import (
    Chunk,
    DeterministicContextBuilder,
    DeterministicEmbedder,
    DeterministicRAGPipeline,
    DeterministicReranker,
    DenseRetriever,
    Document,
    EvaluationCase,
    EvaluationHarness,
    EvaluationPrediction,
    EvaluationResult,
    FixedSizeChunker,
    HybridRetriever,
    InMemoryLexicalIndex,
    InMemoryVectorStore,
    LexicalRetriever,
    RAGPipeline,
    RetrievalEvaluationRunner,
    RetrievalEvaluator,
    SearchResult,
    WeightedScoreFusion,
)


def _chunk(cid, text=None):
    return Chunk(id=cid, document_id="doc", text=text or cid, position=0)


def _sr(cid, score=0.0, text=None):
    return SearchResult(chunk=_chunk(cid, text=text), score=score)


def _case(query, *ids):
    return EvaluationCase(query=query, relevant_chunk_ids=ids)


def _harness():
    return EvaluationHarness(RetrievalEvaluator())


class FakePipeline(RAGPipeline):
    """RAGPipeline that records retrieve calls and returns scripted results.

    ``query()`` / ``ingest()`` raise if ever reached -- the runner must drive
    retrieval solely through ``retrieve``.
    """

    def __init__(self, per_query=None, default=None):
        self.per_query = {q: list(r) for q, r in (per_query or {}).items()}
        self.default = list(default or [])
        self.calls = []  # (query, top_k, candidate_k) tuples, in order

    def ingest(self, document):
        raise AssertionError("runner must never ingest")

    def retrieve(self, query, *, top_k=5, candidate_k=20):
        self.calls.append((query, top_k, candidate_k))
        return list(self.per_query.get(query, self.default))

    def query(self, query, *, top_k=5, candidate_k=20, max_chars=2000):
        raise AssertionError("runner must never call query")


class RecordingHarness(EvaluationHarness):
    """EvaluationHarness that records its run() call and may short-circuit."""

    def __init__(self, result=None):
        super().__init__(RetrievalEvaluator())
        self.sentinel = result
        self.run_calls = 0
        self.last_cases = None
        self.last_predictions = None

    def run(self, cases, predictions):
        self.run_calls += 1
        self.last_cases = cases
        self.last_predictions = predictions
        if self.sentinel is not None:
            return self.sentinel
        return super().run(cases, predictions)


# --- A. Construction / injection ---


def test_runner_constructs_with_pipeline_and_harness():
    runner = RetrievalEvaluationRunner(pipeline=FakePipeline(), harness=_harness())
    assert isinstance(runner.pipeline, RAGPipeline)
    assert isinstance(runner.harness, EvaluationHarness)


def test_runner_rejects_non_pipeline():
    with pytest.raises(TypeError):
        RetrievalEvaluationRunner(pipeline="not a pipeline", harness=_harness())


def test_runner_rejects_non_harness():
    with pytest.raises(TypeError):
        RetrievalEvaluationRunner(pipeline=FakePipeline(), harness="not a harness")


# --- B. Orchestration: per-case retrieve, query passthrough, no sorting ---


def test_run_returns_evaluation_result_for_a_hit():
    cases = [_case("q", "A")]
    pipe = FakePipeline(per_query={"q": [_sr("A"), _sr("B")]})
    result = RetrievalEvaluationRunner(pipeline=pipe, harness=_harness()).run(cases)
    assert isinstance(result, EvaluationResult)
    assert result.total_cases == 1 and result.evaluated_cases == 1
    assert result.hit_rate == 1.0 and result.recall == 1.0  # real metrics, not made up


def test_each_case_retrieved_exactly_once_in_case_order():
    cases = [_case("q1", "A"), _case("q2", "B"), _case("q3", "C")]
    pipe = FakePipeline(default=[_sr("A")])
    RetrievalEvaluationRunner(pipeline=pipe, harness=_harness()).run(cases)
    assert [q for q, _, _ in pipe.calls] == ["q1", "q2", "q3"]  # once each, in order
    assert len(pipe.calls) == 3


def test_retrieve_receives_query_verbatim_no_normalization():
    case = _case("  Needle  ", "A")  # whitespace + casing must survive untouched
    pipe = FakePipeline(default=[_sr("A")])
    harness = RecordingHarness(result=_empty_result())  # no query-match needed
    RetrievalEvaluationRunner(pipeline=pipe, harness=harness).run([case])
    assert pipe.calls[0][0] == "  Needle  "
    # and the prediction the harness saw carries the same verbatim query.
    assert harness.last_predictions[0].query == "  Needle  "


def test_retrieve_receives_forwarded_top_k_and_candidate_k():
    pipe = FakePipeline(default=[_sr("A")])
    RetrievalEvaluationRunner(pipeline=pipe, harness=_harness()).run(
        [_case("q", "A")], top_k=3, candidate_k=9
    )
    assert pipe.calls == [("q", 3, 9)]


def test_run_defaults_to_top_k_5_candidate_k_20():
    pipe = FakePipeline(default=[_sr("A")])
    RetrievalEvaluationRunner(pipeline=pipe, harness=_harness()).run([_case("q", "A")])
    assert pipe.calls == [("q", 5, 20)]


def test_prediction_preserves_pipeline_order_without_sorting():
    scrambled = [_sr("C", score=0.9), _sr("A", score=-2.0), _sr("B", score=0.1)]
    pipe = FakePipeline(default=scrambled)
    harness = RecordingHarness(result=_empty_result())
    RetrievalEvaluationRunner(pipeline=pipe, harness=harness).run([_case("q", "A")])
    results = harness.last_predictions[0].results
    assert [r.chunk.id for r in results] == ["C", "A", "B"]  # order untouched
    assert [r.score for r in results] == [0.9, -2.0, 0.1]  # scores untouched


def test_prediction_carries_the_actual_returned_result_objects():
    sr = _sr("A", score=0.7)
    pipe = FakePipeline(default=[sr])
    harness = RecordingHarness(result=_empty_result())
    RetrievalEvaluationRunner(pipeline=pipe, harness=harness).run([_case("q", "A")])
    # Same SearchResult objects the pipeline handed back -- nothing rebuilt/re-scored.
    assert harness.last_predictions[0].results == (sr,)
    assert harness.last_predictions[0].results[0] is sr


def test_harness_run_called_exactly_once():
    harness = RecordingHarness(result=_empty_result())
    pipe = FakePipeline(default=[_sr("A")])
    cases = [_case("q1", "A"), _case("q2", "A")]
    RetrievalEvaluationRunner(pipeline=pipe, harness=harness).run(cases)
    assert harness.run_calls == 1
    assert harness.last_cases is cases  # original list, same order, not copied
    assert len(harness.last_predictions) == 2


def test_run_returns_harness_result_unchanged():
    sentinel = EvaluationResult(
        hit_rate=0.25, recall=0.5, precision=0.75, total_cases=2, evaluated_cases=2
    )
    harness = RecordingHarness(result=sentinel)
    pipe = FakePipeline(default=[_sr("A")])
    out = RetrievalEvaluationRunner(pipeline=pipe, harness=harness).run(
        [_case("q1", "A"), _case("q2", "B")]
    )
    assert out is sentinel  # runner constructs no EvaluationResult of its own


def test_runner_does_not_compute_metrics_itself():
    # Predictions that would score non-trivially are delegated; a sentinel zero
    # result proves the runner never touches metric math.
    sentinel = _empty_result()
    harness = RecordingHarness(result=sentinel)
    pipe = FakePipeline(default=[_sr("A"), _sr("B")])  # would score 1.0 if computed
    out = RetrievalEvaluationRunner(pipeline=pipe, harness=harness).run([_case("q", "A")])
    assert out is sentinel
    assert (out.hit_rate, out.recall, out.precision) == (0.0, 0.0, 0.0)


def test_runner_drives_retrieve_not_query_or_ingest():
    # FakePipeline.query/ingest raise on contact; run succeeds purely via retrieve.
    pipe = FakePipeline(default=[_sr("A")])
    result = RetrievalEvaluationRunner(pipeline=pipe, harness=_harness()).run(
        [_case("q", "A")]
    )
    assert result.total_cases == 1
    assert len(pipe.calls) == 1


def test_duplicate_ids_across_retrieve_results_are_rejected_by_prediction():
    # EvaluationPrediction forbids duplicate chunk.id; the runner surfaces that
    # contract violation instead of silently deduplicating.
    pipe = FakePipeline(default=[_sr("A"), _sr("A")])
    runner = RetrievalEvaluationRunner(pipeline=pipe, harness=_harness())
    with pytest.raises(ValueError):
        runner.run([_case("q", "A")])


def _empty_result():
    return EvaluationResult(
        hit_rate=0.0, recall=0.0, precision=0.0, total_cases=0, evaluated_cases=0
    )


# --- C. Empty-run semantics (delegated to harness, never self-constructed) ---


def test_empty_cases_still_reach_harness_run_with_empty_lists():
    pipe = FakePipeline(default=[_sr("A")])
    harness = RecordingHarness(result=_empty_result())
    result = RetrievalEvaluationRunner(pipeline=pipe, harness=harness).run([])
    assert pipe.calls == []  # nothing to retrieve
    assert harness.run_calls == 1
    assert harness.last_cases == [] and harness.last_predictions == []
    assert result == _empty_result()


def test_empty_run_matches_real_evaluator_contract():
    result = RetrievalEvaluationRunner(
        pipeline=FakePipeline(), harness=_harness()
    ).run([])
    assert result == RetrievalEvaluator().evaluate([], [])
    assert result.total_cases == 0


def test_query_with_no_retrieved_results_scores_zero_but_is_evaluated():
    pipe = FakePipeline(default=[])  # empty retrieval
    result = RetrievalEvaluationRunner(pipeline=pipe, harness=_harness()).run(
        [_case("q", "A")]
    )
    assert result.total_cases == 1 and result.evaluated_cases == 1
    assert (result.hit_rate, result.recall, result.precision) == (0.0, 0.0, 0.0)
    assert (result.mrr, result.map, result.ndcg) == (0.0, 0.0, 0.0)


# --- D. Input validation (fails before any retrieve) ---


def test_cases_must_be_a_list():
    pipe = FakePipeline()
    with pytest.raises(TypeError):
        RetrievalEvaluationRunner(pipeline=pipe, harness=_harness()).run(
            _case("q", "A")  # type: ignore[arg-type]
        )
    assert pipe.calls == []


def test_case_elements_must_be_evaluation_cases():
    pipe = FakePipeline()
    with pytest.raises(TypeError):
        RetrievalEvaluationRunner(pipeline=pipe, harness=_harness()).run(
            [_case("q", "A"), "not a case"]  # type: ignore[list-item]
        )
    assert pipe.calls == []


@pytest.mark.parametrize(
    "bad, kwargs",
    [
        (True, {}),                 # bool is not an int cutoff
        (1.5, {}),                  # float
        ("5", {}),                  # str
    ],
)
def test_top_k_must_be_an_int(bad, kwargs):
    pipe = FakePipeline()
    with pytest.raises(TypeError):
        RetrievalEvaluationRunner(pipeline=pipe, harness=_harness()).run(
            [_case("q", "A")], top_k=bad, **kwargs  # type: ignore[arg-type]
        )
    assert pipe.calls == []


@pytest.mark.parametrize("bad", [True, 1.5, "20"])
def test_candidate_k_must_be_an_int(bad):
    pipe = FakePipeline()
    with pytest.raises(TypeError):
        RetrievalEvaluationRunner(pipeline=pipe, harness=_harness()).run(
            [_case("q", "A")], candidate_k=bad  # type: ignore[arg-type]
        )
    assert pipe.calls == []


@pytest.mark.parametrize("bad", [0, -3])
def test_cutoffs_must_be_at_least_one(bad):
    pipe = FakePipeline()
    with pytest.raises(ValueError):
        RetrievalEvaluationRunner(pipeline=pipe, harness=_harness()).run(
            [_case("q", "A")], top_k=bad
        )
    assert pipe.calls == []


def test_candidate_k_below_top_k_rejected_before_retrieve():
    pipe = FakePipeline()
    runner = RetrievalEvaluationRunner(pipeline=pipe, harness=_harness())
    with pytest.raises(ValueError):
        runner.run([_case("q", "A")], top_k=30)  # default candidate_k=20 < 30
    assert pipe.calls == []
    with pytest.raises(ValueError):
        runner.run([_case("q", "A")], top_k=10, candidate_k=5)
    assert pipe.calls == []


# --- E. Determinism ---


def test_two_runs_return_equal_results_and_identical_delegation():
    cases = [_case("q1", "A"), _case("q2", "A", "B")]
    script = {
        "q1": [_sr("A"), _sr("X")],
        "q2": [_sr("B"), _sr("A")],
    }

    def run_once():
        pipe = FakePipeline(per_query=script)
        out = RetrievalEvaluationRunner(pipeline=pipe, harness=_harness()).run(
            cases, top_k=2, candidate_k=4
        )
        return out, pipe.calls

    first, log_first = run_once()
    second, log_second = run_once()
    assert first == second
    assert first.to_dict() == second.to_dict()
    assert first.summary() == second.summary()
    assert log_first == log_second


# --- F. Input immutability ---


def test_run_never_mutates_cases_or_the_pipeline_contract():
    sr = _sr("A", score=0.8)
    pipe = FakePipeline(default=[sr])
    cases = [_case("q", "A")]
    snapshot = [(c.query, c.relevant_chunk_ids) for c in cases]

    RetrievalEvaluationRunner(pipeline=pipe, harness=_harness()).run(cases)

    assert [(c.query, c.relevant_chunk_ids) for c in cases] == snapshot
    assert sr.chunk.id == "A" and sr.score == 0.8  # result object untouched


# --- G. End-to-end with real pipeline + real harness ---


def _real_pipeline(content, doc_id="doc"):
    chunker = FixedSizeChunker(chunk_size=12)
    embedder = DeterministicEmbedder(dimension=8)
    vstore, lindex = InMemoryVectorStore(), InMemoryLexicalIndex()
    hybrid = HybridRetriever(
        DenseRetriever(embedder, vstore),
        LexicalRetriever(lindex),
        WeightedScoreFusion(),
    )
    pipe = DeterministicRAGPipeline(
        chunker=chunker,
        embedder=embedder,
        vector_store=vstore,
        lexical_index=lindex,
        retriever=hybrid,
        reranker=DeterministicReranker(),
        context_builder=DeterministicContextBuilder(),
    )
    doc = Document(id=doc_id, content=content)
    pipe.ingest(doc)
    return pipe, chunker, doc


def test_end_to_end_real_pipeline_through_runner_hits_ground_truth():
    content = "needle lies hidden deep in this haystack " + "wool " * 6 + "seed"
    pipe, chunker, doc = _real_pipeline(content)
    needle_id = next(c.id for c in chunker.chunk(doc) if "needle" in c.text)

    # top_k/candidate_k >= chunk count -> the needle chunk is guaranteed back.
    result = RetrievalEvaluationRunner(pipeline=pipe, harness=_harness()).run(
        [_case("needle", needle_id)], top_k=10, candidate_k=10
    )

    assert result.total_cases == 1 and result.evaluated_cases == 1
    assert result.hit_rate == 1.0 and result.recall == 1.0  # needle retrieved
    assert 0.0 < result.precision <= 1.0
    assert result == RetrievalEvaluationRunner(pipeline=pipe, harness=_harness()).run(
        [_case("needle", needle_id)], top_k=10, candidate_k=10
    )  # deterministic rerun


def test_public_import():
    from zmrag import __all__

    assert "RetrievalEvaluationRunner" in __all__
    runner = RetrievalEvaluationRunner(pipeline=FakePipeline(), harness=_harness())
    assert isinstance(runner, RetrievalEvaluationRunner)
