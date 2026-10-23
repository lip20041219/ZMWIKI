"""Evaluation Harness tests: orchestration, validation, reporting.

Covers the Phase 8B EvaluationHarness: delegation to an injected Evaluator
(never internal construction), input consistency validation (types, positional
length, query match), input immutability, determinism, stable machine/human
reporting via EvaluationResult.to_dict/summary, and an end-to-end proof that
DeterministicRAGPipeline output is consumable by the harness.
"""

import pytest

from zmwiki import (
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
    Evaluator,
    FixedSizeChunker,
    HybridRetriever,
    InMemoryLexicalIndex,
    InMemoryVectorStore,
    LexicalRetriever,
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


def _pred(query, *results):
    return EvaluationPrediction(query=query, results=results)


def _result(hit=1.0, recall=1.0, precision=1.0, total=1, mrr=0.0, map=0.0, ndcg=0.0):
    return EvaluationResult(
        hit_rate=hit, recall=recall, precision=precision,
        total_cases=total, evaluated_cases=total,
        mrr=mrr, map=map, ndcg=ndcg,
    )


def _harness(evaluator=None):
    return EvaluationHarness(evaluator or RetrievalEvaluator())


class FakeEvaluator(Evaluator):
    def __init__(self, result):
        self.result = result
        self.calls = 0
        self.last_cases = None
        self.last_predictions = None

    def evaluate(self, cases, predictions):
        self.calls += 1
        self.last_cases = cases
        self.last_predictions = predictions
        return self.result


# --- A. Normal runs ---


def test_single_query_run_returns_evaluation_result():
    cases = [_case("q", "A")]
    predictions = [_pred("q", _sr("A"), _sr("B"))]
    result = _harness().run(cases, predictions)
    # relevant={A}, ranked [A, B] -> ranking metrics are all perfect at rank 1.
    assert result == _result(hit=1.0, recall=1.0, precision=0.5,
                             mrr=1.0, map=1.0, ndcg=1.0)
    assert isinstance(result, EvaluationResult)


def test_multi_query_macro_metrics_passthrough():
    cases = [_case("q1", "A"), _case("q2", "A", "B"), _case("q3", "A")]
    predictions = [
        _pred("q1", _sr("A")),
        _pred("q2", _sr("A"), _sr("C")),
        _pred("q3", _sr("B"), _sr("C")),
    ]
    result = _harness().run(cases, predictions)
    # exactly the macro values RetrievalEvaluator computes; harness adds nothing
    assert result.recall == pytest.approx((1.0 + 0.5 + 0.0) / 3)
    assert result.hit_rate == pytest.approx(2.0 / 3.0)
    assert (result.total_cases, result.evaluated_cases) == (3, 3)


# --- F. Fake evaluator: delegation, single call, passthrough ---


def test_fake_evaluator_called_once_with_original_arguments():
    sentinel = _result(hit=0.25, recall=0.5, precision=0.75, total=2)
    fake = FakeEvaluator(sentinel)
    harness = _harness(fake)
    cases = [_case("q1", "A"), _case("q2", "B")]
    predictions = [_pred("q1", _sr("A")), _pred("q2", _sr("B"))]
    out = harness.run(cases, predictions)

    assert fake.calls == 1
    assert fake.last_cases is cases  # original objects, same order, not copied
    assert fake.last_predictions is predictions
    assert out is sentinel  # harness returns the evaluator's result unchanged


def test_harness_does_not_compute_metrics_itself():
    fake = FakeEvaluator(_result(hit=0.0, recall=0.0, precision=0.0, total=1))
    out = _harness(fake).run([_case("q", "A")], [_pred("q", _sr("A"))])
    assert out == _result(hit=0.0, recall=0.0, precision=0.0, total=1)


def test_harness_constructor_rejects_non_evaluator():
    with pytest.raises(TypeError):
        EvaluationHarness("not an evaluator")  # type: ignore[arg-type]


# --- B. Input validation ---


def test_cases_must_be_list():
    with pytest.raises(TypeError):
        _harness().run(_case("q", "A"), [_pred("q", _sr("A"))])  # type: ignore[arg-type]


def test_predictions_must_be_list():
    with pytest.raises(TypeError):
        _harness().run([_case("q", "A")], _pred("q", _sr("A")))  # type: ignore[arg-type]


def test_case_element_wrong_type_rejected():
    with pytest.raises(TypeError):
        _harness().run([_case("q", "A")], ["not a prediction"])  # type: ignore[list-item]


def test_prediction_element_wrong_type_rejected():
    with pytest.raises(TypeError):
        _harness().run(["not a case"], [_pred("q")])  # type: ignore[list-item]


@pytest.mark.parametrize(
    "cases, predictions",
    [
        ([_case("q", "A")], []),                 # predictions too short
        ([], [_pred("q", _sr("A"))]),            # cases too short
        ([_case("q1", "A")], [_pred("q1"), _pred("q1")]),  # predictions too long
    ],
)
def test_length_mismatch_rejected(cases, predictions):
    with pytest.raises(ValueError):
        _harness().run(cases, predictions)


def test_query_mismatch_rejected_without_correction():
    fake = FakeEvaluator(_result())
    harness = _harness(fake)
    with pytest.raises(ValueError):
        harness.run([_case("needle", "A")], [_pred("haystack", _sr("A"))])
    assert fake.calls == 0  # validation happens before delegation


def test_validation_fails_before_evaluator_is_called():
    fake = FakeEvaluator(_result())
    with pytest.raises(ValueError):
        _harness(fake).run([_case("q1", "A")], [_pred("q1")] * 2)
    assert fake.calls == 0


# --- empty-input semantics (delegated to evaluator, no NaN/None/fake zeros) ---


def test_empty_run_delegates_and_matches_evaluator_contract():
    result = _harness().run([], [])
    assert result == RetrievalEvaluator().evaluate([], [])
    assert result.total_cases == 0
    assert (result.hit_rate, result.recall, result.precision) == (0.0, 0.0, 0.0)


# --- C. Immutability ---


def test_run_never_mutates_inputs():
    chunk_a, chunk_b = _chunk("A", text="alpha body"), _chunk("B", text="beta body")
    cases = [_case("q", "A", "B")]
    predictions = [_pred("q", SearchResult(chunk=chunk_a, score=0.9), SearchResult(chunk=chunk_b, score=0.1))]

    cases_before = [(c.query, c.relevant_chunk_ids) for c in cases]
    preds_before = [(p.query, p.results) for p in predictions]

    _harness().run(cases, predictions)

    assert cases == cases and predictions == predictions
    assert [(c.query, c.relevant_chunk_ids) for c in cases] == cases_before
    assert [(p.query, p.results) for p in predictions] == preds_before
    assert chunk_a.id == "A" and chunk_b.id == "B"
    assert chunk_a.text == "alpha body" and chunk_b.text == "beta body"
    assert predictions[0].results[0].score == 0.9
    assert predictions[0].results[1].score == 0.1


# --- D. Determinism ---


def test_repeated_run_is_equal_in_result_dict_and_summary():
    cases = [_case("q1", "A"), _case("q2", "A", "B"), _case("q3", "A")]
    predictions = [
        _pred("q1", _sr("A")),
        _pred("q2", _sr("A"), _sr("C")),
        _pred("q3", _sr("B")),
    ]
    h = _harness()
    first, second = h.run(cases, predictions), h.run(cases, predictions)
    assert first == second
    assert first.to_dict() == second.to_dict()
    assert first.summary() == second.summary()


def test_cross_evaluator_instance_is_deterministic():
    cases = [_case("q1", "A"), _case("q2", "A", "B")]
    predictions = [_pred("q1", _sr("A")), _pred("q2", _sr("A"), _sr("C"))]
    a = _harness(RetrievalEvaluator()).run(cases, predictions)
    b = _harness(RetrievalEvaluator()).run(cases, predictions)
    assert a == b
    assert a.to_dict() == b.to_dict()
    assert a.summary() == b.summary()


# --- E. Serialization / reporting ---


def test_to_dict_is_plain_json_friendly_and_fixed_key_order():
    result = _harness().run(
        [_case("q1", "A"), _case("q2", "A", "B"), _case("q3", "A")],
        [_pred("q1", _sr("A")), _pred("q2", _sr("A"), _sr("C")), _pred("q3", _sr("B"))],
    )
    d = result.to_dict()
    assert isinstance(d, dict)
    assert list(d.keys()) == [
        "hit_rate", "recall", "precision", "mrr", "map", "ndcg",
        "total_cases", "evaluated_cases",
    ]
    assert all(isinstance(v, (int, float)) for v in d.values())  # no objects/reprs


def test_to_dict_and_summary_contain_no_timestamp_or_absolute_path():
    result = _result(hit=2.0 / 3.0, recall=0.5, precision=0.5, total=3)
    dump = str(result.to_dict()) + result.summary()
    assert "timestamp" not in dump and "time" not in dump  # no clock/date words
    assert "\\" not in dump and "D:" not in dump and "C:" not in dump  # no paths/drives
    assert "@" not in dump  # no object reprs


def test_summary_format_is_fixed_and_deterministic():
    result = _result(hit=2.0 / 3.0, recall=0.5, precision=0.5, total=3,
                     mrr=2.0 / 3.0, map=0.5, ndcg=0.25)
    expected = (
        "## Evaluation Result\n\n"
        "Queries: 3\n"
        "Hit Rate: 0.666667\n"
        "Recall: 0.500000\n"
        "Precision: 0.500000\n"
        "MRR: 0.666667\n"
        "MAP: 0.500000\n"
        "NDCG: 0.250000\n"
    )
    assert result.summary() == expected
    assert result.summary() == _result(
        hit=2.0 / 3.0, recall=0.5, precision=0.5, total=3,
        mrr=2.0 / 3.0, map=0.5, ndcg=0.25,
    ).summary()


# --- G. End-to-end: Pipeline -> Context -> Prediction -> Harness ---


def _build_pipeline(doc_id, content):
    chunker = FixedSizeChunker(chunk_size=12)
    embedder = DeterministicEmbedder(dimension=8)
    vstore, lindex = InMemoryVectorStore(), InMemoryLexicalIndex()
    dense = DenseRetriever(embedder, vstore)
    lexical = LexicalRetriever(lindex)
    hybrid = HybridRetriever(dense, lexical, WeightedScoreFusion())
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
    return doc, chunker, pipe


def _prediction_from_context(query, context):
    return _pred(query, *[SearchResult(chunk=c, score=0.0) for c in context.chunks])


def test_end_to_end_pipeline_context_through_harness_hit_and_miss():
    # Both corpora chunk into windows that are never whitespace-only, and only
    # doc-a mentions "needle", so corpus B can never return doc-a's chunks.
    doc_a, chunker_a, pipe_a = _build_pipeline(
        "doc-a", content="needle lies hidden deep in this haystack " + "wool " * 6 + "seed"
    )
    doc_b, _, pipe_b = _build_pipeline(
        "doc-b", content="economics markets trade growth inflation " + "padding " * 4 + "metric"
    )

    # top_k >= chunk count -> the whole pool is returned, so the needle chunk is in.
    ctx_a = pipe_a.query("needle", top_k=10, candidate_k=10, max_chars=2000)
    needle_id = next(c.id for c in chunker_a.chunk(doc_a) if "needle" in c.text)
    assert any(c.id == needle_id for c in ctx_a.chunks)  # the guaranteed hit

    # A query on corpus B returns only doc-b chunks -- doc-a's needle id is absent.
    ctx_b = pipe_b.query("needle", top_k=10, candidate_k=10, max_chars=2000)
    assert all(c.id.startswith("doc-b") for c in ctx_b.chunks)

    cases = [
        _case("needle", needle_id),   # hit: needle chunk retrieved
        _case("needle", needle_id),   # miss: corpus B never returns a doc-a chunk
    ]
    predictions = [
        _prediction_from_context("needle", ctx_a),
        _prediction_from_context("needle", ctx_b),
    ]
    result = _harness().run(cases, predictions)

    assert result.total_cases == 2
    assert result.hit_rate == 0.5            # exactly one hit, one miss
    assert result.recall == pytest.approx(0.5)
    assert 0.0 <= result.precision <= 1.0
    assert result == _harness().run(cases, predictions)  # deterministic rerun

    assert doc_a.id == "doc-a" and doc_b.id == "doc-b"  # inputs untouched


def test_public_import():
    from zmwiki import __all__

    assert "EvaluationHarness" in __all__
    assert isinstance(_harness(), EvaluationHarness)
