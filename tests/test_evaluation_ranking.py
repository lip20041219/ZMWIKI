"""Phase 8C ranking-metric tests: MRR / MAP / NDCG and @K semantics.

Standard information-retrieval ranking metrics over RetrievalEvaluator.
Relevance is binary and decided solely by ``chunk.id`` membership in
``relevant_chunk_ids``; the prediction's given order is authoritative and is
never re-sorted by score or id. Metrics are macro-averaged over queries.
Coverage: per-query math, @K cutoffs, k validation, macro aggregation, input
immutability, determinism, serialization, and a pipeline E2E.
"""

import json
import math

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
    EvaluationPrediction,
    EvaluationResult,
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


def _pred(query, *ids):
    return EvaluationPrediction(query=query, results=[_sr(cid) for cid in ids])


def _evaluate(cases, predictions, k=None):
    return RetrievalEvaluator().evaluate(cases, predictions, k=k)


# --- MRR ---


@pytest.mark.parametrize(
    "retrieved, relevant, expected",
    [
        (["A", "X", "Y"], ["A"], 1.0),          # first result relevant
        (["X", "A", "Y"], ["A"], 0.5),          # relevant at rank 2
        (["X", "Y", "A"], ["A"], 1.0 / 3.0),    # relevant at rank 3
        (["X", "Y", "Z"], ["A"], 0.0),          # nothing relevant
        (["X", "A", "B"], ["A", "B"], 0.5),     # first relevant at rank 2
    ],
)
def test_mrr_single_query(retrieved, relevant, expected):
    result = _evaluate([_case("q", *relevant)], [_pred("q", *retrieved)])
    assert result.mrr == pytest.approx(expected)


def test_mrr_ignores_results_after_first_relevant():
    # Only the first relevant result matters; later ones add nothing.
    assert _evaluate([_case("q", "A", "B")], [_pred("q", "A", "B")]).mrr == 1.0
    assert _evaluate([_case("q", "A", "B")], [_pred("q", "A", "X", "B")]).mrr == 1.0


def test_mrr_score_is_never_read():
    # Identical orderings with different scores must agree (order is authority).
    a = _evaluate([_case("q", "A")], [_pred("q", "B", "A")]).mrr
    ordered = [SearchResult(chunk=_chunk("B"), score=0.99),
               SearchResult(chunk=_chunk("A"), score=-3.0)]
    b = RetrievalEvaluator().evaluate(
        [_case("q", "A")], [EvaluationPrediction(query="q", results=ordered)]
    ).mrr
    assert a == b == 0.5


# --- MAP / AP ---


@pytest.mark.parametrize(
    "retrieved, relevant, expected",
    [
        (["A", "B"], ["A", "B"], 1.0),               # perfect ranking
        (["X", "A"], ["A"], 0.5),                    # one relevant at rank 2
        (["A", "X", "B"], ["A", "B"], 5.0 / 6.0),    # (P@1 + P@3) / 2
        (["A", "X"], ["A", "B"], 0.5),               # B unretrieved still in denom
        (["X", "Y"], ["A", "B"], 0.0),               # nothing relevant
    ],
)
def test_average_precision_single_query(retrieved, relevant, expected):
    result = _evaluate([_case("q", *relevant)], [_pred("q", *retrieved)])
    assert result.map == pytest.approx(expected)


def test_average_precision_denominator_is_ground_truth_size():
    # Unretrieved relevant ids must not be silently dropped from the AP denom.
    cases = [_case("q", "A", "B")]
    only_a = _pred("q", "A")
    assert _evaluate(cases, [only_a]).map == pytest.approx(0.5)  # 1.0 / 2
    a_and_b = _pred("q", "A", "B")
    assert _evaluate(cases, [a_and_b]).map == pytest.approx(1.0)  # (1 + 1) / 2


def test_average_precision_no_duplicate_boost():
    # A duplicate would double-count a gain; EvaluationPrediction rejects dupes.
    with pytest.raises(ValueError):
        _pred("q", "A", "A")


# --- NDCG ---


def test_ndcg_perfect_ranking_is_one():
    result = _evaluate([_case("q", "A", "B")], [_pred("q", "A", "B", "X")])
    assert result.ndcg == pytest.approx(1.0)


def test_ndcg_relevant_at_rank_two():
    result = _evaluate([_case("q", "A")], [_pred("q", "X", "A")])
    assert result.ndcg == pytest.approx(1.0 / math.log2(3))


def test_ndcg_bad_ranking_is_dcg_over_idcg():
    # [X, A, B] vs ideal [A, B]:
    #   DCG  = 1/log2(3) + 1/log2(4)
    #   IDCG = 1/log2(2) + 1/log2(3)
    result = _evaluate([_case("q", "A", "B")], [_pred("q", "X", "A", "B")])
    dcg = 1.0 / math.log2(3) + 1.0 / math.log2(4)
    idcg = 1.0 / math.log2(2) + 1.0 / math.log2(3)
    assert result.ndcg == pytest.approx(dcg / idcg)


def test_ndcg_empty_retrieval_is_zero():
    # DCG = IDCG = 0 over no results; the ratio is defined as 0.0, never 0/0.
    result = _evaluate([_case("q", "A")], [_pred("q")])
    assert result.ndcg == 0.0
    assert result.mrr == 0.0
    assert result.map == 0.0


def test_ndcg_nonrelevant_between_relevants_hurts():
    # [A, X, B]: the non-relevant X at rank 2 pushes B to rank 3, losing gain.
    #   DCG  = 1/log2(2) + 1/log2(4);  IDCG = 1/log2(2) + 1/log2(3)
    result = _evaluate([_case("q", "A", "B")], [_pred("q", "A", "X", "B")])
    dcg = 1.0 / math.log2(2) + 1.0 / math.log2(4)
    idcg = 1.0 / math.log2(2) + 1.0 / math.log2(3)
    assert result.ndcg == pytest.approx(dcg / idcg)
    assert result.ndcg < 1.0  # perfect, but reordered, would be 1.0
    # Swapping the two relevant ids (still ranks 1 and 2) changes nothing.
    assert _evaluate([_case("q", "A", "B")], [_pred("q", "B", "A", "X")]).ndcg == 1.0


# --- @K semantics ---


def test_ranking_cutoff_mrr_and_ndcg():
    cases = [_case("q", "A")]
    predictions = [_pred("q", "X", "A")]
    at_1 = _evaluate(cases, predictions, k=1)
    at_2 = _evaluate(cases, predictions, k=2)
    assert at_1.mrr == 0.0 and at_2.mrr == 0.5          # MRR@1 = 0, MRR@2 = 0.5
    assert at_1.ndcg == 0.0                             # NDCG@1 = 0
    assert at_2.ndcg == pytest.approx(1.0 / math.log2(3))  # NDCG@2


@pytest.mark.parametrize("k", [3, 4, 5])
def test_k_equals_or_exceeds_result_count_matches_full(k):
    # k >= number of results truncates nothing: identical to k=None.
    cases = [_case("q", "A", "B")]
    predictions = [_pred("q", "B", "A", "X")]  # 3 results
    assert _evaluate(cases, predictions, k=k) == _evaluate(cases, predictions, k=None)


def test_k_cutoff_does_not_change_phase_8a_metrics():
    # HR / Recall / Precision always use the whole prediction, never @K.
    cases = [_case("q", "A")]
    predictions = [_pred("q", "X", "A", "B")]
    full = _evaluate(cases, predictions)
    cut = _evaluate(cases, predictions, k=1)
    assert (cut.hit_rate, cut.recall, cut.precision) == (1.0, 1.0, 1.0 / 3.0)
    assert cut.hit_rate == full.hit_rate
    assert cut.recall == full.recall
    assert cut.precision == full.precision
    # ...while the ranking metrics genuinely reflect the depth-1 cut.
    assert cut.mrr == 0.0 and full.mrr == 0.5
    assert cut.map == 0.0 and full.map == pytest.approx(0.5)


# --- k validation ---


@pytest.mark.parametrize(
    "k, exc",
    [
        (True, TypeError),      # bool is not an int cutoff
        (False, TypeError),
        (1.5, TypeError),       # float
        ("2", TypeError),       # str
        (0, ValueError),        # must be >= 1
        (-3, ValueError),
    ],
)
def test_invalid_k_rejected(k, exc):
    with pytest.raises(exc):
        _evaluate([_case("q", "A")], [_pred("q", "A")], k=k)  # type: ignore[arg-type]


def test_k_none_means_full_results_no_implicit_truncation():
    cases = [_case("q", "A")]
    predictions = [_pred("q", "X", "Y", "A", "Z")]
    assert _evaluate(cases, predictions) == _evaluate(cases, predictions, k=None)
    assert _evaluate(cases, predictions).mrr == pytest.approx(1.0 / 3)  # rank 3


# --- Macro average ---


def test_macro_average_mrr_map_ndcg():
    cases = [_case("q1", "A"), _case("q2", "A"), _case("q3", "A")]
    predictions = [_pred("q1", "A"), _pred("q2", "X", "A"), _pred("q3", "Y", "Z")]
    result = _evaluate(cases, predictions)
    assert result.mrr == pytest.approx((1.0 + 0.5 + 0.0) / 3)
    assert result.map == pytest.approx((1.0 + 0.5 + 0.0) / 3)
    assert result.ndcg == pytest.approx((1.0 + 1.0 / math.log2(3) + 0.0) / 3)
    assert (result.total_cases, result.evaluated_cases) == (3, 3)


def test_empty_cases_ranking_metrics_are_zero_aggregates():
    result = _evaluate([], [])
    assert result.mrr == 0.0 and result.map == 0.0 and result.ndcg == 0.0
    assert result.total_cases == 0


# --- Input immutability ---


def test_evaluate_never_mutates_inputs_with_cutoff():
    chunk_a, chunk_b, chunk_c = _chunk("A"), _chunk("B"), _chunk("C")
    cases = [_case("q", "A", "B")]
    predictions = [
        EvaluationPrediction(
            query="q",
            results=[
                SearchResult(chunk=chunk_a, score=0.9),
                SearchResult(chunk=chunk_b, score=0.1),
                SearchResult(chunk=chunk_c, score=-1.0),
            ],
        )
    ]
    before_cases = [(c.query, c.relevant_chunk_ids) for c in cases]
    before_preds = [(p.query, list(p.results)) for p in predictions]

    _evaluate(cases, predictions, k=2)

    assert [(c.query, c.relevant_chunk_ids) for c in cases] == before_cases
    assert [(p.query, list(p.results)) for p in predictions] == before_preds
    # Original order and scores untouched (no sort / pop / reverse / rewrite).
    assert [r.chunk.id for r in predictions[0].results] == ["A", "B", "C"]
    assert [r.score for r in predictions[0].results] == [0.9, 0.1, -1.0]
    assert chunk_a.id == "A" and chunk_a.text == "A"


# --- Determinism ---


def test_ranking_metrics_deterministic_across_calls_and_instances():
    cases = [_case("q1", "A"), _case("q2", "A", "B"), _case("q3", "A")]
    predictions = [_pred("q1", "A"), _pred("q2", "B", "A", "X"), _pred("q3", "Y", "Z")]
    first = _evaluate(cases, predictions, k=2)
    second = _evaluate(cases, predictions, k=2)
    third = RetrievalEvaluator().evaluate(cases, predictions, k=2)
    assert first == second == third
    assert first.to_dict() == second.to_dict()
    assert first.summary() == second.summary()


# --- Serialization of the ranking metrics ---


def test_to_dict_includes_ranking_metrics_in_fixed_order():
    result = _evaluate([_case("q", "A")], [_pred("q", "X", "A")])
    d = result.to_dict()
    assert list(d.keys()) == [
        "hit_rate", "recall", "precision", "mrr", "map", "ndcg",
        "total_cases", "evaluated_cases",
    ]
    assert result.mrr == pytest.approx(0.5)
    assert result.map == pytest.approx(0.5)
    assert d["mrr"] == result.mrr
    json.dumps(d)  # JSON-friendly (no numpy / repr / path / timestamp)


def test_summary_contains_ranking_metrics_at_six_decimals():
    cases = [_case("q1", "A"), _case("q2", "A", "B"), _case("q3", "A")]
    predictions = [_pred("q1", "A"), _pred("q2", "B", "A", "X"), _pred("q3", "Y", "Z")]
    text = _evaluate(cases, predictions, k=2).summary()
    assert "MRR: " in text and "MAP: " in text and "NDCG: " in text
    lines = {ln.split(":")[0]: ln.split(": ")[1] for ln in text.splitlines()
             if ": " in ln}
    assert lines["MRR"] == lines["MAP"]  # single relevant each -> AP == RR == mrr
    assert set(lines["MRR"]) <= set("0123456789.")


# --- Pipeline E2E (evaluation imports no pipeline; the test file may) ---


def _needle_pipeline():
    doc = Document(
        id="rank-e2e",
        content="needle " + "filler " * 10 + "trailing words fill the tail",
    )
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
    pipe.ingest(doc)
    return doc, chunker, pipe


def test_end_to_end_pipeline_feeds_ranking_metrics():
    doc, chunker, pipe = _needle_pipeline()
    ctx = pipe.query("needle", top_k=20, candidate_k=20, max_chars=5000)
    assert ctx.chunks

    # Single ground-truth id, present exactly once in the full returned pool.
    needle_id = next(c.id for c in chunker.chunk(doc) if "needle" in c.text)
    rank = next(i for i, c in enumerate(ctx.chunks, start=1) if c.id == needle_id)

    cases = [_case("needle", needle_id)]
    prediction = [
        EvaluationPrediction(
            query="needle",
            results=[SearchResult(chunk=c, score=0.0) for c in ctx.chunks],
        )
    ]
    result = _evaluate(cases, prediction)

    # Not a quality claim -- exact ranking math over real pipeline output.
    assert result.mrr == pytest.approx(1.0 / rank)
    assert result.map == pytest.approx(1.0 / rank)          # one relevant -> AP == RR
    assert result.ndcg == pytest.approx(1.0 / math.log2(rank + 1))
    assert result == _evaluate(cases, prediction)            # deterministic rerun

    # A ghost ground-truth id that no pipeline produced must score zero.
    ghost = _evaluate([_case("needle", "not-indexed")], prediction)
    assert (ghost.mrr, ghost.map, ghost.ndcg) == (0.0, 0.0, 0.0)
    assert isinstance(result, EvaluationResult)


def test_public_surface_has_ranking_metrics_on_result():
    result = _evaluate([_case("q", "A")], [_pred("q", "A")])
    assert result.mrr == 1.0 and result.map == 1.0 and result.ndcg == 1.0
    assert result.to_dict()["mrr"] == 1.0
