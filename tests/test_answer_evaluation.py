"""Answer evaluation tests: the three deterministic lexical baselines.

Everything here is hermetic and dependency-free -- no model, no network, no LLM
judge. These tests pin the metric definitions (including their empty-input
conventions), the immutable result, and the negative cases that show the three
measures are independent of one another.
"""

import dataclasses
import json

import pytest

from zmwiki import (
    AnswerEvaluation,
    AnswerEvaluator,
    Chunk,
    Citation,
    Context,
    SimpleTokenizer,
    aggregate_evaluations,
)

REFERENCE = "The cache evicts the least recently used entry."
CONTEXT_TEXT = "A cache stores recently used data. An eviction policy decides which entry to remove."


def _chunk(cid, text, position=0):
    return Chunk(id=cid, document_id="doc", text=text, position=position)


def _context(text=CONTEXT_TEXT):
    return Context(text=text, chunks=(_chunk("doc::0", text or "x"),))


def _citation(text, cid="doc::0"):
    return Citation(
        document_id="doc", chunk_id=cid, position=0, text=text
    )


def _evaluate(
    *,
    answer,
    reference=REFERENCE,
    context=None,
    sources=(),
    query="what does the cache evict?",
    query_id="case-1",
    **kwargs,
):
    evaluator = AnswerEvaluator(**kwargs)
    return evaluator.evaluate(
        query_id=query_id,
        query=query,
        answer=answer,
        context=context if context is not None else _context(),
        reference_answer=reference,
        sources=sources,
    )


# --- API: shape, immutability, determinism, validation -------------------------


def test_result_is_frozen_and_json_serializable():
    result = _evaluate(answer="The cache evicts the least recently used entry.")
    with pytest.raises(dataclasses.FrozenInstanceError):
        result.reference_coverage = 0.5  # type: ignore[misc]
    payload = json.dumps(result.to_dict())  # plain built-ins only
    assert "reference_coverage" in payload
    assert result.summary().startswith("## Answer Evaluation")


def test_evaluation_fields_are_validated():
    base = dict(
        query_id="q",
        reference_coverage=0.5,
        groundedness=0.5,
        citation_coverage=0.5,
        has_answer=True,
        has_sources=True,
        source_count=1,
        answer_sentences=2,
        grounded_sentences=1,
        cited_sentences=1,
        reference_tokens=4,
        matched_reference_tokens=2,
    )
    AnswerEvaluation(**base)  # the well-formed one constructs
    with pytest.raises(ValueError):
        AnswerEvaluation(**{**base, "reference_coverage": 1.5})
    with pytest.raises(TypeError):
        AnswerEvaluation(**{**base, "reference_coverage": "0.5"})
    with pytest.raises(ValueError):
        AnswerEvaluation(**{**base, "grounded_sentences": 5})  # > sentences
    with pytest.raises(ValueError):
        AnswerEvaluation(**{**base, "matched_reference_tokens": 99})
    with pytest.raises(ValueError):
        AnswerEvaluation(**{**base, "source_count": 0})  # disagrees with the flag
    with pytest.raises(TypeError):
        AnswerEvaluation(**{**base, "has_answer": "yes"})
    with pytest.raises(TypeError):
        AnswerEvaluation(**{**base, "query_id": ""})


def test_evaluation_is_deterministic_across_calls_and_instances():
    first = _evaluate(answer="The cache evicts the least recently used entry.")
    second = _evaluate(answer="The cache evicts the least recently used entry.")
    assert first == second
    assert first.to_dict() == second.to_dict()
    assert first.summary() == second.summary()


def test_tokenizer_and_threshold_are_injectable_and_validated():
    assert isinstance(AnswerEvaluator().tokenizer, SimpleTokenizer)
    assert AnswerEvaluator().groundedness_threshold == 0.5
    assert AnswerEvaluator(tokenizer=SimpleTokenizer()).groundedness_threshold == 0.5
    assert AnswerEvaluator(groundedness_threshold=1).groundedness_threshold == 1.0
    with pytest.raises(TypeError):
        AnswerEvaluator(tokenizer="nope")  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        AnswerEvaluator(groundedness_threshold="0.5")  # type: ignore[arg-type]
    for bad in (0, -1, 1.5):
        with pytest.raises(ValueError):
            AnswerEvaluator(groundedness_threshold=bad)


@pytest.mark.parametrize(
    ("kwargs", "error"),
    [
        ({"query_id": ""}, TypeError),
        ({"query_id": 42}, TypeError),
        ({"query": "   "}, TypeError),
        ({"answer": 42}, TypeError),
        ({"context": "not a context"}, TypeError),
        ({"reference_answer": ""}, TypeError),
        ({"reference_answer": "  "}, TypeError),
        ({"sources": [_citation("x")]}, TypeError),  # a list is not a tuple
        ({"sources": ("not a citation",)}, TypeError),
    ],
)
def test_invalid_inputs_are_rejected(kwargs, error):
    call = {
        "query_id": "case-1",
        "query": "q",
        "answer": "a",
        "context": _context(),
        "reference_answer": "r",
        "sources": (),
    }
    call.update(kwargs)
    with pytest.raises(error):
        AnswerEvaluator().evaluate(**call)


# --- Reference coverage --------------------------------------------------------


def test_reference_coverage_exact_match_is_one():
    result = _evaluate(answer=REFERENCE)
    assert result.reference_coverage == 1.0
    assert result.matched_reference_tokens == result.reference_tokens


def test_reference_coverage_partial_match():
    # reference tokens: the, cache, evicts, least, recently, used, entry (7)
    result = _evaluate(answer="The cache evicts something.")
    assert result.matched_reference_tokens == 3
    assert result.reference_coverage == pytest.approx(3 / 7)


def test_reference_coverage_is_zero_for_an_unrelated_answer():
    assert _evaluate(answer="Quantum submarines navigate oceans.").reference_coverage == 0.0


@pytest.mark.parametrize(
    "answer",
    [
        "the cache evicts the least recently used entry",       # case
        "THE CACHE EVICTS THE LEAST RECENTLY USED ENTRY.",      # case + punctuation
        "  the   cache\nevicts the least recently used entry ",  # whitespace
        "The cache evicts the least recently used entry!!!",    # punctuation only
    ],
)
def test_reference_coverage_ignores_case_whitespace_and_punctuation(answer):
    assert _evaluate(answer=answer).reference_coverage == 1.0


def test_reference_coverage_matches_cjk_at_character_granularity():
    result = _evaluate(answer="苹果派很好吃", reference="苹果派好吃")
    assert result.reference_coverage == 1.0  # 苹 果 派 好 吃 all present
    assert _evaluate(answer="香蕉牛奶", reference="苹果派").reference_coverage == 0.0


def test_reference_coverage_counts_duplicate_reference_tokens_once():
    # "cat cat cat" is one fact, so the answer needs "cat" once.
    result = _evaluate(answer="cat", reference="cat cat cat")
    assert result.reference_tokens == 1
    assert result.reference_coverage == 1.0


def test_reference_coverage_ignores_extra_answer_tokens():
    result = _evaluate(answer=REFERENCE + " Extra words change nothing at all.")
    assert result.reference_coverage == 1.0


# --- Groundedness --------------------------------------------------------------


def test_fully_grounded_answer():
    context = _context("A cache stores recently used data. An eviction policy removes an entry.")
    result = _evaluate(answer="An eviction policy removes an entry.", context=context)
    assert result.groundedness == 1.0
    assert result.grounded_sentences == result.answer_sentences == 1


def test_partially_grounded_answer_counts_sentences():
    context = _context("An eviction policy removes an entry.")
    answer = "An eviction policy removes an entry. Quantum submarines navigate oceans."
    result = _evaluate(answer=answer, context=context)
    assert result.answer_sentences == 2
    assert result.grounded_sentences == 1
    assert result.groundedness == 0.5


def test_ungrounded_answer():
    result = _evaluate(answer="Quantum submarines navigate agricultural oceans.")
    assert result.groundedness == 0.0
    assert result.grounded_sentences == 0


def test_every_sentence_is_scored_independently():
    context = _context("Chunking splits a document into smaller pieces.")
    answer = "Chunking splits a document. A queue buffers messages. Into smaller pieces."
    result = _evaluate(answer=answer, context=context)
    assert result.answer_sentences == 3
    assert result.grounded_sentences == 2  # the middle sentence shares nothing


def test_empty_context_makes_everything_ungrounded():
    result = _evaluate(answer=REFERENCE, context=Context(text="", chunks=()))
    assert result.groundedness == 0.0
    assert result.citation_coverage == 0.0


def test_empty_answer_conventions():
    result = _evaluate(answer="", context=_context())
    assert result.has_answer is False
    assert result.answer_sentences == 0
    assert result.groundedness == 0.0
    assert result.citation_coverage == 0.0
    assert result.reference_coverage == 0.0  # no tokens matched, no division by zero


def test_whitespace_only_answer_counts_as_no_answer():
    result = _evaluate(answer="   \n\t ")
    assert result.has_answer is False
    assert result.answer_sentences == 0
    assert result.groundedness == 0.0


def test_threshold_is_applied_to_sentence_coverage():
    context = _context("a b c d")  # 4 tokens
    answer = "a b x y"  # 2 of 4 tokens supported -> 0.5
    assert _evaluate(answer=answer, context=context).groundedness == 1.0
    assert (
        _evaluate(answer=answer, context=context, groundedness_threshold=0.75).groundedness
        == 0.0
    )


# --- Citation / source coverage ------------------------------------------------


def test_no_sources():
    result = _evaluate(answer=REFERENCE)
    assert result.has_sources is False
    assert result.source_count == 0
    assert result.citation_coverage == 0.0


def test_one_and_many_sources_are_counted():
    one = _evaluate(answer=REFERENCE, sources=(_citation(CONTEXT_TEXT),))
    assert (one.has_sources, one.source_count) == (True, 1)
    many = _evaluate(
        answer=REFERENCE,
        sources=(_citation(CONTEXT_TEXT, "doc::0"), _citation(CONTEXT_TEXT, "doc::1")),
    )
    assert (many.has_sources, many.source_count) == (True, 2)


def test_citation_coverage_is_measured_against_the_sources_not_the_context():
    # The context is empty, so nothing is grounded -- but the cited source does
    # support the answer, which the separate citation measure records.
    result = _evaluate(
        answer="An eviction policy removes an entry.",
        context=Context(text="", chunks=()),
        sources=(_citation("An eviction policy removes an entry."),),
    )
    assert result.groundedness == 0.0
    assert result.citation_coverage == 1.0
    assert result.cited_sentences == 1


def test_citation_coverage_is_zero_when_sources_do_not_support_the_answer():
    result = _evaluate(
        answer="An eviction policy removes an entry.",
        sources=(_citation("Quantum submarines navigate oceans."),),
    )
    assert result.has_sources is True
    assert result.citation_coverage == 0.0


def test_source_coverage_is_sentence_level_not_claim_level():
    # Both sentences are covered by the union of the two sources; the evaluation
    # cannot and does not say which source supports which sentence.
    sources = (
        _citation("An eviction policy removes an entry.", "doc::0"),
        _citation("A cache stores recently used data.", "doc::1"),
    )
    result = _evaluate(
        answer="An eviction policy removes an entry. A cache stores recently used data.",
        sources=sources,
    )
    assert result.citation_coverage == 1.0
    assert set(result.to_dict()) >= {"citation_coverage", "source_count", "cited_sentences"}


# --- Negative cases: the three measures are independent ------------------------


def test_negative_case_b_wrong_but_grounded():
    """Groundedness does not imply correctness."""
    context = _context("The query planner chooses which index to use based on selectivity.")
    result = _evaluate(
        answer="The query planner chooses which index to use based on selectivity.",
        reference="A B-tree index keeps keys sorted and balanced.",
        context=context,
    )
    assert result.groundedness == 1.0
    # Only "index" is shared with the reference ("B-tree" tokenizes as b + tree,
    # so the reference has 9 unique tokens): fully grounded, but it does not
    # answer the reference question.
    assert result.matched_reference_tokens == 1
    assert result.reference_tokens == 9
    assert result.reference_coverage == pytest.approx(1 / 9)


def test_negative_case_b2_negation_is_invisible_to_both_measures():
    """The documented limitation: one flipped word changes nothing."""
    context = _context("LRU evicts the least recently used entry.")
    truth = _evaluate(
        answer="LRU evicts the least recently used entry.",
        context=context,
        reference="LRU evicts the least recently used entry.",
    )
    negated = _evaluate(
        answer="LRU never evicts the least recently used entry.",
        context=context,
        reference="LRU evicts the least recently used entry.",
    )
    # One flipped word, and both baselines report the identical score.
    assert truth.groundedness == negated.groundedness == 1.0
    assert truth.reference_coverage == negated.reference_coverage == 1.0


def test_negative_case_c_correct_but_ungrounded():
    """Correctness does not imply groundedness."""
    result = _evaluate(
        answer="Quantum submarines navigate using agricultural sensor arrays.",
        reference="Quantum submarines navigate using agricultural sensor arrays.",
        context=_context("A cache stores recently used data."),
    )
    assert result.reference_coverage == 1.0
    assert result.groundedness == 0.0


def test_negative_case_d_correct_and_cited():
    result = _evaluate(
        answer=REFERENCE,
        context=_context(),
        sources=(_citation(CONTEXT_TEXT),),
    )
    assert result.reference_coverage == 1.0
    assert result.has_sources is True
    assert result.source_count == 1


def test_negative_case_e_correct_and_uncited():
    result = _evaluate(answer=REFERENCE, context=_context(), sources=())
    assert result.reference_coverage == 1.0
    assert result.has_sources is False
    assert result.source_count == 0


def test_negative_case_a_correct_and_grounded():
    sentence = "An eviction policy decides which entry to remove."
    result = _evaluate(
        answer=sentence,
        reference=sentence,
        context=_context(sentence),
        sources=(_citation(sentence),),
    )
    assert result.reference_coverage == 1.0
    assert result.groundedness == 1.0
    assert result.citation_coverage == 1.0


# --- Aggregation ---------------------------------------------------------------


def test_aggregate_over_evaluations():
    good = _evaluate(answer=REFERENCE, sources=(_citation(CONTEXT_TEXT),), query_id="a")
    bad = _evaluate(answer="", query_id="b")
    aggregate = aggregate_evaluations([good, bad])
    assert list(aggregate) == [
        "mean_reference_coverage",
        "mean_groundedness",
        "mean_citation_coverage",
        "answer_rate",
        "source_presence_rate",
        "mean_source_count",
    ]
    assert aggregate["mean_reference_coverage"] == pytest.approx(1.0 / 2)
    assert aggregate["answer_rate"] == 0.5
    assert aggregate["source_presence_rate"] == 0.5
    assert aggregate["mean_source_count"] == 0.5


def test_aggregate_of_nothing_is_zeroed_not_nan():
    aggregate = aggregate_evaluations([])
    assert all(value == 0.0 for value in aggregate.values())
    assert json.dumps(aggregate)  # JSON-safe


def test_aggregate_accepts_any_iterable():
    result = _evaluate(answer=REFERENCE)
    assert aggregate_evaluations(iter([result])) == aggregate_evaluations([result])


# --- public API ----------------------------------------------------------------


def test_public_import():
    from zmwiki import __all__

    expected = {"AnswerEvaluation", "AnswerEvaluator", "aggregate_evaluations"}
    assert expected <= set(__all__)
