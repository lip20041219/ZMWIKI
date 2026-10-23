"""Claim attribution tests: extraction, support, attribution, and its limits.

Hermetic and dependency-free: no model, no network, no LLM judge. The point of
these tests is the *separation* -- a claim can be supported by the context and
unattributed, attributed to a source it contradicts, or unsupported entirely,
and the two rates must never collapse into one.
"""

import dataclasses
import json

import pytest

from zmwiki import (
    AnswerEvaluator,
    AttributionResult,
    Citation,
    Claim,
    ClaimAttribution,
    ClaimAttributor,
    ClaimExtractor,
    Context,
    DeterministicClaimAttributor,
    DeterministicClaimExtractor,
)
from zmwiki.text import (
    normalize_text,
    safe_ratio,
    split_sentences,
    token_coverage,
    unique_tokens,
)

LRU_SOURCE = "LRU evicts the least recently used entry and is simple to reason about."
LRU_CLAIM = "LRU evicts the least recently used entry."


def _citation(text, chunk_id="doc::0", position=0):
    return Citation(
        document_id="doc", chunk_id=chunk_id, position=position, text=text
    )


def _context(text, chunk_id="doc::0"):
    from zmwiki import Chunk

    chunk = Chunk(id=chunk_id, document_id="doc", text=text or "x", position=0)
    return Context(text=text, chunks=(chunk,))


def _claims(answer):
    return DeterministicClaimExtractor().extract(answer)


def _attribute(answer, *, context=None, citations=(), **kwargs):
    attributor = DeterministicClaimAttributor(**kwargs)
    return attributor.attribute(
        _claims(answer),
        context=context if context is not None else _context(""),
        citations=citations,
    )


# --- Shared text helpers -------------------------------------------------------


def test_split_sentences_handles_both_scripts():
    assert split_sentences("One. Two! Three?") == ["One.", "Two!", "Three?"]
    assert split_sentences("苹果派好吃。它很甜！真的吗？") == [
        "苹果派好吃。",
        "它很甜！",
        "真的吗？",
    ]


def test_split_sentences_keeps_punctuation_but_drops_newlines():
    assert split_sentences("line one\nline two") == ["line one", "line two"]
    assert split_sentences("a. b") == ["a.", "b"]


def test_split_sentences_ignores_empty_pieces():
    assert split_sentences("a. . b") == ["a.", "b"]
    assert split_sentences("") == []
    assert split_sentences("   \n\t ") == []


def test_normalize_text_makes_composed_and_decomposed_forms_agree():
    composed = "café"
    decomposed = "café"
    assert composed != decomposed
    assert normalize_text(composed) == normalize_text(decomposed)


def test_unique_tokens_keeps_first_occurrence_order():
    assert unique_tokens(["b", "a", "b", "c", "a"]) == ["b", "a", "c"]


def test_token_coverage_is_a_safe_ratio():
    assert token_coverage(["a", "b", "c", "d"], ["a", "b"]) == 0.5
    assert token_coverage(["a", "a", "b"], ["a"]) == 0.5  # duplicates count once
    assert token_coverage([], ["a"]) == 0.0
    assert token_coverage(["a"], []) == 0.0
    assert safe_ratio(1, 0) == 0.0
    assert safe_ratio(0, 4) == 0.0


# --- Claim extraction ----------------------------------------------------------


def test_extractor_is_a_claim_extractor():
    assert issubclass(DeterministicClaimExtractor, ClaimExtractor)
    with pytest.raises(TypeError):
        ClaimExtractor()  # type: ignore[abstract]


def test_empty_answer_yields_no_claims():
    assert _claims("") == ()
    assert _claims("   \n\t ") == ()


def test_one_sentence_is_one_claim():
    claims = _claims(LRU_CLAIM)
    assert len(claims) == 1
    assert claims[0].claim_id == "claim-0"
    assert claims[0].sentence_index == 0
    assert claims[0].text == LRU_CLAIM  # terminator kept


def test_multiple_sentences_become_indexed_claims():
    claims = _claims("Redis is in memory. It also persists. It is fast!")
    assert [c.claim_id for c in claims] == ["claim-0", "claim-1", "claim-2"]
    assert [c.sentence_index for c in claims] == [0, 1, 2]
    assert claims[2].text == "It is fast!"


def test_a_sentence_without_a_terminator_is_still_a_claim():
    claims = _claims("no terminator here")
    assert len(claims) == 1
    assert claims[0].text == "no terminator here"


def test_mixed_chinese_and_english_splits_on_both():
    claims = _claims("Redis is in memory. 它支持持久化！Why not?")
    assert [c.text for c in claims] == ["Redis is in memory.", "它支持持久化！", "Why not?"]


def test_extractor_rejects_non_strings():
    with pytest.raises(TypeError):
        DeterministicClaimExtractor().extract(42)  # type: ignore[arg-type]


def test_claim_validates_its_fields():
    Claim(claim_id="c", text="x", sentence_index=0)
    with pytest.raises(TypeError):
        Claim(claim_id="", text="x", sentence_index=0)
    with pytest.raises(TypeError):
        Claim(claim_id="c", text="  ", sentence_index=0)
    with pytest.raises(TypeError):
        Claim(claim_id="c", text="x", sentence_index="0")  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        Claim(claim_id="c", text="x", sentence_index=-1)


def test_claim_is_frozen_and_json_ready():
    claim = _claims(LRU_CLAIM)[0]
    with pytest.raises(dataclasses.FrozenInstanceError):
        claim.text = "other"  # type: ignore[misc]
    assert json.loads(json.dumps(claim.to_dict()))["claim_id"] == "claim-0"


# --- Attribution: supported vs attributed --------------------------------------


def test_attributor_is_a_claim_attributor():
    assert issubclass(DeterministicClaimAttributor, ClaimAttributor)
    with pytest.raises(TypeError):
        ClaimAttributor()  # type: ignore[abstract]


def test_attributed_claim_with_a_covering_citation():
    result = _attribute(
        LRU_CLAIM, context=_context(LRU_SOURCE), citations=(_citation(LRU_SOURCE, "doc::1"),)
    )
    claim = result.claims[0]
    assert claim.supported is True
    assert claim.attributed is True
    assert claim.supporting_citation_ids == ("doc::1",)
    assert claim.support_score == 1.0


def test_supported_but_unattributed_when_there_are_no_citations():
    result = _attribute(LRU_CLAIM, context=_context(LRU_SOURCE))
    claim = result.claims[0]
    assert claim.supported is True
    assert claim.attributed is False
    assert claim.supporting_citation_ids == ()
    assert result.supported_claim_rate == 1.0
    assert result.attributed_claim_rate == 0.0


def test_supported_but_unattributed_with_a_wrong_citation():
    result = _attribute(
        LRU_CLAIM,
        context=_context(LRU_SOURCE),
        citations=(_citation("Quantum submarines navigate agricultural oceans.", "doc::9"),),
    )
    claim = result.claims[0]
    assert claim.supported is True
    assert claim.attributed is False
    assert claim.supporting_citation_ids == ()


def test_unsupported_claim():
    result = _attribute(
        "Quantum submarines navigate agricultural oceans.",
        context=_context(LRU_SOURCE),
        citations=(_citation(LRU_SOURCE, "doc::1"),),
    )
    claim = result.claims[0]
    assert claim.supported is False
    assert claim.attributed is False
    assert claim.support_score == 0.0


def test_support_score_is_partial_coverage_not_a_boolean():
    result = _attribute("LRU evicts the entry.", context=_context("LRU evicts an item."))
    # unique claim tokens: lru, evicts, the, entry -- two of them are in the context
    assert result.claims[0].support_score == pytest.approx(2 / 4)
    assert result.claims[0].supported is True  # 0.5 still clears the threshold


def test_multiple_supporting_citations_are_all_kept_in_input_order():
    first = _citation(LRU_SOURCE, "doc::1")
    second = _citation("LRU evicts the least recently used entry.", "doc::2")
    third = _citation("unrelated text about agriculture", "doc::3")
    result = _attribute(LRU_CLAIM, context=_context(LRU_SOURCE), citations=(first, second, third))
    assert result.claims[0].supporting_citation_ids == ("doc::1", "doc::2")
    # Input order, never re-sorted by score.
    reversed_result = _attribute(
        LRU_CLAIM, context=_context(LRU_SOURCE), citations=(second, first)
    )
    assert reversed_result.claims[0].supporting_citation_ids == ("doc::2", "doc::1")


def test_duplicate_citations_collapse_to_their_first_occurrence():
    citation = _citation(LRU_SOURCE, "doc::1")
    result = _attribute(
        LRU_CLAIM,
        context=_context(LRU_SOURCE),
        citations=(citation, citation, citation),
    )
    assert result.claims[0].supporting_citation_ids == ("doc::1",)


def test_threshold_boundary_is_inclusive():
    context = _context("a b c d")
    answer = "a b x y."  # exactly half the tokens are covered
    assert _attribute(answer, context=context).claims[0].support_score == 0.5
    assert _attribute(answer, context=context, threshold=0.5).claims[0].supported is True
    assert _attribute(answer, context=context, threshold=0.51).claims[0].supported is False
    assert _attribute(answer, context=context, threshold=1.0).claims[0].supported is False


def test_attribution_threshold_is_validated():
    assert DeterministicClaimAttributor().threshold == 0.5
    assert DeterministicClaimAttributor(threshold=1).threshold == 1.0
    for bad in (0, -0.5, 1.5):
        with pytest.raises(ValueError):
            DeterministicClaimAttributor(threshold=bad)
    with pytest.raises(TypeError):
        DeterministicClaimAttributor(threshold="0.5")  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        DeterministicClaimAttributor(tokenizer="nope")  # type: ignore[arg-type]


def test_attributor_inputs_are_validated():
    claims = _claims(LRU_CLAIM)
    attributor = DeterministicClaimAttributor()
    with pytest.raises(TypeError):
        attributor.attribute(claims, context="not a context")  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        attributor.attribute([("not", "a claim")], context=_context(LRU_SOURCE))  # type: ignore[list-item]
    with pytest.raises(TypeError):
        attributor.attribute(claims, context=_context(LRU_SOURCE), citations="nope")  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        attributor.attribute(  # type: ignore[arg-type]
            claims, context=_context(LRU_SOURCE), citations=("not a citation",)
        )


def test_empty_context_and_no_citations_never_produce_nan():
    result = _attribute(LRU_CLAIM, context=Context(text="", chunks=()))
    claim = result.claims[0]
    assert claim.support_score == 0.0
    assert claim.supported is False
    assert result.mean_support_score == 0.0
    assert result.supported_claim_rate == 0.0
    assert result.attributed_claim_rate == 0.0


def test_empty_answer_attributes_nothing_without_dividing_by_zero():
    result = _attribute("", context=_context(LRU_SOURCE), citations=(_citation(LRU_SOURCE),))
    assert result.total_claims == 0
    assert result.claims == ()
    assert result.supported_claim_rate == 0.0
    assert result.attributed_claim_rate == 0.0
    assert result.mean_support_score == 0.0


def test_claim_with_no_tokens_scores_zero_rather_than_nan():
    claim = Claim(claim_id="claim-0", text="!!!", sentence_index=0)
    result = DeterministicClaimAttributor().attribute(
        [claim], context=_context(LRU_SOURCE), citations=(_citation(LRU_SOURCE),)
    )
    assert result.claims[0].support_score == 0.0
    assert result.claims[0].supported is False


# --- Multi-claim answers -------------------------------------------------------


def test_each_claim_is_scored_independently():
    answer = "LRU evicts the least recently used entry. Quantum submarines navigate oceans."
    result = _attribute(
        answer, context=_context(LRU_SOURCE), citations=(_citation(LRU_SOURCE, "doc::1"),)
    )
    assert result.total_claims == 2
    assert result.claims[0].supported is True
    assert result.claims[0].attributed is True
    assert result.claims[1].supported is False
    assert result.claims[1].attributed is False
    assert result.supported_claims == 1
    assert result.attributed_claims == 1
    assert result.supported_claim_rate == 0.5
    assert result.attributed_claim_rate == 0.5
    assert result.mean_support_score == pytest.approx((1.0 + 0.0) / 2)


def test_a_second_claim_can_be_supported_by_the_context_but_uncited():
    answer = "LRU evicts the least recently used entry. The hit rate guides tuning."
    context = _context(LRU_SOURCE + " Monitoring the hit rate guides tuning.")
    result = _attribute(answer, context=context, citations=(_citation(LRU_SOURCE, "doc::1"),))
    assert result.claims[0].attributed is True
    assert result.claims[1].supported is True
    assert result.claims[1].attributed is False


# --- Language, normalization, determinism --------------------------------------


def test_chinese_claims_are_attributed_at_character_granularity():
    result = _attribute(
        "缓存会淘汰最少使用的条目。",
        context=_context("缓存会淘汰最少使用的条目。"),
        citations=(_citation("缓存会淘汰最少使用的条目。", "doc::1"),),
    )
    claim = result.claims[0]
    assert claim.supported is True
    assert claim.attributed is True
    assert claim.support_score == 1.0


def test_mixed_language_claims_work():
    answer = "Redis is in memory. 它支持持久化。"
    context = _context("Redis is in memory. 它支持持久化。")
    result = _attribute(answer, context=context, citations=(_citation(context.text, "doc::1"),))
    assert result.total_claims == 2
    assert result.supported_claim_rate == 1.0
    assert result.attributed_claim_rate == 1.0


def test_unicode_normalization_does_not_break_matching():
    context = _context("café crème is served")
    result = _attribute(
        "café crème is served.",
        context=context,
        citations=(_citation(context.text, "doc::1"),),
    )
    assert result.claims[0].supported is True
    assert result.claims[0].attributed is True


def test_whitespace_and_case_and_punctuation_do_not_matter():
    context = _context(LRU_SOURCE)
    variant = _attribute("  LRU   EVICTS the least recently used entry!!!  ", context=context)
    plain = _attribute(LRU_CLAIM, context=context)
    assert variant.claims[0].support_score == plain.claims[0].support_score == 1.0


def test_attribution_is_deterministic_across_calls_and_instances():
    first = _attribute(LRU_CLAIM, context=_context(LRU_SOURCE), citations=(_citation(LRU_SOURCE),))
    second = _attribute(LRU_CLAIM, context=_context(LRU_SOURCE), citations=(_citation(LRU_SOURCE),))
    assert first == second
    assert first.to_dict() == second.to_dict()
    assert first.summary() == second.summary()


def test_result_is_frozen_and_json_serializable():
    result = _attribute(LRU_CLAIM, context=_context(LRU_SOURCE))
    with pytest.raises(dataclasses.FrozenInstanceError):
        result.total_claims = 5  # type: ignore[misc]
    payload = json.dumps(result.to_dict())
    assert '"claims"' in payload
    assert result.summary().startswith("## Claim Attribution")


# --- The documented limitation -------------------------------------------------


def test_negation_is_attributed_to_the_source_it_contradicts():
    """Limitation evidence, not a bug to fix: overlap cannot see the flip."""
    truth = _attribute(LRU_CLAIM, context=_context(LRU_SOURCE), citations=(_citation(LRU_SOURCE),))
    negated = _attribute(
        "LRU never evicts the least recently used entry.",
        context=_context(LRU_SOURCE),
        citations=(_citation(LRU_SOURCE),),
    )
    assert negated.claims[0].supported is True
    assert negated.claims[0].attributed is True
    assert negated.claims[0].support_score == pytest.approx(7 / 8)  # "never" is the only miss
    assert truth.claims[0].support_score == 1.0


def test_a_paraphrase_with_different_words_scores_low():
    """The other side of the same limitation: correct, differently worded, unseen."""
    result = _attribute(
        "The oldest entry gets thrown out.",
        context=_context(LRU_SOURCE),
        citations=(_citation(LRU_SOURCE, "doc::1"),),
    )
    assert result.claims[0].supported is False
    assert result.claims[0].attributed is False


# --- Value validation ----------------------------------------------------------


def test_claim_attribution_validates_its_fields():
    ClaimAttribution(claim_id="c", supported=True, supporting_citation_ids=("a",), support_score=0.5)
    with pytest.raises(ValueError):
        ClaimAttribution(claim_id="c", supported=True, supporting_citation_ids=(), support_score=1.5)
    with pytest.raises(TypeError):
        ClaimAttribution(claim_id="c", supported="yes", supporting_citation_ids=(), support_score=0.5)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        ClaimAttribution(claim_id="c", supported=True, supporting_citation_ids=["a"], support_score=0.5)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        ClaimAttribution(claim_id="c", supported=True, supporting_citation_ids=("a", "a"), support_score=0.5)
    with pytest.raises(TypeError):
        ClaimAttribution(claim_id="c", supported=True, supporting_citation_ids=("",), support_score=0.5)


def test_attribution_result_validates_internal_consistency():
    good = ClaimAttribution(
        claim_id="c", supported=True, supporting_citation_ids=(), support_score=1.0
    )
    base = dict(
        claims=(good,),
        supported_claim_rate=1.0,
        attributed_claim_rate=0.0,
        total_claims=1,
        supported_claims=1,
        attributed_claims=0,
        mean_support_score=1.0,
    )
    AttributionResult(**base)
    with pytest.raises(ValueError):
        AttributionResult(**{**base, "total_claims": 3})  # disagrees with the list
    with pytest.raises(ValueError):
        AttributionResult(**{**base, "supported_claims": 0})  # disagrees with the claim
    with pytest.raises(ValueError):
        AttributionResult(**{**base, "supported_claim_rate": 0.4})  # disagrees with the count
    with pytest.raises(ValueError):
        AttributionResult(**{**base, "mean_support_score": 2.0})
    with pytest.raises(TypeError):
        AttributionResult(**{**base, "claims": [good]})  # a list is not a tuple
    with pytest.raises(TypeError):
        AttributionResult(**{**base, "claims": ("not a claim",)})


# --- Integration with the answer evaluator -------------------------------------


def test_answer_evaluator_delegates_to_the_attribution_layer():
    evaluator = AnswerEvaluator()
    result = evaluator.evaluate_attribution(
        answer=LRU_CLAIM,
        context=_context(LRU_SOURCE),
        sources=(_citation(LRU_SOURCE, "doc::1"),),
    )
    assert isinstance(result, AttributionResult)
    assert result.attributed_claim_rate == 1.0
    # Same tokenizer path as the evaluator's own measures, so they agree.
    assert result.claims[0].support_score == 1.0


def test_answer_evaluator_attribution_inputs_are_validated():
    evaluator = AnswerEvaluator()
    with pytest.raises(TypeError):
        evaluator.evaluate_attribution(answer=42, context=_context(LRU_SOURCE))  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        evaluator.evaluate_attribution(answer=LRU_CLAIM, context="nope")  # type: ignore[arg-type]


def test_existing_answer_evaluation_api_is_unchanged():
    # The attribution layer is additive: the three lexical rates still work.
    evaluation = AnswerEvaluator().evaluate(
        query_id="q",
        query="what evicts?",
        answer=LRU_CLAIM,
        context=_context(LRU_SOURCE),
        reference_answer=LRU_CLAIM,
        sources=(_citation(LRU_SOURCE, "doc::1"),),
    )
    assert evaluation.reference_coverage == 1.0
    assert evaluation.groundedness == 1.0
    assert evaluation.citation_coverage == 1.0
    assert "supported_claim_rate" not in evaluation.to_dict()


def test_public_import():
    from zmwiki import __all__

    expected = {
        "AttributionResult",
        "Claim",
        "ClaimAttribution",
        "ClaimAttributor",
        "ClaimExtractor",
        "DeterministicClaimAttributor",
        "DeterministicClaimExtractor",
    }
    assert expected <= set(__all__)


def test_attribution_module_parses_no_citation_markers():
    import ast
    import pathlib

    import zmwiki.attribution as module

    source = pathlib.Path(module.__file__).read_text(encoding="utf-8")
    # No marker grammar anywhere -- not even a regex import: the only text the
    # attributor reads is Citation.text, reached through the Citation objects.
    assert "import re" not in source
    assert "re." not in source.replace("re. ", "")  # prose like "re. sentence" aside
    tree = ast.parse(source)
    imported = {
        alias.name.split(".")[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    } | {
        (node.module or "").split(".")[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and not node.level
    }
    assert imported <= {"__future__", "abc", "collections", "dataclasses"}
