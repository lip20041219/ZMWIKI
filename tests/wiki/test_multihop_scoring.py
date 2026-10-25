"""Evaluation-correctness tests for the multi-hop benchmark (Phase 08).

Covers the scoring distinctions the benchmark must make: grounded vs
keyword-only answers, terminal-status accounting, source id resolution for
nested paths, cited-source deduplication, and tolerant action parsing for the
real-model failure mode where a valid action object is followed by prose.
"""

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

_BENCHMARK_DIR = Path(__file__).resolve().parents[2] / "benchmark"
sys.path.insert(0, str(_BENCHMARK_DIR))

import rescore_multihop_results as rs  # noqa: E402
import run_multihop_evaluation as m4  # noqa: E402
from zmwiki.wiki.wiki_agent import AgentProtocolError, _parse_action  # noqa: E402
from zmwiki.wiki.wiki_index import WikiIndex  # noqa: E402
from zmwiki.wiki.wiki_schema import WikiPage  # noqa: E402


def _case(case_id, *, answerable=True, hops=2, required=("a",)):
    return {
        "id": case_id,
        "question": "q",
        "answerable": answerable,
        "hops": hops,
        "required_page_ids": list(required),
        "answer_keyword_groups": [["x"]],
        "refusal_keywords": ["not confirm"],
    }


def _result(
    case_id, *, correct, grounded, status, recall=1.0, sources=("a",),
):
    return m4.CaseResult(
        id=case_id,
        question="q",
        answer="x" if correct else "wrong",
        answer_correct=correct,
        status=status,
        sources=list(sources),
        accessed_pages=["a"],
        required_pages=["a"],
        evidence_recall=recall,
        source_precision=1.0 if sources else 0.0,
        steps=1,
        pages_read=1 if recall else 0,
        grounded_correct=grounded,
        unsupported_correct=correct and not grounded,
    )


def test_aggregate_separates_grounded_keyword_and_terminal_failures():
    cases = [_case(f"c{i}") for i in range(6)]
    results = [
        _result("c0", correct=True, grounded=True, status="answered"),
        _result("c1", correct=True, grounded=False,
                status="insufficient_evidence", sources=()),
        _result("c2", correct=False, grounded=False,
                status="budget_exhausted", recall=0.0, sources=()),
        _result("c3", correct=False, grounded=False,
                status="protocol_error", recall=0.0, sources=()),
        _result("c4", correct=False, grounded=False,
                status="model_error", recall=0.0, sources=()),
        _result("c5", correct=False, grounded=False, status="answered"),
    ]

    agg = m4.aggregate_results(cases, results)

    assert agg["answer_accuracy"] == pytest.approx(2 / 6)  # legacy keyword score
    assert agg["verified_answer_accuracy"] == pytest.approx(1 / 6)
    assert agg["unsupported_correct_count"] == 1
    assert agg["status_counts"] == {
        "answered": 2,
        "budget_exhausted": 1,
        "insufficient_evidence": 1,
        "model_error": 1,
        "protocol_error": 1,
    }
    assert agg["status_rates"] == {
        "insufficient_evidence": pytest.approx(1 / 6),
        "budget_exhausted": pytest.approx(1 / 6),
        "protocol_error": pytest.approx(1 / 6),
        "model_error": pytest.approx(1 / 6),
    }
    assert agg["mean_evidence_recall"] == pytest.approx(3 / 6)
    legacy_support = agg["retrieval_evidence_metrics"][
        "evidence_support_rate_answerable"
    ]
    assert legacy_support["value"] == pytest.approx(2 / 6)
    assert agg["verified_multi_hop_success_rate"] == pytest.approx(1 / 6)


def test_grounding_flags_are_serialized():
    result = _result("c0", correct=True, grounded=False,
                     status="insufficient_evidence", sources=())
    payload = result.to_dict()

    assert payload["grounded_correct"] is False
    assert payload["unsupported_correct"] is True


def test_source_ids_resolve_nested_paths_and_dedupe():
    index = WikiIndex()
    index.upsert(WikiPage(
        id="wikipedia-11675",
        path="wikipedia/11675",
        title="Alan Turing",
        content="body",
    ))

    ids = m4._source_ids_from_paths(
        ("wikipedia/11675", "wikipedia/11675", "missing/path"), index
    )

    assert ids == ["wikipedia-11675", "path"]


def test_baseline_citations_are_deduplicated():
    class _Pipeline:
        def answer_with_sources(self, query, **kwargs):
            return SimpleNamespace(
                answer="x",
                sources=[
                    SimpleNamespace(chunk_id="a::0"),
                    SimpleNamespace(chunk_id="a::1"),
                    SimpleNamespace(chunk_id="b::0"),
                ],
            )

        def query(self, query, **kwargs):
            return SimpleNamespace(chunks=[
                SimpleNamespace(id="a::0"),
                SimpleNamespace(id="b::0"),
            ])

    results = m4.run_baseline([_case("c0", hops=1)], _Pipeline())

    assert results[0].sources == ["a", "b"]
    assert results[0].grounded_correct is True
    # Deduplicated: |{a,b} ∩ {a}| / 2 == 0.5 (the old raw list gave 1/3).
    assert results[0].source_precision == 0.5


# ---------------------------------------------------------------------------
# Phase 09 false-negative regressions (real dataset annotations + real answers)
# ---------------------------------------------------------------------------


def _phase09_case(case_id):
    dataset = {c["id"]: c for c in m4.load_dataset()["cases"]}
    return dataset[case_id]


def test_mh05_numeric_total_budget_is_scored_correct():
    case = _phase09_case("mh05_total_budget_aggregation")
    answer = (
        "Carol Chen works on two projects: Project Phoenix (budget $500,000) "
        "and Project Orion (budget $300,000). The total budget of all projects "
        "she works on is $800,000."
    )
    assert m4.score_answer(answer, case) is True


def test_mh08_grounded_refusal_is_counted_as_handled():
    case = _phase09_case("mh08_ceo_unanswerable")
    answer = (
        "I could not confirm who the CEO of Acme Corp is. The pages I read "
        "mention that Building B's fifth floor contains the CEO's office and "
        "that Eve Martinez (VP of Engineering) reports directly to the CEO, but "
        "no page names the CEO."
    )
    assert m4.score_answer(answer, case) is True

    result = _result(
        "mh08_ceo_unanswerable", correct=True, grounded=True,
        status="answered", recall=None, sources=("building_b",),
    )
    agg = m4.aggregate_results([case], [result])
    assert agg["unanswerable_handling"] == 1.0


def test_rescore_payload_recomputes_flags_and_aggregates():
    case5 = _phase09_case("mh05_total_budget_aggregation")
    case8 = _phase09_case("mh08_ceo_unanswerable")
    result5 = _result("mh05_total_budget_aggregation", correct=False,
                      grounded=False, status="answered")
    result5.answer = (
        "Carol Chen works on Project Phoenix ($500,000) and Project Orion "
        "($300,000). The total is $800,000."
    )
    result8 = _result("mh08_ceo_unanswerable", correct=False, grounded=False,
                      status="answered", recall=None, sources=("building_b",))
    result8.answer = "I could not confirm who the CEO is; no page names the CEO."
    payload = {
        "dataset_sha256_16": "old",
        "results": {"experiment": {
            "per_case": [result5.to_dict(), result8.to_dict()],
            "aggregate": None,
        }},
    }

    rescored = rs.rescore_payload(
        payload,
        {"cases": [case5, case8], "scoring_rule": "keyword_group_match"},
    )

    assert {c["id"] for c in rescored["rescoring"]["changes"]["experiment"]} == {
        "mh05_total_budget_aggregation", "mh08_ceo_unanswerable",
    }
    agg = rescored["results"]["experiment"]["aggregate"]
    assert agg["answerable_accuracy"] == 1.0
    assert agg["unanswerable_handling"] == 1.0
    assert payload["results"]["experiment"]["aggregate"] is None  # input untouched


def test_action_parser_ignores_trailing_model_text():
    raw = (
        '{"action": "read", "paths": ["wiki/bob"]}\n\n'
        "<response>\n"
        '{"content": "Bob is the Director of Engineering."}'
    )

    assert _parse_action(raw) == {"action": "read", "paths": ["wiki/bob"]}


def test_action_parser_still_rejects_incomplete_or_invalid_output():
    with pytest.raises(AgentProtocolError):
        _parse_action('{"action": "read", "paths": [')
    with pytest.raises(AgentProtocolError):
        _parse_action('{"action": "unknown"}\n trailing')
    with pytest.raises(AgentProtocolError):
        _parse_action("[1, 2, 3]")
