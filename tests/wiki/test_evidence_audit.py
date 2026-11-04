"""Tests for the offline EvidenceAudit v0 module."""

import json
import sys
from pathlib import Path

import pytest

_BENCHMARK_DIR = Path(__file__).resolve().parents[2] / "benchmark"
sys.path.insert(0, str(_BENCHMARK_DIR))

import evidence_audit as ea  # noqa: E402
import run_multihop_evaluation as m4  # noqa: E402


@pytest.fixture(scope="module")
def pages():
    return {p.id: p.content for p in m4.build_wiki_pages(m4.load_documents())}


@pytest.fixture(scope="module")
def audit_cases():
    path = _BENCHMARK_DIR / "datasets" / "evidence_audit_v0_cases.json"
    return json.loads(path.read_text(encoding="utf-8"))["cases"]


def _audit(pages, answer, sources, read=None):
    return ea.audit_answer(answer, sources, read if read is not None else sources, pages)


def test_correct_entity_relation_supported(pages):
    result = _audit(pages, "Alice Smith reports directly to Bob Jones.", ["alice", "bob"])
    assert result.evidence_verdict == "supported"
    assert result.claims[0].verdict == "supported"
    assert result.claim_support_rate == 1.0


def test_relation_reversal_contradicted(pages):
    result = _audit(pages, "Bob Jones reports directly to Alice Smith.", ["bob", "alice", "eve"])
    assert result.evidence_verdict == "contradicted"
    assert result.claims[0].verdict == "contradicted"


def test_wrong_person_attribution_contradicted(pages):
    result = _audit(
        pages,
        "Technical roadmap planning is owned by Eve Martinez.",
        ["bob", "eve"],
    )
    assert result.evidence_verdict == "contradicted"
    assert result.claims[0].verdict == "contradicted"


def test_wrong_number_contradicted(pages):
    result = _audit(pages, "Project Phoenix has a budget of $900,000.", ["project_phoenix"])
    assert result.evidence_verdict == "contradicted"
    assert result.claims[0].verdict == "contradicted"


def test_negation_conflict_contradicted(pages):
    result = _audit(
        pages,
        "Eve Martinez does not own the annual engineering budget.",
        ["eve"],
    )
    assert result.evidence_verdict == "contradicted"
    assert result.claims[0].negated is True
    assert result.claims[0].verdict == "contradicted"


def test_irrelevant_citation_is_unverifiable(pages):
    result = _audit(pages, "Alice Smith reports directly to Bob Jones.", ["david"])
    assert result.evidence_verdict == "unverifiable"
    assert result.claims[0].verdict == "unverifiable"
    assert result.claims[0].supporting_page_ids == ()
    assert result.claims[0].contradicting_page_ids == ()


def test_multi_page_support_supported(pages):
    result = _audit(
        pages,
        "Alice Smith leads Project Phoenix. Alice Smith reports directly to Bob Jones.",
        ["project_phoenix", "alice", "bob"],
    )
    assert result.evidence_verdict == "supported"
    assert [c.verdict for c in result.claims] == ["supported", "supported"]
    assert result.claim_support_rate == 1.0
    assert result.citation_claim_precision == 1.0


def test_complex_unverifiable_claim(pages):
    result = _audit(pages, "The CEO's name is John Smith.", ["building_b", "eve"])
    assert result.evidence_verdict == "unverifiable"
    assert all(c.verdict == "unverifiable" for c in result.claims)


def test_empty_answer_has_no_claims(pages):
    result = _audit(pages, "", ["alice"], ["alice"])
    assert result.evidence_verdict == "no_answer"
    assert result.claims == ()
    assert result.claim_support_rate == 0.0


def test_no_sources_is_no_evidence(pages):
    result = _audit(pages, "Alice Smith reports directly to Bob Jones.", [], ["alice", "bob"])
    assert result.evidence_verdict == "no_evidence"
    assert result.valid_source_ids == ()


def test_sources_not_actually_read_are_invalid(pages):
    result = _audit(
        pages,
        "Alice Smith reports directly to Bob Jones.",
        ["alice", "bob"],
        read=["bob"],
    )
    assert result.evidence_verdict == "unverifiable"
    assert result.valid_source_ids == ("bob",)
    assert result.invalid_source_ids == ("alice",)
    assert result.claims[0].verdict == "unverifiable"


def test_duplicate_claim_and_source_dedup(pages):
    result = _audit(
        pages,
        "Alice Smith reports directly to Bob Jones. Alice Smith reports directly to Bob Jones.",
        ["alice", "bob", "alice"],
    )
    assert result.evidence_verdict == "supported"
    assert len(result.claims) == 1
    assert result.valid_source_ids == ("alice", "bob")


def test_paraphrase_responsibility_supported(pages):
    result = _audit(pages, "Bob Jones is in charge of technical roadmap planning.", ["bob"])
    assert result.evidence_verdict == "supported"
    assert result.claims[0].relation == "responsible_for"


def test_paraphrase_reports_under_supported(pages):
    result = _audit(pages, "Alice Smith works under Bob Jones.", ["alice", "bob"])
    assert result.evidence_verdict == "supported"
    assert result.claims[0].relation == "reports_to"


def test_paraphrase_membership_supported(pages):
    result = _audit(pages, "Carol Chen is part of Project Phoenix.", ["project_phoenix", "carol"])
    assert result.evidence_verdict == "supported"
    assert result.claims[0].relation == "member_of"


def test_team_size_paraphrase_supported(pages):
    result = _audit(pages, "Bob Jones's team has twelve engineers.", ["bob"])
    assert result.evidence_verdict == "supported"
    assert result.claims[0].relation == "team_size"


def test_direct_report_count_not_conflated_with_team_size(pages):
    result = _audit(pages, "Bob Jones has five direct reports.", ["bob"])
    # Source says team of twelve engineers; direct-report count is a different
    # statistic, so this is unverifiable, not contradicted.
    assert result.evidence_verdict == "unverifiable"
    assert result.claims[0].relation == "direct_report_count"
    assert result.claims[0].verdict == "unverifiable"


def test_director_count_supported(pages):
    result = _audit(pages, "Eve Martinez manages two directors.", ["eve"])
    assert result.evidence_verdict == "supported"
    assert result.claims[0].relation == "director_count"


def test_compound_claim_is_split(pages):
    result = _audit(
        pages,
        "Alice Smith is a senior engineer and reports directly to Bob Jones.",
        ["alice", "bob"],
    )
    relations = {c.relation for c in result.claims}
    assert "role" in relations
    assert "reports_to" in relations
    assert result.evidence_verdict == "supported"


def test_feature_ablation_paraphrase_improves_coverage(pages):
    answer = "Bob Jones is in charge of technical roadmap planning."
    without = ea.audit_answer(answer, ["bob"], ["bob"], pages, features=[])
    with_features = ea.audit_answer(answer, ["bob"], ["bob"], pages)
    assert without.audit_coverage <= with_features.audit_coverage
    assert with_features.evidence_verdict == "supported"


def test_feature_ablation_dimensions_changes_relation(pages):
    answer = "Bob Jones has five direct reports."
    legacy = ea.audit_answer(answer, ["bob"], ["bob"], pages, features=["paraphrase", "clause_split"])
    v02 = ea.audit_answer(answer, ["bob"], ["bob"], pages)
    assert legacy.claims[0].relation == "team_size"
    assert v02.claims[0].relation == "direct_report_count"


def test_unknown_feature_rejected(pages):
    with pytest.raises(ValueError, match="unknown audit features"):
        ea.audit_answer("x", [], [], pages, features=["no_such_feature"])


def test_fixture_cases_match_expected(pages, audit_cases):
    assert len(audit_cases) == 10
    for case in audit_cases:
        result = ea.audit_answer(
            case["answer"], case["sources"], case["read_page_ids"], pages
        )
        assert result.evidence_verdict == case["expected_verdict"], case["id"]
        assert [c.verdict for c in result.claims] == case["expected_claim_verdicts"], case["id"]


def test_historical_three_errors_are_contradicted(pages):
    cases = {
        "mh38": (
            "Technical roadmap planning is owned by Eve Martinez. Quarterly "
            "performance reviews are run by Bob Jones. The annual engineering "
            "budget is also owned by Bob Jones.",
            ["bob", "eve"],
        ),
        "mh10": (
            "Bob Jones has five direct reports: Alice Smith, Carol Chen, "
            "David Kim, Frank Osei, and Grace Liu.",
            ["bob", "alice", "carol", "david"],
        ),
        "mh49": (
            "Project Orion's technical lead is David Kim, who reports to Eve "
            "Martinez. Eve Martinez reports to the Chief Technology Officer "
            "(CTO), who reports to the CEO.",
            ["project_orion", "david", "eve"],
        ),
    }
    for name, (answer, sources) in cases.items():
        result = _audit(pages, answer, sources)
        assert result.evidence_verdict == "contradicted", name
        assert any(c.verdict == "contradicted" for c in result.claims), name
