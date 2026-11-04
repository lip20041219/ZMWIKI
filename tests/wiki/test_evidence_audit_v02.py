"""Offline integration checks for the Stage 15.2 calibration artifact."""

import json
import subprocess
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "benchmark" / "run_evidence_audit_v02.py"
RESULT = ROOT / "benchmark" / "results" / "stage15_2_evidence_audit_v02.json"


@pytest.fixture(scope="module")
def calibration(tmp_path_factory):
    output = tmp_path_factory.mktemp("evidence_audit_v02") / "result.json"
    subprocess.run(
        [sys.executable, str(SCRIPT), "--output", str(output)],
        cwd=ROOT, check=True, capture_output=True, text=True, timeout=60,
    )
    actual = json.loads(output.read_text(encoding="utf-8"))
    assert actual == json.loads(RESULT.read_text(encoding="utf-8"))
    return actual


def _check_summary(summary, rows, *, grouped=False):
    assert summary["records"] == (sum(row["records"] for row in rows) if grouped else len(rows))
    for key in ("claims", "supported", "contradicted", "unverifiable"):
        assert summary[key] == sum(row[key] for row in rows)
    assert summary["claims"] == sum(summary[key] for key in ("supported", "contradicted", "unverifiable"))
    covered = sum(bool(row["supported"] or row["contradicted"]) for row in rows)
    claims = summary["claims"]
    if not grouped:
        assert summary["case_coverage"] == (covered / len(rows) if rows else 0)
    assert summary["claim_coverage"] == ((summary["supported"] + summary["contradicted"]) / claims if claims else 0)
    assert summary["claim_support_rate"] == (summary["supported"] / claims if claims else 0)
    assert summary["claim_contradiction_rate"] == (summary["contradicted"] / claims if claims else 0)
    judged = summary["supported"] + summary["contradicted"]
    assert summary["citation_claim_precision"] == (summary["supported"] / judged if judged else 0)


def test_full_flow_and_input_provenance(calibration):
    assert set(calibration) == {"full_50_r3", "all_rounds", "known_correct", "historical", "fixture", "ablation"}
    dataset = json.loads((ROOT / "benchmark/datasets/multihop_wiki.json").read_text(encoding="utf-8"))
    assert dataset["version"] == "1.4"
    case_ids = {case["id"] for case in dataset["cases"]}
    assert len(case_ids) == 50
    for rnd in (1, 2, 3):
        source = ROOT / f"benchmark/results/stage12_multiround_20261101_120442_r{rnd}.json"
        data = json.loads(source.read_text(encoding="utf-8"))
        assert data["dataset_version"] == dataset["version"]
        assert {case["id"] for group in ("traditional_rag", "wiki_agent") for case in data["results"][group]["per_case"]} == case_ids
        assert all(len(data["results"][group]["per_case"]) == 50 for group in ("traditional_rag", "wiki_agent"))
    assert {row["id"] for row in calibration["full_50_r3"]["per_case"]} == case_ids
    assert calibration["full_50_r3"]["records"] == 100
    assert calibration["all_rounds"]["records"] == 300
    assert calibration["full_50_r3"]["unique_cases"] == calibration["all_rounds"]["unique_cases"] == 50


def test_r3_aggregation_and_verdict_partition(calibration):
    r3 = calibration["full_50_r3"]
    rows = r3["per_case"]
    assert len(rows) == 50
    _check_summary(r3["all"], rows, grouped=True)
    for name, selected in (("answerable", [r for r in rows if r["answerable"]]),
                           ("unanswerable", [r for r in rows if not r["answerable"]])):
        _check_summary(r3[name], selected, grouped=True)
    for key in ("records", "claims", "supported", "contradicted", "unverifiable"):
        assert r3["all"][key] == r3["answerable"][key] + r3["unanswerable"][key]
        assert calibration["all_rounds"]["all"][key] == sum(calibration["all_rounds"][part][key] for part in ("answerable", "unanswerable"))
    assert calibration["all_rounds"]["all"]["records"] == 300
    for part in ("all", "answerable", "unanswerable"):
        s = calibration["all_rounds"][part]
        assert s["claims"] == s["supported"] + s["contradicted"] + s["unverifiable"]
        assert s["claim_coverage"] == (s["supported"] + s["contradicted"]) / s["claims"]

    # Per-case rows collapse two systems, so their covered count cannot be
    # inferred from a nonzero case total; check it against each raw record.
    sys.path.insert(0, str(ROOT / "benchmark"))
    sys.path.insert(0, str(ROOT / "src"))
    import evidence_audit as ea
    import run_multihop_evaluation as m4

    pages = {page.id: page.content for page in m4.build_wiki_pages(m4.load_documents())}
    source = json.loads((ROOT / "benchmark/results/stage12_multiround_20261101_120442_r3.json").read_text(encoding="utf-8"))
    raw = [case for group in ("traditional_rag", "wiki_agent") for case in source["results"][group]["per_case"]]
    for part, chosen in (("all", raw), ("answerable", [c for c in raw if c["answerable"]]),
                         ("unanswerable", [c for c in raw if not c["answerable"]])):
        covered = sum(any(claim.verdict in (ea.SUPPORTED, ea.CONTRADICTED)
                          for claim in ea.audit_answer(c["answer"], c["sources"], c["read_page_ids"], pages).claims)
                      for c in chosen)
        assert r3[part]["case_coverage"] == covered / len(chosen)


def test_human_review_and_historical_errors(calibration):
    review = ROOT / "benchmark/reviews"
    batch1 = json.loads((review / "stage13_1_batch1_verdicts_20261101.json").read_text(encoding="utf-8"))
    batch2 = json.loads((review / "stage13_3_batch2_verdicts_20261101.json").read_text(encoding="utf-8"))
    correct_ids = {v["review_id"] for v in batch1["verdicts"] + batch2["verdicts"] if v["human_verdict"] == "correct"}
    known = calibration["known_correct"]
    assert len(correct_ids) == known["records"] == known["unique_review_ids"] == 35
    assert {r["id"] for r in known["per_case"]} == correct_ids
    assert known["unique_cases"] == len({r["case_id"] for r in known["per_case"]})
    _check_summary(known["summary"], known["per_case"])
    assert all(r["contradicted"] == 0 for r in known["per_case"])
    historical = calibration["historical"]
    assert set(historical) == {"mh38_responsibility_synthesis", "mh10_direct_reports", "mh49_orion_upward_chain_ceo"}
    for row in historical.values():
        assert row["verdict"] == "contradicted" and row["contradicted"] > 0
        assert row["claims"] == len(row["claims_detail"])
        for label, count in (("supported", row["supported"]), ("contradicted", row["contradicted"]), ("unverifiable", row["unverifiable"])):
            assert sum(c["verdict"] == label for c in row["claims_detail"]) == count


def test_fixture_and_ablation_contract(calibration):
    expected = json.loads((ROOT / "benchmark/datasets/evidence_audit_v0_cases.json").read_text(encoding="utf-8"))["cases"]
    fixture = calibration["fixture"]
    assert len(expected) == fixture["cases"] == fixture["matches"] == 10
    assert {r["id"] for r in fixture["per_case"]} == {r["id"] for r in expected}
    assert all(r["match"] and r["actual_verdict"] == r["expected_verdict"]
               and r["actual_claim_verdicts"] == r["expected_claim_verdicts"] for r in fixture["per_case"])
    assert set(calibration["ablation"]) == {"core_v0.1", "paraphrase", "clause_split", "dimensions", "paraphrase+clause", "all_v0.2"}
    for variant in calibration["ablation"].values():
        assert set(variant) == {"known_correct", "r3_answerable"}
        assert variant["known_correct"]["records"] == 35
        assert variant["r3_answerable"]["records"] == calibration["full_50_r3"]["answerable"]["records"]
        for summary in variant.values():
            assert summary["claims"] == sum(summary[k] for k in ("supported", "contradicted", "unverifiable"))
    assert calibration["ablation"]["all_v0.2"]["known_correct"] == calibration["known_correct"]["summary"]
    assert calibration["ablation"]["all_v0.2"]["r3_answerable"] == calibration["full_50_r3"]["answerable"]
