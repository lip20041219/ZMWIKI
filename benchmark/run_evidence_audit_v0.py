"""Run offline EvidenceAudit v0 over calibration sets.

This script does not call a model and does not modify any benchmark result or
dataset. It audits:
  1. the three historical wrong answers, with their required pages supplied as
     hypothetical valid citations (to test content support, not source absence);
  2. the 35 Stage 13 human-correct records, using their recorded sources;
  3. the 10 cited-but-unsupported fixture cases.

Usage:
    python benchmark/run_evidence_audit_v0.py [--output path.json]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "benchmark"))
sys.path.insert(0, str(ROOT / "src"))

import evidence_audit as ea  # noqa: E402
import run_multihop_evaluation as m4  # noqa: E402

HISTORICAL = {
    "mh38_responsibility_synthesis": {
        "answer": (
            "Technical roadmap planning is owned by Eve Martinez, the VP of "
            "Engineering. Quarterly performance reviews are run by Bob Jones. "
            "The annual engineering budget is also owned by Bob Jones."
        ),
        "sources": ["bob", "eve"],
        "read_page_ids": ["bob", "eve"],
    },
    "mh10_direct_reports": {
        "answer": (
            "Bob Jones has five direct reports: Alice Smith, Carol Chen, "
            "David Kim, Frank Osei, and Grace Liu."
        ),
        "sources": ["bob", "alice", "carol", "david"],
        "read_page_ids": ["bob", "alice", "carol", "david"],
    },
    "mh49_orion_upward_chain_ceo": {
        "answer": (
            "Project Orion's technical lead is David Kim, who reports to Eve "
            "Martinez. Eve Martinez reports to the Chief Technology Officer "
            "(CTO), who reports to the CEO."
        ),
        "sources": ["project_orion", "david", "eve"],
        "read_page_ids": ["project_orion", "david", "eve"],
    },
}


def _load_correct_records() -> list[dict]:
    review_dir = ROOT / "benchmark" / "reviews"
    s0 = json.loads((review_dir / "stage13_0_human_review_20261101.json").read_text(encoding="utf-8"))
    s1 = json.loads((review_dir / "stage13_1_batch1_verdicts_20261101.json").read_text(encoding="utf-8"))
    s3 = json.loads((review_dir / "stage13_3_batch2_verdicts_20261101.json").read_text(encoding="utf-8"))
    records = {
        r["review_id"]: r
        for r in s0["risk_records"] + s0["auto_correct_sample_records"]
    }
    verdicts = {
        r["review_id"]: r
        for r in s1["verdicts"] + s3["verdicts"]
    }
    return [
        records[rid]
        for rid, verdict in verdicts.items()
        if verdict["human_verdict"] == "correct"
    ]


def _summary(result: ea.AuditResult) -> dict:
    return {
        "evidence_verdict": result.evidence_verdict,
        "claims": len(result.claims),
        "claim_support_rate": result.claim_support_rate,
        "claim_contradiction_rate": result.claim_contradiction_rate,
        "citation_claim_precision": result.citation_claim_precision,
        "audit_coverage": result.audit_coverage,
        "supported": sum(1 for c in result.claims if c.verdict == ea.SUPPORTED),
        "contradicted": sum(1 for c in result.claims if c.verdict == ea.CONTRADICTED),
        "unverifiable": sum(1 for c in result.claims if c.verdict == ea.UNVERIFIABLE),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    pages = {p.id: p.content for p in m4.build_wiki_pages(m4.load_documents())}
    payload: dict = {"historical": {}, "known_correct": {}, "fixture": {}}

    for case_id, item in HISTORICAL.items():
        result = ea.audit_answer(
            item["answer"], item["sources"], item["read_page_ids"], pages
        )
        payload["historical"][case_id] = {
            **_summary(result),
            "claims_detail": [c.to_dict() for c in result.claims],
        }

    correct = _load_correct_records()
    all_claims = 0
    supported_claims = 0
    contradicted_claims = 0
    unverifiable_claims = 0
    covered_cases = 0
    contradicted_cases = []
    verdict_counts: dict[str, int] = {}
    for record in correct:
        result = ea.audit_answer(
            record["model_answer"], record["source_ids"],
            record["read_page_ids"], pages,
        )
        summary = _summary(result)
        payload["known_correct"][record["review_id"]] = summary
        verdict_counts[summary["evidence_verdict"]] = (
            verdict_counts.get(summary["evidence_verdict"], 0) + 1
        )
        all_claims += summary["claims"]
        supported_claims += summary["supported"]
        contradicted_claims += summary["contradicted"]
        unverifiable_claims += summary["unverifiable"]
        if summary["supported"] or summary["contradicted"]:
            covered_cases += 1
        if summary["contradicted"]:
            contradicted_cases.append(record["review_id"])

    payload["known_correct_summary"] = {
        "records": len(correct),
        "unique_review_ids": len({r["review_id"] for r in correct}),
        "claims": all_claims,
        "supported_claims": supported_claims,
        "contradicted_claims": contradicted_claims,
        "unverifiable_claims": unverifiable_claims,
        "claim_support_rate": supported_claims / all_claims if all_claims else 0.0,
        "claim_contradiction_rate": (
            contradicted_claims / all_claims if all_claims else 0.0
        ),
        "citation_claim_precision": (
            supported_claims / (supported_claims + contradicted_claims)
            if (supported_claims + contradicted_claims) else 0.0
        ),
        "audit_coverage": covered_cases / len(correct) if correct else 0.0,
        "contradicted_cases": len(contradicted_cases),
        "contradicted_case_ids": contradicted_cases,
        "claim_false_contradiction_rate": (
            contradicted_claims / all_claims if all_claims else 0.0
        ),
        "verdict_counts": verdict_counts,
    }

    fixture_path = ROOT / "benchmark" / "datasets" / "evidence_audit_v0_cases.json"
    fixture = json.loads(fixture_path.read_text(encoding="utf-8"))["cases"]
    fixture_matches = 0
    for case in fixture:
        result = ea.audit_answer(
            case["answer"], case["sources"], case["read_page_ids"], pages
        )
        ok = (
            result.evidence_verdict == case["expected_verdict"]
            and [c.verdict for c in result.claims] == case["expected_claim_verdicts"]
        )
        fixture_matches += int(ok)
        payload["fixture"][case["id"]] = {
            "expected_verdict": case["expected_verdict"],
            "expected_claim_verdicts": case["expected_claim_verdicts"],
            "actual_verdict": result.evidence_verdict,
            "actual_claim_verdicts": [c.verdict for c in result.claims],
            "match": ok,
        }
    payload["fixture_summary"] = {
        "cases": len(fixture),
        "matches": fixture_matches,
    }

    text = json.dumps(payload, ensure_ascii=False, indent=2)
    if args.output:
        args.output.write_text(text, encoding="utf-8")
        print(f"wrote {args.output}")
    else:
        print(text)


if __name__ == "__main__":
    main()
