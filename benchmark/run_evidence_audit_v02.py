"""Run EvidenceAudit v0.2 calibration over the full 50-case Stage 12 results.

This script is offline and read-only. It:
  * audits Stage 12 R3 (100 records = 2 systems x 50 cases) and all 3 rounds;
  * reports answerable/unanswerable records separately;
  * recalibrates the 35 Stage 13 known-correct records;
  * reruns the 3 historical errors and 10 fixture cases;
  * runs feature-set ablations to show which v0.2 rules improve coverage.

Usage:
    python benchmark/run_evidence_audit_v02.py [--output path.json]
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
    "mh38_responsibility_synthesis": (
        "Technical roadmap planning is owned by Eve Martinez, the VP of "
        "Engineering. Quarterly performance reviews are run by Bob Jones. "
        "The annual engineering budget is also owned by Bob Jones.",
        ["bob", "eve"],
    ),
    "mh10_direct_reports": (
        "Bob Jones has five direct reports: Alice Smith, Carol Chen, David Kim, "
        "Frank Osei, and Grace Liu.",
        ["bob", "alice", "carol", "david"],
    ),
    "mh49_orion_upward_chain_ceo": (
        "Project Orion's technical lead is David Kim, who reports to Eve "
        "Martinez. Eve Martinez reports to the Chief Technology Officer (CTO), "
        "who reports to the CEO.",
        ["project_orion", "david", "eve"],
    ),
}

FEATURE_SETS = {
    "core_v0.1": [],
    "paraphrase": ["paraphrase"],
    "clause_split": ["clause_split"],
    "dimensions": ["dimensions"],
    "paraphrase+clause": ["paraphrase", "clause_split"],
    "all_v0.2": None,
}


def _load_correct_records() -> list[dict]:
    review_dir = ROOT / "benchmark" / "reviews"
    s0 = json.loads((review_dir / "stage13_0_human_review_20261101.json").read_text(encoding="utf-8"))
    s1 = json.loads((review_dir / "stage13_1_batch1_verdicts_20261101.json").read_text(encoding="utf-8"))
    s3 = json.loads((review_dir / "stage13_3_batch2_verdicts_20261101.json").read_text(encoding="utf-8"))
    records = {r["review_id"]: r for r in s0["risk_records"] + s0["auto_correct_sample_records"]}
    verdicts = {r["review_id"]: r for r in s1["verdicts"] + s3["verdicts"]}
    return [
        records[rid] for rid, v in verdicts.items()
        if v["human_verdict"] == "correct"
    ]


def _audit(answer, sources, read, pages, features=None):
    return ea.audit_answer(answer, sources, read, pages, features=features)


def _row_summary(result):
    return {
        "verdict": result.evidence_verdict,
        "claims": len(result.claims),
        "supported": sum(1 for c in result.claims if c.verdict == ea.SUPPORTED),
        "contradicted": sum(1 for c in result.claims if c.verdict == ea.CONTRADICTED),
        "unverifiable": sum(1 for c in result.claims if c.verdict == ea.UNVERIFIABLE),
        "coverage": result.audit_coverage,
    }


def _aggregate(rows, key_prefix=""):
    claims = sum(r[key_prefix + "claims"] for r in rows)
    sup = sum(r[key_prefix + "supported"] for r in rows)
    con = sum(r[key_prefix + "contradicted"] for r in rows)
    unv = sum(r[key_prefix + "unverifiable"] for r in rows)
    covered = sum(1 for r in rows if r[key_prefix + "supported"] or r[key_prefix + "contradicted"])
    return {
        "records": len(rows),
        "claims": claims,
        "supported": sup,
        "contradicted": con,
        "unverifiable": unv,
        "claim_support_rate": sup / claims if claims else 0.0,
        "claim_contradiction_rate": con / claims if claims else 0.0,
        "citation_claim_precision": sup / (sup + con) if (sup + con) else 0.0,
        "case_coverage": covered / len(rows) if rows else 0.0,
        "claim_coverage": (sup + con) / claims if claims else 0.0,
    }


def _round_rows(rnd: int, pages):
    data = json.loads(
        (ROOT / "benchmark" / "results"
         / f"stage12_multiround_20261101_120442_r{rnd}.json").read_text(encoding="utf-8")
    )
    rows = []
    for group in ("traditional_rag", "wiki_agent"):
        for case in data["results"][group]["per_case"]:
            result = _audit(
                case["answer"], case["sources"], case["read_page_ids"], pages,
            )
            rows.append({
                "id": case["id"],
                "group": group,
                "answerable": case["answerable"],
                **_row_summary(result),
            })
    return rows


def _per_case(rows):
    by_id = {}
    for row in rows:
        entry = by_id.setdefault(row["id"], {
            "id": row["id"],
            "answerable": row["answerable"],
            "records": 0,
            "claims": 0,
            "supported": 0,
            "contradicted": 0,
            "unverifiable": 0,
        })
        entry["records"] += 1
        for key in ("claims", "supported", "contradicted", "unverifiable"):
            entry[key] += row[key]
    result = []
    for entry in sorted(by_id.values(), key=lambda x: x["id"]):
        entry["claim_coverage"] = (
            (entry["supported"] + entry["contradicted"]) / entry["claims"]
            if entry["claims"] else 0.0
        )
        result.append(entry)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    pages = {p.id: p.content for p in m4.build_wiki_pages(m4.load_documents())}
    payload: dict = {}

    # Full 50-case R3: 100 records, 50 unique cases.
    r3 = _round_rows(3, pages)
    payload["full_50_r3"] = {
        "records": len(r3),
        "unique_cases": len({r["id"] for r in r3}),
        "answerable": _aggregate([r for r in r3 if r["answerable"]]),
        "unanswerable": _aggregate([r for r in r3 if not r["answerable"]]),
        "all": _aggregate(r3),
        "per_case": _per_case(r3),
    }

    # All 3 rounds sensitivity: 300 records, 50 unique cases.
    all_rounds = []
    for rnd in (1, 2, 3):
        all_rounds.extend(_round_rows(rnd, pages))
    payload["all_rounds"] = {
        "records": len(all_rounds),
        "unique_cases": len({r["id"] for r in all_rounds}),
        "answerable": _aggregate([r for r in all_rounds if r["answerable"]]),
        "unanswerable": _aggregate([r for r in all_rounds if not r["answerable"]]),
        "all": _aggregate(all_rounds),
    }

    # 35 Stage 13 known-correct records.
    correct = _load_correct_records()
    correct_rows = []
    for record in correct:
        result = _audit(
            record["model_answer"], record["source_ids"],
            record["read_page_ids"], pages,
        )
        correct_rows.append({
            "id": record["review_id"],
            "case_id": record["case_id"],
            **_row_summary(result),
        })
    payload["known_correct"] = {
        "records": len(correct_rows),
        "unique_review_ids": len({r["id"] for r in correct_rows}),
        "unique_cases": len({r["case_id"] for r in correct_rows}),
        "summary": _aggregate(correct_rows),
        "per_case": correct_rows,
    }

    # Historical 3.
    payload["historical"] = {}
    for case_id, (answer, sources) in HISTORICAL.items():
        result = _audit(answer, sources, sources, pages)
        payload["historical"][case_id] = {
            **_row_summary(result),
            "claims_detail": [c.to_dict() for c in result.claims],
        }

    # Fixture 10.
    fixture = json.loads(
        (ROOT / "benchmark" / "datasets" / "evidence_audit_v0_cases.json")
        .read_text(encoding="utf-8")
    )["cases"]
    fixture_rows = []
    for case in fixture:
        result = _audit(
            case["answer"], case["sources"], case["read_page_ids"], pages,
        )
        match = (
            result.evidence_verdict == case["expected_verdict"]
            and [c.verdict for c in result.claims] == case["expected_claim_verdicts"]
        )
        fixture_rows.append({
            "id": case["id"],
            "expected_verdict": case["expected_verdict"],
            "actual_verdict": result.evidence_verdict,
            "expected_claim_verdicts": case["expected_claim_verdicts"],
            "actual_claim_verdicts": [c.verdict for c in result.claims],
            "match": match,
        })
    payload["fixture"] = {
        "cases": len(fixture_rows),
        "matches": sum(1 for r in fixture_rows if r["match"]),
        "per_case": fixture_rows,
    }

    # Feature ablation on known-correct and full R3 answerable records.
    ablation = {}
    for name, features in FEATURE_SETS.items():
        c_rows = []
        for record in correct:
            result = _audit(
                record["model_answer"], record["source_ids"],
                record["read_page_ids"], pages, features=features,
            )
            c_rows.append(_row_summary(result))
        ablation[name] = {
            "known_correct": _aggregate(c_rows),
        }
    # Recompute R3 answerable coverage for each feature set directly from raw
    # per-case records (the row above is the all-feature baseline).
    data_r3 = json.loads(
        (ROOT / "benchmark" / "results"
         / "stage12_multiround_20261101_120442_r3.json").read_text(encoding="utf-8")
    )
    raw_r3_answerable = [
        case
        for group in ("traditional_rag", "wiki_agent")
        for case in data_r3["results"][group]["per_case"]
        if case["answerable"]
    ]
    for name, features in FEATURE_SETS.items():
        rows = []
        for case in raw_r3_answerable:
            result = _audit(
                case["answer"], case["sources"], case["read_page_ids"],
                pages, features=features,
            )
            rows.append(_row_summary(result))
        ablation[name]["r3_answerable"] = _aggregate(rows)
    payload["ablation"] = ablation

    text = json.dumps(payload, ensure_ascii=False, indent=2)
    if args.output:
        args.output.write_text(text, encoding="utf-8")
        print(f"wrote {args.output}")
    else:
        print(text)


if __name__ == "__main__":
    main()
