"""Stage 13.0: build a human-review dataset from Stage 12 results.

Read-only with respect to benchmark data, scorer, agent, and historical
results.  This script only writes new files under benchmark/reviews/.
It makes no model/API calls.
"""

from __future__ import annotations

import csv
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "benchmark"))
sys.path.insert(0, str(ROOT / "src"))

import run_multihop_evaluation as m4  # noqa: E402
import run_stage10_comparison as m10  # noqa: E402

BATCH_ID = "stage12_multiround_20261101_120442"
RESULTS_DIR = ROOT / "benchmark" / "results"
REVIEW_DIR = ROOT / "benchmark" / "reviews"
OUT_JSON = REVIEW_DIR / "stage13_0_human_review_20261101.json"
OUT_CSV = REVIEW_DIR / "stage13_0_human_review_20261101.csv"

ROUND_FILES = {
    index: RESULTS_DIR / f"{BATCH_ID}_r{index}.json" for index in (1, 2, 3)
}
CHECKPOINT_FILES = {
    index: RESULTS_DIR / f"{BATCH_ID}_r{index}.checkpoint.jsonl"
    for index in (1, 2, 3)
}
SUMMARY_FILE = RESULTS_DIR / f"{BATCH_ID}_summary.json"
LOG_FILE = RESULTS_DIR / f"{BATCH_ID}.log"
DATASET_FILE = ROOT / "benchmark" / "datasets" / "multihop_wiki.json"
SCORER_FILE = ROOT / "benchmark" / "run_multihop_evaluation.py"
SCORER_TEST_FILE = ROOT / "tests" / "wiki" / "test_multihop_evaluation.py"
STAGE10_FILE = RESULTS_DIR / "stage10_comparison_20261031_211900.json"
STAGE10_CHECKPOINT = RESULTS_DIR / "stage10_comparison_20261031_211900.checkpoint.jsonl"
STAGE11_FILE = RESULTS_DIR / "stage11_ablation_20261031_220542.json"
STAGE11_CHECKPOINT = RESULTS_DIR / "stage11_ablation_20261031_220542.checkpoint.jsonl"

GROUPS = ("traditional_rag", "wiki_agent")

# (round, group, case_id, suspected_issue, suspected_detail)
RISK_RECORDS = [
    (1, "traditional_rag", "mh08_ceo_unanswerable",
     "negation_or_refusal_recognition",
     "refusal_keyword_match=False: 'I do not know' and 'none ... state' are not in refusal_keywords"),
    (1, "traditional_rag", "mh18_orion_contributor_manager_office",
     "continuous_phrase_or_numeric_format",
     "keyword group 2 False: 'which has three floors in total' does not contain 'building a has three floors'"),
    (1, "traditional_rag", "mh24_engineering_budget_vs_projects",
     "cross_sentence_relation_or_entity_binding",
     "comparison_check=False: numbers are in one sentence and 'therefore greater' is in another; no single-sentence left/right/marker match"),
    (1, "traditional_rag", "mh30_david_university",
     "negation_or_refusal_recognition",
     "refusal_keyword_match=False: 'do not state' is not in refusal_keywords"),
    (1, "traditional_rag", "mh40_walkway_data_team_synthesis",
     "continuous_phrase_or_numeric_format",
     "keyword group 1 False: 'second and third floors' does not contain 'second floor'"),
    (1, "traditional_rag", "mh47_exec_level_to_phoenix_chain",
     "relation_direction_or_order",
     "relation_chain_check=False while all keyword groups match: direct/indirect pairs require literal direct(ly) near named pairs"),
    (1, "traditional_rag", "mh50_orion_exact_completion_date",
     "negation_or_refusal_recognition",
     "refusal_keyword_match=False: 'does not give' is not in refusal_keywords"),
    (1, "traditional_rag", "mh51_total_employee_count",
     "negation_or_refusal_recognition",
     "refusal_keyword_match=False: 'I do not know' and 'none ... state' are not in refusal_keywords"),
    (2, "traditional_rag", "mh02_project_budget",
     "continuous_phrase_or_numeric_format",
     "keyword group 1 False: '$500,000' does not contain 'thousand' or 'hundred'"),
    (2, "traditional_rag", "mh20_phoenix_deadline_office",
     "continuous_phrase_or_numeric_format",
     "keyword group 1 False: 'and is led by Alice Smith' is not one of the accepted leader phrases"),
    (2, "traditional_rag", "mh24_engineering_budget_vs_projects",
     "cross_sentence_relation_or_entity_binding",
     "comparison_check=False: left/right entities and the comparison marker are split across sentences"),
    (2, "traditional_rag", "mh29_alice_salary",
     "negation_or_refusal_recognition",
     "refusal_keyword_match=False: 'None ... mention' and 'no compensation information is provided' are not in refusal_keywords"),
    (2, "traditional_rag", "mh30_david_university",
     "negation_or_refusal_recognition",
     "refusal_keyword_match=False: 'do not state' is not in refusal_keywords"),
    (2, "traditional_rag", "mh40_walkway_data_team_synthesis",
     "continuous_phrase_or_numeric_format",
     "keyword group 1 False: 'second and third floors' does not contain 'second floor'"),
    (2, "traditional_rag", "mh47_exec_level_to_phoenix_chain",
     "relation_direction_or_order",
     "relation_chain_check=False while all keyword groups match: direct/indirect pairs require literal direct(ly) near named pairs"),
    (2, "traditional_rag", "mh50_orion_exact_completion_date",
     "negation_or_refusal_recognition",
     "refusal_keyword_match=False: 'could not confirm' is not in refusal_keywords"),
    (2, "traditional_rag", "mh51_total_employee_count",
     "negation_or_refusal_recognition",
     "refusal_keyword_match=False: 'I do not know'/'none states' are not in refusal_keywords"),
    (3, "traditional_rag", "mh25_project_launch_order",
     "continuous_phrase_or_numeric_format",
     "keyword group 0 False: 'launched or initiated first' is not one of the accepted phrases"),
    (3, "traditional_rag", "mh40_walkway_data_team_synthesis",
     "continuous_phrase_or_numeric_format",
     "keyword group 1 False: 'second and third floors' does not contain 'second floor'"),
    (3, "traditional_rag", "mh47_exec_level_to_phoenix_chain",
     "relation_direction_or_order",
     "relation_chain_check=False while all keyword groups match: direct/indirect pairs require literal direct(ly) near named pairs"),
    (3, "traditional_rag", "mh50_orion_exact_completion_date",
     "negation_or_refusal_recognition",
     "refusal_keyword_match=False: 'could not confirm' is not in refusal_keywords"),
    (3, "traditional_rag", "mh51_total_employee_count",
     "negation_or_refusal_recognition",
     "refusal_keyword_match=False: 'I do not know'/'none ... state' are not in refusal_keywords"),
    (3, "traditional_rag", "mh54_alice_carol_role_office_disambiguation",
     "continuous_phrase_or_numeric_format",
     "BORDERLINE: keyword group 4 False because the answer omits 'core team member'; leader and floors are otherwise correct"),
    (1, "wiki_agent", "mh46_phoenix_reporting_chain",
     "relation_direction_or_order",
     "relation_chain leadership_pairs=False: 'lead of Project Phoenix' is not in markers leads/led/leader"),
    (1, "wiki_agent", "mh47_exec_level_to_phoenix_chain",
     "relation_direction_or_order",
     "relation_chain direct_pairs=False: answer uses 'manages' and a summary 'both are direct'; no literal direct(ly) near each named pair"),
    (2, "wiki_agent", "mh46_phoenix_reporting_chain",
     "relation_direction_or_order",
     "relation_chain leadership_pairs=False: 'lead of Project Phoenix' is not in markers leads/led/leader"),
    (2, "wiki_agent", "mh47_exec_level_to_phoenix_chain",
     "relation_direction_or_order",
     "relation_chain direct_pairs=False: answer uses 'manages' and a summary 'both are direct'; no literal direct(ly) near each named pair"),
    (3, "wiki_agent", "mh08_ceo_unanswerable",
     "negation_or_refusal_recognition",
     "refusal=True but fabrication_keyword_match=True: 'ceo's name is' is a substring of 'The CEO's name is not stated'"),
    (3, "wiki_agent", "mh24_engineering_budget_vs_projects",
     "cross_sentence_relation_or_entity_binding",
     "comparison_check=False: 'well above that combined amount' lacks an accepted right alias and spans a sentence boundary"),
    (1, "wiki_agent", "mh38_responsibility_synthesis",
     "no_source_keyword_false_positive",
     "auto_correct=True, verified_correct=False, status=insufficient_evidence, sources=[]; answer swaps Bob/Eve responsibilities"),
    (2, "wiki_agent", "mh10_direct_reports",
     "no_source_keyword_false_positive",
     "auto_correct=True, verified_correct=False, status=insufficient_evidence, sources=[]; answer fabricates David/Frank/Grace as direct reports"),
    (2, "wiki_agent", "mh49_orion_upward_chain_ceo",
     "no_source_keyword_false_positive",
     "auto_correct=True, verified_correct=False, status=insufficient_evidence, sources=[]; answer inserts a CTO level between Eve and CEO"),
]

RISK_TYPES = [
    ("continuous_phrase_or_numeric_format", "连续短语与数字格式"),
    ("cross_sentence_relation_or_entity_binding", "跨句关系与实体绑定"),
    ("alias_or_pronoun_resolution", "别名与代词解析"),
    ("negation_or_refusal_recognition", "否定表达和拒答识别"),
    ("relation_direction_or_order", "关系方向与顺序"),
    ("source_present_but_unsupported", "来源存在但内容不支持"),
    ("no_source_keyword_false_positive", "无来源答案被关键词误判为正确"),
]


def _rel(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def _file_info(path: Path) -> dict:
    data = path.read_bytes()
    return {
        "path": _rel(path),
        "bytes": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
    }


def _load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _result_index(round_payload: dict) -> dict:
    return {
        (group, case["id"]): case
        for group in GROUPS
        for case in round_payload["results"][group]["per_case"]
    }


def _relation_chain_pairs(answer: str, chain: dict) -> list[dict]:
    sentences = [
        m4._normalize(sentence) for sentence in m4._split_sentences(answer)
    ]
    details = []

    def add(kind: str, pairs: list[dict], markers_default: list[str], **kwargs):
        for pair in pairs:
            markers = pair.get("markers", markers_default)
            passed = m4._has_relation_pair(
                sentences,
                pair["from"],
                pair["to"],
                markers,
                direction="either",
                entity_groups=chain.get("entities", []),
                **kwargs,
            )
            details.append({
                "kind": kind,
                "from": pair["from"][0],
                "to": pair["to"][0],
                "passed": passed,
            })

    add("direct", chain.get("direct_pairs", []), ["direct", "directly"],
        adjacent=True, inverse_requires_phrase=True)
    add("indirect", chain.get("indirect_pairs", []),
        ["indirect", "indirectly", "through", "two levels"])
    add("forbidden_direct", chain.get("forbidden_direct_pairs", []),
        ["direct", "directly"], adjacent=True,
        marker_inside_pair=True, relative_clause_guard=True)
    add("leadership", chain.get("leadership_pairs", []),
        ["leads", "led", "leader"])
    return details


def _scoring_details(answer: str | None, case: dict) -> dict:
    normalized = m4._normalize(answer or "")
    details = {
        "keyword_groups": [
            {
                "index": index,
                "matched": m4._matches_any(normalized, group),
            }
            for index, group in enumerate(case.get("answer_keyword_groups", []))
        ],
        "comparison_check": (
            m4._check_comparison(answer or "", case)
            if case.get("comparison_checks") else None
        ),
        "relation_chain_check": (
            m4._check_relation_chain(answer or "", case)
            if case.get("relation_chain") else None
        ),
        "relation_chain_pairs": (
            _relation_chain_pairs(answer or "", case["relation_chain"])
            if case.get("relation_chain") else []
        ),
        "answer_bindings_check": (
            m4._check_answer_bindings(answer or "", case)
            if case.get("answer_bindings") else None
        ),
        "shared_entities_check": (
            m4._check_shared_entities(answer or "", case)
            if case.get("shared_entities") else None
        ),
        "ratio_check": (
            m4._check_ratio(answer or "", case)
            if case.get("ratio_checks") else None
        ),
    }
    if not case["answerable"]:
        details.update({
            "refusal_keyword_match": m4._matches_any(
                normalized, case.get("refusal_keywords", [])
            ),
            "fabrication_keyword_match": m4._matches_any(
                normalized, case.get("fabrication_keywords", [])
            ),
            "fabrication_pattern_match": m4._matches_any_pattern(
                answer or "", case.get("fabrication_patterns", [])
            ),
            "fabrication_pattern_cs_match": m4._matches_any_pattern(
                answer or "", case.get("fabrication_patterns_cs", []),
                ignore_case=False,
            ),
            "fabrication_positive_pattern_match": m4._matches_positive_pattern(
                answer or "", case.get("fabrication_patterns_positive", [])
            ),
        })
    else:
        details.update({
            "refusal_keyword_match": None,
            "fabrication_keyword_match": None,
            "fabrication_pattern_match": None,
            "fabrication_pattern_cs_match": None,
            "fabrication_positive_pattern_match": None,
        })
    details["auto_correct_recomputed"] = m4.score_answer(answer, case)
    return details


def _usage(result: dict) -> dict:
    calls = result.get("model_calls") or []
    return {
        "model_requests": result.get("model_requests", 0),
        "prompt_tokens": result.get("prompt_tokens", 0),
        "completion_tokens": result.get("completion_tokens", 0),
        "total_tokens": result.get("total_tokens", 0),
        "cache_hit_tokens": result.get("cache_hit_tokens", 0),
        "cache_miss_tokens": result.get("cache_miss_tokens", 0),
        "latency_seconds": result.get("latency_seconds", 0.0),
        "wall_seconds": result.get("wall_seconds", 0.0),
        "model_call_count": len(calls),
        "usage_missing_calls": sum(
            1 for call in calls
            if (call.get("usage") or {}).get("total_tokens", 0) == 0
        ),
    }


def _case_round_outcomes(case_id: str, round_indices: dict) -> list[dict]:
    outcomes = []
    for round_index, round_payload in round_indices.items():
        for group in GROUPS:
            result = round_payload["results"][group]["per_case"]
            result = next(c for c in result if c["id"] == case_id)
            outcomes.append({
                "round": round_index,
                "group": group,
                "auto_correct": result["answer_correct"],
                "verified_correct": result["grounded_correct"],
                "status": result["status"],
                "source_count": len(result["sources"]),
                "answer_present": result["answer"] is not None,
            })
    return outcomes


def _make_record(
    round_index: int,
    group: str,
    case_id: str,
    *,
    record_type: str,
    suspected_issue: str,
    suspected_detail: str,
    sample_stratum: str | None = None,
    sample_reason: str = "",
    sample_tags: list[str] | None = None,
    case_rounds: dict,
    cases: dict,
) -> dict:
    case = cases[case_id]
    result = next(
        c for c in case_rounds[round_index]["results"][group]["per_case"]
        if c["id"] == case_id
    )
    answer = result["answer"]
    assert answer is not None, f"missing answer for {round_index}/{group}/{case_id}"
    details = _scoring_details(answer, case)
    assert details["auto_correct_recomputed"] == result["answer_correct"], (
        f"score mismatch for {round_index}/{group}/{case_id}"
    )
    review_id = (
        f"s13_{record_type}_r{round_index}_{group}_"
        f"{case_id}"
    )
    return {
        "review_id": review_id,
        "record_type": record_type,
        "round": round_index,
        "group": group,
        "case_id": case_id,
        "question": case["question"],
        "reference_answer": case["reference_answer"],
        "model_answer": answer,
        "source_ids": list(result["sources"]),
        "read_page_ids": list(result.get("read_page_ids", [])),
        "accessed_page_ids": list(result.get("accessed_pages", [])),
        "required_page_ids": list(case["required_page_ids"]),
        "alternative_evidence_sets": list(
            case.get("alternative_evidence_sets", [])
        ),
        "answerable": case["answerable"],
        "question_type": case.get("question_type"),
        "hops": case.get("hops"),
        "status": result["status"],
        "failure_type": result.get("failure_type"),
        "error": result.get("error"),
        "auto_correct": result["answer_correct"],
        "verified_correct": result["grounded_correct"],
        "evidence_recall": result.get("evidence_recall"),
        "read_evidence_recall": result.get("read_evidence_recall"),
        "source_precision": result.get("source_precision"),
        "scoring_rule": result.get("scoring_rule"),
        "scoring_details": details,
        "usage": _usage(result),
        "same_case_rounds": _case_round_outcomes(case_id, case_rounds),
        "suspected_issue": suspected_issue,
        "suspected_detail": suspected_detail,
        "sample_stratum": sample_stratum,
        "sample_reason": sample_reason,
        "sample_tags": sample_tags or [],
        "human_verdict": "",
        "human_evidence_verdict": "",
        "review_notes": "",
        "original_result_path": _rel(ROUND_FILES[round_index]),
        "original_checkpoint_path": _rel(CHECKPOINT_FILES[round_index]),
    }


def _has_only_rag(case_id: str, outcomes: dict) -> bool:
    return any(
        outcomes[case_id]["traditional_rag"][i]
        and not outcomes[case_id]["wiki_agent"][i]
        for i in range(3)
    )


def _has_only_agent(case_id: str, outcomes: dict) -> bool:
    return any(
        outcomes[case_id]["wiki_agent"][i]
        and not outcomes[case_id]["traditional_rag"][i]
        for i in range(3)
    )


def _has_both(case_id: str, outcomes: dict) -> bool:
    return any(
        outcomes[case_id]["traditional_rag"][i]
        and outcomes[case_id]["wiki_agent"][i]
        for i in range(3)
    )


def _has_unanswerable(case_id: str, cases: dict, outcomes: dict) -> bool:
    return (
        not cases[case_id]["answerable"]
        and (
            any(outcomes[case_id]["traditional_rag"])
            or any(outcomes[case_id]["wiki_agent"])
        )
    )


def _has_multi_hop(case_id: str, cases: dict, outcomes: dict) -> bool:
    case = cases[case_id]
    is_multi = (
        case.get("relation_chain")
        or case.get("hops", 0) >= 2
        or "relation" in case.get("question_type", "")
        or "chain" in case.get("question_type", "")
    )
    return bool(
        is_multi
        and (
            any(outcomes[case_id]["traditional_rag"])
            or any(outcomes[case_id]["wiki_agent"])
        )
    )


def _has_binding(case_id: str, cases: dict, outcomes: dict) -> bool:
    case = cases[case_id]
    return bool(
        (
            case.get("answer_bindings")
            or case.get("shared_entities")
            or case.get("relation_chain")
            or case.get("ratio_checks")
            or case.get("comparison_checks")
        )
        and (
            any(outcomes[case_id]["traditional_rag"])
            or any(outcomes[case_id]["wiki_agent"])
        )
    )


def _take_evenly(case_ids: list[str], quota: int) -> list[str]:
    case_ids = sorted(case_ids)
    if quota <= 0:
        return []
    if quota >= len(case_ids):
        return case_ids
    if quota == 1:
        return [case_ids[0]]
    return [
        case_ids[round(index * (len(case_ids) - 1) / (quota - 1))]
        for index in range(quota)
    ]


def _select_sample(cases: dict, outcomes: dict) -> dict[str, str]:
    strata = [
        ("only_rag_correct", lambda cid: _has_only_rag(cid, outcomes), 5),
        ("unanswerable", lambda cid: _has_unanswerable(cid, cases, outcomes), 4),
        ("only_agent_correct", lambda cid: _has_only_agent(cid, outcomes), 6),
        ("entity_attribute_binding", lambda cid: _has_binding(cid, cases, outcomes), 4),
        ("multi_hop_relation", lambda cid: _has_multi_hop(cid, cases, outcomes), 3),
        ("both_correct", lambda cid: _has_both(cid, outcomes), 2),
    ]
    assigned = {}
    for name, predicate, quota in strata:
        candidates = [
            cid for cid in cases if cid not in assigned and predicate(cid)
        ]
        selected = _take_evenly(candidates, quota)
        for cid in selected:
            assigned[cid] = name
    return assigned


def _representative(
    case_id: str,
    stratum: str,
    outcomes: dict,
    *,
    preferred_group: str | None = None,
) -> tuple[int, str]:
    rag = outcomes[case_id]["traditional_rag"]
    agent = outcomes[case_id]["wiki_agent"]
    if stratum == "only_rag_correct":
        for index in range(3):
            if rag[index] and not agent[index]:
                return index + 1, "traditional_rag"
    if stratum == "only_agent_correct":
        for index in range(3):
            if agent[index] and not rag[index]:
                return index + 1, "wiki_agent"
    if stratum in ("unanswerable", "entity_attribute_binding", "multi_hop_relation"):
        for index in range(3):
            if agent[index]:
                return index + 1, "wiki_agent"
        for index in range(3):
            if rag[index]:
                return index + 1, "traditional_rag"
    if stratum == "both_correct":
        groups = (
            [preferred_group] if preferred_group
            else ["traditional_rag", "wiki_agent"]
        )
        for group in groups:
            values = rag if group == "traditional_rag" else agent
            other = agent if group == "traditional_rag" else rag
            for index in range(3):
                if values[index] and other[index]:
                    return index + 1, group
    for index in range(3):
        if rag[index] or agent[index]:
            group = "traditional_rag" if rag[index] else "wiki_agent"
            return index + 1, group
    raise RuntimeError(f"no auto-correct representative for {case_id}")


def _sample_tags(case_id: str, cases: dict, outcomes: dict) -> list[str]:
    tags = []
    if _has_only_rag(case_id, outcomes):
        tags.append("only_rag_correct")
    if _has_only_agent(case_id, outcomes):
        tags.append("only_agent_correct")
    if _has_both(case_id, outcomes):
        tags.append("both_correct")
    if _has_unanswerable(case_id, cases, outcomes):
        tags.append("unanswerable")
    if _has_multi_hop(case_id, cases, outcomes):
        tags.append("multi_hop_relation")
    if _has_binding(case_id, cases, outcomes):
        tags.append("entity_attribute_binding")
    return tags


def _checkpoint_lookup(path: Path) -> dict:
    records = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        records[record["case_id"]] = record
    return records


def _validate_checkpoint(
    round_index: int,
    group: str,
    case_id: str,
    result: dict,
    checkpoint: dict,
) -> None:
    key = "rag" if group == "traditional_rag" else "agent"
    record = checkpoint[case_id][key]
    for field in ("answer", "answer_correct", "status", "sources", "read_page_ids"):
        assert record.get(field) == result.get(field), (
            f"checkpoint mismatch {round_index}/{group}/{case_id}/{field}"
        )


def _historical_evidence() -> list[dict]:
    stage10 = _load_json(STAGE10_FILE)
    stage11 = _load_json(STAGE11_FILE)

    def pick(payload: dict, group: str, case_id: str) -> dict:
        return next(
            case for case in payload["results"][group]["per_case"]
            if case["id"] == case_id
        )

    pronoun = pick(stage10, "wiki_agent", "mh47_exec_level_to_phoenix_chain")
    alias = pick(stage11, "no_links", "mh43_alice_david_join_degree_building")
    return [
        {
            "source": _rel(STAGE10_FILE),
            "group": "wiki_agent",
            "case_id": "mh47_exec_level_to_phoenix_chain",
            "auto_correct": pronoun["answer_correct"],
            "model_answer": pronoun["answer"],
            "reason": "pronoun/coreference: 'She reports directly to Bob Jones' split from the Alice heading",
        },
        {
            "source": _rel(STAGE11_FILE),
            "group": "no_links",
            "case_id": "mh43_alice_david_join_degree_building",
            "auto_correct": alias["answer_correct"],
            "model_answer": alias["answer"],
            "reason": "alias: 'joined Acme Corp first' does not match 'joined acme first'",
        },
    ]


def _failed_checks(record: dict) -> list[str]:
    details = record["scoring_details"]
    failed = []
    for group in details["keyword_groups"]:
        if not group["matched"]:
            failed.append(f"keyword_group[{group['index']}]")
    for key in (
        "comparison_check", "relation_chain_check", "answer_bindings_check",
        "shared_entities_check", "ratio_check",
    ):
        if details.get(key) is False:
            failed.append(key)
    for key in (
        "refusal_keyword_match", "fabrication_keyword_match",
        "fabrication_pattern_match", "fabrication_pattern_cs_match",
        "fabrication_positive_pattern_match",
    ):
        if details.get(key) is True and key != "refusal_keyword_match":
            failed.append(key)
    if details.get("refusal_keyword_match") is False:
        failed.append("refusal_keyword_match")
    if record["auto_correct"] and not record["verified_correct"]:
        failed.append("grounding/verified_correct")
    return failed


def _build_risk_catalog(records: list[dict], samples: list[dict]) -> list[dict]:
    catalog = []
    for key, label in RISK_TYPES:
        matching = [r for r in records if r["suspected_issue"] == key]
        entry = {
            "risk_type": key,
            "label": label,
            "stage12_review_ids": [r["review_id"] for r in matching],
            "stage12_confirmed_count": len(matching),
            "evidence": [
                {
                    "review_id": r["review_id"],
                    "case_id": r["case_id"],
                    "round": r["round"],
                    "group": r["group"],
                    "auto_correct": r["auto_correct"],
                    "verified_correct": r["verified_correct"],
                    "status": r["status"],
                    "failed_checks": _failed_checks(r),
                    "answer_excerpt": (r["model_answer"] or "")[:500],
                }
                for r in matching[:4]
            ],
            "note": "",
        }
        if key == "continuous_phrase_or_numeric_format":
            entry["note"] = (
                "Some misses are deliberate lexical policy: "
                "tests/wiki/test_multihop_evaluation.py explicitly asserts "
                "that bare '$500,000' fails a group that requires "
                "'thousand'/'hundred'. Human review should separate intended "
                "trade-offs from defects."
            )
        if key == "negation_or_refusal_recognition":
            entry["note"] = (
                "The refusal tests cover only a small keyword set; observed "
                "gaps ('I do not know', 'does not give', 'could not confirm') "
                "are outside that coverage."
            )
        if key == "alias_or_pronoun_resolution":
            entry["note"] = (
                "No Stage 12 confirmed instance; historical Stage 10/11 "
                "evidence is attached. Add Stage 12 evidence if human review finds one."
            )
            entry["historical_evidence"] = _historical_evidence()
        if key == "source_present_but_unsupported":
            candidates = [
                s for s in samples
                if s["source_ids"]
                and (
                    "entity_attribute_binding" in s["sample_tags"]
                    or "multi_hop_relation" in s["sample_tags"]
                )
            ]
            entry["note"] = (
                "No confirmed Stage 12 instance. These auto-correct samples "
                "have sources and binding/multi-hop structure; verify whether "
                "the cited pages actually support every claim."
            )
            entry["candidate_sample_review_ids"] = [
                s["review_id"] for s in candidates[:6]
            ]
            entry["candidate_evidence"] = [
                {
                    "review_id": s["review_id"],
                    "case_id": s["case_id"],
                    "source_ids": s["source_ids"],
                    "answer_excerpt": s["model_answer"][:500],
                }
                for s in candidates[:6]
            ]
        if not matching and key not in (
            "alias_or_pronoun_resolution", "source_present_but_unsupported"
        ):
            entry["note"] = "No confirmed Stage 12 instance."
        catalog.append(entry)
    return catalog


def _write_csv(path: Path, records: list[dict]) -> None:
    fields = [
        "review_id", "record_type", "round", "group", "case_id",
        "question", "reference_answer", "model_answer", "source_ids",
        "read_page_ids", "auto_correct", "verified_correct",
        "suspected_issue", "suspected_detail", "status",
        "evidence_recall", "source_precision", "sample_stratum",
        "sample_reason", "sample_tags", "human_verdict",
        "human_evidence_verdict", "review_notes", "original_result_path",
    ]
    with open(path, "w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for record in records:
            row = {}
            for field in fields:
                value = record.get(field)
                if isinstance(value, (list, dict)):
                    value = json.dumps(value, ensure_ascii=False)
                row[field] = value
            writer.writerow(row)


def main() -> None:
    for path in (OUT_JSON, OUT_CSV):
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite existing review file: {path}")
    REVIEW_DIR.mkdir(parents=True, exist_ok=True)

    dataset = _load_json(DATASET_FILE)
    cases = {case["id"]: case for case in dataset["cases"]}
    rounds = {index: _load_json(path) for index, path in ROUND_FILES.items()}
    checkpoints = {
        index: _checkpoint_lookup(path)
        for index, path in CHECKPOINT_FILES.items()
    }
    summary = _load_json(SUMMARY_FILE)

    outcomes = {
        case_id: {"traditional_rag": [], "wiki_agent": []}
        for case_id in cases
    }
    for round_index, payload in rounds.items():
        for group in GROUPS:
            for result in payload["results"][group]["per_case"]:
                outcomes[result["id"]][group].append(
                    bool(result["answer_correct"])
                )

    risk_records = []
    for round_index, group, case_id, issue, detail in RISK_RECORDS:
        record = _make_record(
            round_index,
            group,
            case_id,
            record_type="risk",
            suspected_issue=issue,
            suspected_detail=detail,
            case_rounds=rounds,
            cases=cases,
        )
        result = next(
            c for c in rounds[round_index]["results"][group]["per_case"]
            if c["id"] == case_id
        )
        _validate_checkpoint(
            round_index, group, case_id, result, checkpoints[round_index]
        )
        risk_records.append(record)

    assigned = _select_sample(cases, outcomes)
    sample_records = []
    both_seen = 0
    for case_id, stratum in assigned.items():
        preferred_group = None
        if stratum == "both_correct":
            preferred_group = (
                "traditional_rag" if both_seen == 0 else "wiki_agent"
            )
            both_seen += 1
        round_index, group = _representative(
            case_id, stratum, outcomes, preferred_group=preferred_group
        )
        tags = _sample_tags(case_id, cases, outcomes)
        result = next(
            c for c in rounds[round_index]["results"][group]["per_case"]
            if c["id"] == case_id
        )
        assert result["answer_correct"] is True
        reason = (
            f"primary={stratum}; tags={','.join(tags)}; "
            f"representative=r{round_index}/{group}; "
            f"verified_correct={result['grounded_correct']}"
        )
        record = _make_record(
            round_index,
            group,
            case_id,
            record_type="auto_correct_sample",
            suspected_issue="none_suspected",
            suspected_detail="Stratified auto-correct spot-check; no issue asserted.",
            sample_stratum=stratum,
            sample_reason=reason,
            sample_tags=tags,
            case_rounds=rounds,
            cases=cases,
        )
        _validate_checkpoint(
            round_index, group, case_id, result, checkpoints[round_index]
        )
        sample_records.append(record)

    # Deduplicate cases within the sample and verify quotas.
    assert len({r["case_id"] for r in sample_records}) == len(sample_records)
    assert len(risk_records) == 32, len(risk_records)
    assert len(sample_records) == 24, len(sample_records)
    assert sum(1 for r in risk_records if r["group"] == "traditional_rag") == 23
    assert sum(1 for r in risk_records if r["group"] == "wiki_agent") == 9
    assert sum(1 for r in risk_records if not r["auto_correct"]) == 29
    assert sum(1 for r in risk_records if r["auto_correct"]) == 3
    for record in risk_records + sample_records:
        assert record["human_verdict"] == ""
        assert record["human_evidence_verdict"] == ""
        assert record["review_notes"] == ""
        assert record["model_answer"] is not None

    risk_counts = {}
    for record in risk_records:
        risk_counts[record["suspected_issue"]] = (
            risk_counts.get(record["suspected_issue"], 0) + 1
        )
    sample_strata = {}
    for record in sample_records:
        sample_strata[record["sample_stratum"]] = (
            sample_strata.get(record["sample_stratum"], 0) + 1
        )

    source_files = [
        _file_info(path)
        for path in (
            DATASET_FILE, SCORER_FILE, SCORER_TEST_FILE, SUMMARY_FILE, LOG_FILE,
            *ROUND_FILES.values(), *CHECKPOINT_FILES.values(),
            STAGE10_FILE, STAGE10_CHECKPOINT,
            STAGE11_FILE, STAGE11_CHECKPOINT,
        )
    ]

    output = {
        "metadata": {
            "review_dataset_id": "stage13_0_human_review_20261101",
            "created_at": "2026-11-01",
            "batch_id": BATCH_ID,
            "mode": "read-only audit; no model/API calls",
            "dataset": {
                "path": _rel(DATASET_FILE),
                "version": dataset.get("version"),
                "sha256": _file_info(DATASET_FILE)["sha256"],
            },
            "scorer": {
                "version": m10.SCORER_VERSION,
                "source": _rel(SCORER_FILE),
                "sha256": _file_info(SCORER_FILE)["sha256"],
                "tests": _rel(SCORER_TEST_FILE),
                "tests_sha256": _file_info(SCORER_TEST_FILE)["sha256"],
            },
            "scoring_test_notes": [
                "test_alternative_keyword_in_group intentionally asserts that "
                "'$500,000' is False for a group requiring ['500','five'] + "
                "['thousand']; mh02 R2 hits the same policy, so human review "
                "should decide whether this is intended or too strict.",
                "test_partial_match_is_incorrect requires exact phrases such as "
                "'Director of Engineering' and rejects 'Director'; this matches "
                "the observed strict contiguous-phrase behavior.",
                "Refusal tests cover only a small keyword set; the observed "
                "'I do not know' / 'does not give' / 'could not confirm' gaps "
                "are not covered by existing tests.",
            ],
            "git_sha": summary.get("git_sha"),
            "stage10_config_hash": summary.get("stage10_config_hash"),
            "stage12_config_hash": summary.get("stage12_config_hash"),
            "risk_record_count": len(risk_records),
            "risk_counts_by_round": {
                str(index): sum(1 for r in risk_records if r["round"] == index)
                for index in (1, 2, 3)
            },
            "risk_counts_by_group": {
                group: sum(1 for r in risk_records if r["group"] == group)
                for group in GROUPS
            },
            "risk_counts_by_issue": risk_counts,
            "sample_count": len(sample_records),
            "sample_unique_case_count": len(
                {r["case_id"] for r in sample_records}
            ),
            "sample_counts_by_stratum": sample_strata,
            "sample_method": (
                "Unique case-level stratified sample; primary stratum assigned "
                "in priority order only_rag_correct > unanswerable > "
                "only_agent_correct > entity_attribute_binding > "
                "multi_hop_relation > both_correct. Within each stratum, "
                "cases are sorted by case_id and selected evenly (no random "
                "seed). One representative case-round per case; repeated "
                "rounds are shown in same_case_rounds and are not counted as "
                "independent samples."
            ),
            "human_fields_initialized_empty": True,
            "source_files": source_files,
        },
        "risk_records": risk_records,
        "auto_correct_sample_records": sample_records,
        "risk_catalog": _build_risk_catalog(risk_records, sample_records),
    }

    OUT_JSON.write_text(
        json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    _write_csv(OUT_CSV, risk_records + sample_records)

    print(f"wrote {OUT_JSON}")
    print(f"wrote {OUT_CSV}")
    print(f"risk_records={len(risk_records)}")
    print("risk_by_round=", output["metadata"]["risk_counts_by_round"])
    print("risk_by_group=", output["metadata"]["risk_counts_by_group"])
    print("risk_by_issue=", risk_counts)
    print(f"sample_records={len(sample_records)}")
    print("sample_by_stratum=", sample_strata)
    print("human_fields_empty=all")


if __name__ == "__main__":
    main()
