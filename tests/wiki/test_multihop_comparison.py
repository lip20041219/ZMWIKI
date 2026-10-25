"""Phase 10: manual-link vs auto-link comparison tests (no model calls)."""

import sys
from pathlib import Path

import pytest

_BENCHMARK_DIR = Path(__file__).resolve().parents[2] / "benchmark"
sys.path.insert(0, str(_BENCHMARK_DIR))

import compare_multihop_runs as cmp  # noqa: E402
import run_multihop_evaluation as m4  # noqa: E402

CASES = [
    {
        "id": "c1", "question": "q1", "answerable": True, "hops": 2,
        "required_page_ids": ["a"], "answer_keyword_groups": [["x"]],
    },
    {
        "id": "c2", "question": "q2", "answerable": False, "hops": 0,
        "required_page_ids": [], "answer_keyword_groups": [],
        "refusal_keywords": ["cannot"],
    },
]


def _result(case_id, answer, *, status="answered", steps=3, sources=("a",),
            recall=1.0):
    case = next(c for c in CASES if c["id"] == case_id)
    correct = m4.score_answer(answer, case)
    grounded = correct and status == "answered" and bool(sources)
    return m4.CaseResult(
        id=case_id,
        question=case["question"],
        answer=answer,
        answer_correct=correct,
        status=status,
        sources=list(sources),
        accessed_pages=list(sources),
        required_pages=case["required_page_ids"],
        evidence_recall=recall,
        source_precision=1.0 if sources else 0.0,
        steps=steps,
        pages_read=len(sources),
        answerable=case["answerable"],
        grounded_correct=grounded,
        unsupported_correct=correct and not grounded,
        step_details=[{"step": 0, "action": "answer"}],
    )


def _payload(link_source, results, *, model="deepseek-chat",
             base_url="https://api.deepseek.com/v1", max_steps=8,
             dataset_hash="hash1"):
    return {
        "experiment_id": "multihop_wiki_comparison",
        "dataset": "benchmark/datasets/multihop_wiki.json",
        "dataset_version": "1.0",
        "dataset_sha256_16": dataset_hash,
        "link_source": link_source,
        "timestamp": "2026-01-01T00:00:00+0000",
        "config": {
            "baseline": {"pipeline": "DeterministicRAGPipeline"},
            "experiment": {
                "agent": "WikiAgent",
                "live": True,
                "base_url": base_url,
                "model": model,
                "max_steps": max_steps,
                "search_top_k": 5,
                "link_source": link_source,
                "api_key_recorded": False,
            },
            "limit": None,
        },
        "source_file_sha256_16": {"benchmark/run_multihop_evaluation.py": "abc"},
        "results": {
            "baseline": {"per_case": [], "aggregate": {}},
            "experiment": {
                "per_case": [r.to_dict() for r in results],
                "aggregate": m4.aggregate_results(CASES, results),
            },
        },
    }


def _manual():
    return _payload("manual", [
        _result("c1", "wrong", steps=3),
        _result("c2", "idk", steps=3, sources=(), recall=None),
    ])


def _auto():
    return _payload("auto", [
        _result("c1", "x", steps=2),
        _result("c2", "cannot say", steps=2, recall=None),
    ])


def test_build_comparison_reports_metric_deltas_and_traces():
    report = cmp.build_comparison(_manual(), _auto())

    assert report["metrics"]["verified_answer_accuracy"] == {
        "manual": 0.0, "auto": 1.0, "delta": 1.0,
    }
    assert report["metrics"]["mean_steps"]["delta"] == -1.0
    assert report["summary"]["answer_correct"]["improved"] == ["c1", "c2"]
    assert report["summary"]["answer_correct"]["regressed"] == []
    assert report["per_case"][0]["auto"]["answer"] == "x"
    assert report["per_case"][0]["auto"]["step_details"] == [
        {"step": 0, "action": "answer"}
    ]
    assert report["provenance"]["dataset_sha256_16"] == "hash1"
    assert report["provenance"]["model"] == "deepseek-chat"


@pytest.mark.parametrize("override", [
    {"model": "other-model"},
    {"base_url": "https://other.example/v1"},
    {"max_steps": 4},
    {"dataset_hash": "hash2"},
])
def test_comparability_rejects_config_or_dataset_mismatch(override):
    manual = _manual()
    auto = _payload("auto", [
        _result("c1", "x", steps=2),
        _result("c2", "cannot say", steps=2, recall=None),
    ], **override)

    issues = cmp.comparability_issues(manual, auto)
    assert issues
    with pytest.raises(ValueError, match="not comparable"):
        cmp.build_comparison(manual, auto)


def test_comparability_rejects_swapped_link_sources():
    manual = _manual()
    auto = _payload("manual", [
        _result("c1", "x", steps=2),
        _result("c2", "cannot say", steps=2, recall=None),
    ])

    issues = cmp.comparability_issues(manual, auto)
    assert any("expected 'auto'" in issue for issue in issues)
    with pytest.raises(ValueError):
        cmp.build_comparison(manual, auto)
