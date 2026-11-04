"""Offline tests for the Stage 12 multi-round runner.

All model calls use fake transports. No real API key or endpoint is touched.
"""

import copy
import json
import statistics
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "benchmark"))

import run_stage10_comparison as stage10  # noqa: E402
import run_stage12_multiround as stage12  # noqa: E402
from zmwiki.llm_generators import HttpResponse  # noqa: E402
from zmwiki.wiki.wiki_index import WikiIndex  # noqa: E402
from zmwiki.wiki.wiki_schema import WikiPage  # noqa: E402


def _page(page_id: str, title: str, content: str) -> WikiPage:
    return WikiPage(
        id=page_id,
        path=f"wiki/{page_id}",
        title=title,
        content=content,
    )


def _index() -> WikiIndex:
    index = WikiIndex()
    index.upsert(_page("p1", "Alice Smith", "Alice Smith leads Project X."))
    index.upsert(_page("p2", "Bob Jones", "Bob Jones is an engineer."))
    return index


def _case(
    case_id: str, question: str, reference: str, required: str
) -> dict:
    return {
        "id": case_id,
        "question": question,
        "reference_answer": reference,
        "answer_keyword_groups": [["alice"]] if required == "p1" else [["bob"]],
        "answerable": True,
        "required_page_ids": [required],
        "hops": 1,
        "question_type": "single_page",
    }


CASE1 = _case("c1", "Who leads Project X?", "Alice Smith leads Project X.", "p1")
CASE2 = _case("c2", "Who is Bob?", "Bob Jones is an engineer.", "p2")
CASES = [CASE1, CASE2]
INDEX = _index()
DATASET = {
    "version": "1.4",
    "scoring_rule": "keyword_group_match",
    "cases": CASES,
}
DATASET_HASH = "a" * 64


class FakeTransport:
    def __init__(self, responses):
        self.responses = list(responses)
        self.requests = []

    def __call__(self, request):
        self.requests.append(request)
        if not self.responses:
            raise AssertionError("unexpected model request")
        content = self.responses.pop(0)
        return HttpResponse(
            status=200,
            body=json.dumps({
                "choices": [{"message": {"content": content}}],
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 5,
                    "total_tokens": 15,
                    "prompt_cache_hit_tokens": 2,
                    "prompt_cache_miss_tokens": 8,
                },
            }).encode(),
        )


class RaisingTransport:
    def __call__(self, request):
        raise RuntimeError("fake network failure")


def _answer(case: dict, correct: bool) -> str:
    text = case["reference_answer"] if correct else "Unknown."
    return json.dumps({
        "action": "answer",
        "text": text,
        "sources": [case["required_page_ids"][0]],
    })


def _agent_responses(case: dict, correct: bool) -> list[str]:
    page_id = case["required_page_ids"][0]
    return [
        json.dumps({"action": "search", "query": case["question"]}),
        json.dumps({"action": "read", "paths": [f"wiki/{page_id}"]}),
        _answer(case, correct),
    ]


def _transports(rag_correct: dict, agent_correct: dict):
    result = []
    for round_index in range(3):
        rag = FakeTransport([
            _answer(case, rag_correct[case["id"]][round_index])
            for case in CASES
        ])
        agent = FakeTransport([
            response
            for case in CASES
            for response in _agent_responses(
                case, agent_correct[case["id"]][round_index]
            )
        ])
        result.append((rag, agent))
    return result


def _config():
    return stage10.Stage10Config(top_k=1, max_steps=8)


def _run_batch(tmp_path, *, batch_id="stage12_test", rounds=3, resume=False):
    return stage12.run_batch(
        DATASET,
        CASES,
        INDEX,
        _config(),
        rounds=rounds,
        batch_id=batch_id,
        dataset_sha256=DATASET_HASH,
        results_dir=tmp_path,
        resume=resume,
        transports=_transports(
            {
                "c1": [True, True, True],
                "c2": [True, False, False],
            },
            {
                "c1": [True, True, True],
                "c2": [True, True, True],
            },
        )[:rounds],
    )


def test_three_rounds_are_isolated_and_summary_is_consistent(tmp_path):
    summary = _run_batch(tmp_path)

    assert summary["rounds"] == 3
    assert len(set(summary["run_ids"])) == 3
    for index in range(1, 4):
        output = tmp_path / f"stage12_test_r{index}.json"
        assert output.exists()
        payload = json.loads(output.read_text(encoding="utf-8"))
        assert payload["run_id"] == f"stage12_test_r{index}"
        assert payload["round_index"] == index

    rag = summary["groups"]["traditional_rag"]
    assert [row["answer_accuracy"]["value"] for row in rag["per_round"]] == [
        1.0, 0.5, 0.5,
    ]
    assert rag["per_case_consistency"]["unstable_case_ids"] == ["c2"]
    assert rag["per_case_consistency"]["always_correct"] == 1
    assert rag["across_rounds"]["answer_accuracy"]["mean"] == pytest.approx(
        2 / 3
    )
    assert rag["across_rounds"]["answer_accuracy"]["std"] == pytest.approx(
        statistics.stdev([1.0, 0.5, 0.5])
    )

    agent = summary["groups"]["wiki_agent"]
    assert [row["answer_accuracy"]["value"] for row in agent["per_round"]] == [
        1.0, 1.0, 1.0,
    ]
    assert agent["per_case_consistency"]["unstable"] == 0

    paired = summary["paired"]
    assert paired["agent_only_correct"] == 2
    assert paired["rag_only_correct"] == 0
    assert paired["ties"] == 4
    assert paired["mean_agent_minus_rag"] == pytest.approx(1 / 3)
    assert paired["flipped_case_ids"] == ["c2"]


def test_resume_uses_only_the_same_round_checkpoint(tmp_path):
    batch_id = "stage12_resume"
    run_id, _, checkpoint = stage12._round_paths(tmp_path, batch_id, 1)
    config = _config()
    expected = stage12._expected_metadata(config, DATASET_HASH, 1)

    stage12._ensure_round_owner(
        checkpoint,
        stage12._owner_payload(
            batch_id=batch_id,
            run_id=run_id,
            round_index=1,
            rounds=1,
            dataset_sha256=DATASET_HASH,
            expected=expected,
        ),
        resume=False,
    )
    first = stage10.run_stage10(
        [CASE1],
        INDEX,
        config,
        dataset_sha256=DATASET_HASH,
        rag_transport=FakeTransport([_answer(CASE1, True)]),
        agent_transport=FakeTransport(_agent_responses(CASE1, True)),
        checkpoint_path=checkpoint,
        resume=False,
    )
    assert len(first) == 1

    rag = FakeTransport([_answer(CASE2, True)])
    agent = FakeTransport(_agent_responses(CASE2, True))
    payload = stage12.run_round(
        DATASET,
        CASES,
        INDEX,
        config,
        round_index=1,
        rounds=1,
        batch_id=batch_id,
        dataset_sha256=DATASET_HASH,
        results_dir=tmp_path,
        resume=True,
        rag_transport=rag,
        agent_transport=agent,
    )
    assert len(rag.requests) == 1
    assert len(agent.requests) == 3
    assert [case["id"] for case in payload["results"]["wiki_agent"]["per_case"]] == [
        "c1", "c2",
    ]


def test_checkpoint_owner_rejects_a_different_round(tmp_path):
    _, _, checkpoint = stage12._round_paths(tmp_path, "stage12_owner", 2)
    checkpoint.write_text("{}\n", encoding="utf-8")
    expected = stage12._expected_metadata(_config(), DATASET_HASH, 2)
    wrong_owner = stage12._owner_payload(
        batch_id="stage12_owner",
        run_id="stage12_owner_r1",
        round_index=1,
        rounds=2,
        dataset_sha256=DATASET_HASH,
        expected=expected,
    )
    stage12._owner_path(checkpoint).write_text(
        json.dumps(wrong_owner), encoding="utf-8"
    )

    with pytest.raises(RuntimeError, match="owner mismatch"):
        stage12._ensure_round_owner(
            checkpoint,
            stage12._owner_payload(
                batch_id="stage12_owner",
                run_id="stage12_owner_r2",
                round_index=2,
                rounds=2,
                dataset_sha256=DATASET_HASH,
                expected=expected,
            ),
            resume=True,
        )


def test_checkpoint_without_owner_is_rejected_on_resume(tmp_path):
    _, _, checkpoint = stage12._round_paths(tmp_path, "stage12_noowner", 1)
    checkpoint.write_text("{}\n", encoding="utf-8")
    expected = stage12._expected_metadata(_config(), DATASET_HASH, 1)
    with pytest.raises(RuntimeError, match="without owner"):
        stage12._ensure_round_owner(
            checkpoint,
            stage12._owner_payload(
                batch_id="stage12_noowner",
                run_id="stage12_noowner_r1",
                round_index=1,
                rounds=1,
                dataset_sha256=DATASET_HASH,
                expected=expected,
            ),
            resume=True,
        )


def test_existing_round_and_summary_are_never_overwritten(tmp_path):
    _run_batch(tmp_path, rounds=1)

    with pytest.raises(FileExistsError):
        stage12.run_round(
            DATASET,
            CASES,
            INDEX,
            _config(),
            round_index=1,
            rounds=1,
            batch_id="stage12_test",
            dataset_sha256=DATASET_HASH,
            results_dir=tmp_path,
            resume=False,
        )
    with pytest.raises(FileExistsError):
        _run_batch(tmp_path, rounds=1, resume=False)


def test_provenance_mismatch_is_rejected(tmp_path):
    _run_batch(tmp_path, rounds=1)
    output = tmp_path / "stage12_test_r1.json"
    payload = json.loads(output.read_text(encoding="utf-8"))
    expected = stage12._expected_metadata(_config(), DATASET_HASH, 1)

    mutated = copy.deepcopy(payload)
    mutated["source_file_sha256_16"] = {"changed.py": "0" * 16}
    with pytest.raises(RuntimeError, match="source_file_sha256_16 mismatch"):
        stage12._verify_round_payload(mutated, expected, payload["run_id"])

    mutated = copy.deepcopy(payload)
    mutated["answer_generation"]["agent_prompt_sha256_16"] = "0" * 16
    with pytest.raises(RuntimeError, match="prompt .* mismatch"):
        stage12._verify_round_payload(mutated, expected, payload["run_id"])


def test_dry_run_sends_no_model_request(tmp_path, monkeypatch, capsys):
    def _network(*args, **kwargs):
        raise AssertionError("network must not be used during dry run")

    class _NoLogger:
        def __init__(self, *args, **kwargs):
            raise AssertionError("logger must not be created during dry run")

    monkeypatch.setattr(stage10, "urllib_transport", _network)
    monkeypatch.setattr(stage10, "RunLogger", _NoLogger)
    stage12.main([
        "--output-dir", str(tmp_path),
        "--batch-id", "stage12_dry",
        "--limit", "1",
    ])
    output = capsys.readouterr().out
    assert "DRY RUN: no model request sent." in output
    assert list(tmp_path.iterdir()) == []


def test_missing_usage_is_flagged_not_zero_imputed():
    case = CASE1
    result = stage10.run_wiki_agent_case(
        case, INDEX, _config(), transport=RaisingTransport()
    )
    assert result.status == "model_error"
    assert result.model_requests == 1
    assert result.total_tokens == 0
    aggregate = stage10.aggregate_group([case], [result])
    payload = {
        "results": {
            "wiki_agent": {
                "aggregate": aggregate,
                "per_case": [result.to_dict()],
            }
        }
    }
    metrics = stage12._round_metrics(payload, "wiki_agent")
    assert metrics["usage_missing_calls"] == 1
    assert metrics["usage_missing_case_ids"] == [case["id"]]
    assert metrics["total_tokens"] == 0


def test_wilson_and_cross_round_statistics():
    interval = stage12.wilson_interval(44, 50)
    assert interval["low"] < 0.88 < interval["high"]
    stats = stage12._summary_stats([0.8, 0.9, 1.0])
    assert stats["mean"] == pytest.approx(0.9)
    assert stats["std"] == pytest.approx(0.1)
    assert stats["ci95"]["low"] == pytest.approx(0.9 - 4.303 * 0.1 / (3 ** 0.5))
