"""Offline tests for the Stage 10 Traditional RAG vs WikiAgent comparison.

All model calls use a fake transport. No real API key or endpoint is touched.
"""

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "benchmark"))

import run_multihop_evaluation as m4  # noqa: E402
import run_stage10_comparison as stage10  # noqa: E402
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


def _case(case_id="c1", question="Who leads Project X?",
          reference="Alice Smith leads Project X.", required=("p1",)) -> dict:
    return {
        "id": case_id,
        "question": question,
        "reference_answer": reference,
        "answer_keyword_groups": [["alice"]],
        "answerable": True,
        "required_page_ids": list(required),
        "hops": 1,
        "question_type": "single_page",
    }


class FakeTransport:
    def __init__(self, responses):
        self.responses = list(responses)
        self.requests = []

    def __call__(self, request):
        self.requests.append(request)
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


def _answer(text="Alice Smith leads Project X.", sources=("p1",)) -> str:
    return json.dumps({"action": "answer", "text": text, "sources": list(sources)})


def _agent_responses(question="Who leads Project X?", path="wiki/p1",
                     text="Alice Smith leads Project X.", source="p1"):
    return [
        json.dumps({"action": "search", "query": question}),
        json.dumps({"action": "read", "paths": [path]}),
        _answer(text, [source]),
    ]


def _payload(request) -> dict:
    return json.loads(request.body.decode("utf-8"))


def test_shared_answer_rules_are_in_both_prompts():
    for line in stage10.SHARED_ANSWER_RULES.splitlines():
        assert line in stage10.RAG_SYSTEM_PROMPT
        assert line in stage10.wiki_agent._SYSTEM_PROMPT


def test_rag_and_agent_use_same_model_and_scorer():
    case = _case()
    index = _index()
    config = stage10.Stage10Config(top_k=1)
    rag_transport = FakeTransport([_answer()])
    agent_transport = FakeTransport(_agent_responses())

    rag = stage10.run_traditional_rag_case(
        case, index, config, transport=rag_transport
    )
    agent = stage10.run_wiki_agent_case(
        case, index, config, transport=agent_transport
    )

    requests = list(rag_transport.requests) + list(agent_transport.requests)
    assert requests
    for request in requests:
        payload = _payload(request)
        assert payload["model"] == "deepseek-flash"
        assert payload["temperature"] == 0.0
        assert request.url.startswith("https://api.deepseek.com")

    assert rag.scoring_rule == agent.scoring_rule == "keyword_group_match"
    assert rag.answer_correct is m4.score_answer(rag.answer, case)
    assert agent.answer_correct is m4.score_answer(agent.answer, case)
    assert rag.status == agent.status == "answered"
    assert rag.sources == agent.sources == ["p1"]


def test_traditional_rag_is_limited_to_top_k_pages():
    index = WikiIndex()
    index.upsert(_page("p1", "Alice Smith", "Alice Smith leads Project X."))
    index.upsert(_page("p2", "Bob Jones", "Bob Jones is an engineer."))
    case = _case()
    transport = FakeTransport([_answer()])

    result = stage10.run_traditional_rag_case(
        case, index, stage10.Stage10Config(top_k=1), transport=transport
    )

    assert result.retrieval_page_ids == ["p1"]
    assert result.read_page_ids == ["p1"]
    assert result.pages_read == 1
    user_prompt = _payload(transport.requests[0])["messages"][1]["content"]
    assert "[p1]" in user_prompt
    assert "[p2]" not in user_prompt


def test_traditional_rag_uses_one_citation_correction():
    case = _case()
    transport = FakeTransport([
        _answer(sources=["p2"]),
        _answer(sources=["p1"]),
    ])

    result = stage10.run_traditional_rag_case(
        case, _index(), stage10.Stage10Config(top_k=1), transport=transport
    )

    assert result.status == "answered"
    assert result.answer_correct is True
    assert result.citation_correction_used is True
    assert result.model_requests == 2


def test_traditional_rag_rejects_tool_actions():
    case = _case()
    result = stage10.run_traditional_rag_case(
        case, _index(), stage10.Stage10Config(top_k=1),
        transport=FakeTransport([
            json.dumps({"action": "search", "query": "more pages"}),
        ]),
    )

    assert result.status == "protocol_error"
    assert result.answer_correct is False
    assert "non-answer action" in (result.error or "")
    assert result.read_page_ids == ["p1"]


def test_traditional_rag_protocol_error_is_retained():
    case = _case()
    result = stage10.run_traditional_rag_case(
        case, _index(), stage10.Stage10Config(top_k=1),
        transport=FakeTransport(["not a JSON action"]),
    )

    assert result.status == "protocol_error"
    assert result.answer_correct is False
    assert result.error
    aggregate = stage10.aggregate_group([case], [result])
    assert aggregate["status_counts"]["protocol_error"] == 1
    assert aggregate["failure_rate"] == 1.0


def test_wiki_agent_model_error_is_retained():
    class RaisingTransport:
        def __call__(self, request):
            raise RuntimeError("fake network failure")

    case = _case()
    result = stage10.run_wiki_agent_case(
        case, _index(), stage10.Stage10Config(top_k=1),
        transport=RaisingTransport(),
    )

    assert result.status == "model_error"
    assert result.answer_correct is False
    assert "fake network failure" in (result.error or "")
    assert result.model_requests == 1


def test_wiki_agent_preserves_tools_and_config():
    case = _case()
    transport = FakeTransport(_agent_responses())
    result = stage10.run_wiki_agent_case(
        case, _index(), stage10.Stage10Config(top_k=1, max_steps=8),
        transport=transport,
    )

    assert result.status == "answered"
    assert [step["action"] for step in result.step_details] == [
        "search", "read", "answer",
    ]
    assert result.step_details[0]["params"]["top_k"] == 1
    assert result.read_page_ids == ["p1"]
    assert result.sources == ["p1"]


def test_checkpoint_resume_skips_completed_case(tmp_path):
    case1 = _case()
    case2 = _case(
        case_id="c2", question="Who is Bob?",
        reference="Bob Jones is an engineer.", required=("p2",),
    )
    index = _index()
    config = stage10.Stage10Config(top_k=1)
    checkpoint = tmp_path / "stage10.checkpoint.jsonl"
    dataset_hash = "a" * 64

    first_rag = FakeTransport([_answer()])
    first_agent = FakeTransport(_agent_responses())
    first = stage10.run_stage10(
        [case1], index, config,
        dataset_sha256=dataset_hash,
        rag_transport=first_rag, agent_transport=first_agent,
        checkpoint_path=checkpoint, resume=False,
    )
    assert len(first) == 1
    assert len(first_rag.requests) == 1
    assert len(first_agent.requests) == 3
    with pytest.raises(FileExistsError):
        stage10.run_stage10(
            [case1], index, config,
            dataset_sha256=dataset_hash,
            rag_transport=FakeTransport([_answer()]),
            agent_transport=FakeTransport(_agent_responses()),
            checkpoint_path=checkpoint, resume=False,
        )

    second_rag = FakeTransport([
        _answer("Bob Jones is an engineer.", ["p2"]),
    ])
    second_agent = FakeTransport(_agent_responses(
        question="Who is Bob?", path="wiki/p2",
        text="Bob Jones is an engineer.", source="p2",
    ))
    second = stage10.run_stage10(
        [case1, case2], index, config,
        dataset_sha256=dataset_hash,
        rag_transport=second_rag, agent_transport=second_agent,
        checkpoint_path=checkpoint, resume=True,
    )
    assert [pair[0]["id"] for pair in second] == ["c1", "c2"]
    assert len(second_rag.requests) == 1
    assert len(second_agent.requests) == 3
    assert len(checkpoint.read_text(encoding="utf-8").splitlines()) == 2


def test_resume_rejects_different_config(tmp_path):
    case = _case()
    checkpoint = tmp_path / "stage10.checkpoint.jsonl"
    dataset_hash = "a" * 64
    stage10.run_stage10(
        [case], _index(), stage10.Stage10Config(top_k=1),
        dataset_sha256=dataset_hash,
        rag_transport=FakeTransport([_answer()]),
        agent_transport=FakeTransport(_agent_responses()),
        checkpoint_path=checkpoint, resume=False,
    )
    with pytest.raises(RuntimeError, match="config hash mismatch"):
        stage10.run_stage10(
            [case], _index(), stage10.Stage10Config(top_k=2),
            dataset_sha256=dataset_hash,
            rag_transport=FakeTransport([_answer()]),
            agent_transport=FakeTransport(_agent_responses()),
            checkpoint_path=checkpoint, resume=True,
        )


def test_serialization_writes_both_groups_and_protects_files(tmp_path):
    case = _case()
    index = _index()
    config = stage10.Stage10Config(top_k=1)
    rag = stage10.run_traditional_rag_case(
        case, index, config, transport=FakeTransport([_answer()])
    )
    agent = stage10.run_wiki_agent_case(
        case, index, config, transport=FakeTransport(_agent_responses())
    )
    dataset = {
        "version": "1.4",
        "scoring_rule": "keyword_group_match",
        "cases": [case],
    }
    output = tmp_path / "stage10_comparison.json"

    payload = stage10.serialize_stage10(
        dataset, config, [(case, rag, agent)], output,
        dataset_sha256="a" * 64,
    )

    assert set(payload["results"]) == {
        "traditional_rag", "wiki_agent", "paired_per_case", "paired_summary",
    }
    assert payload["results"]["traditional_rag"]["aggregate"]["total_cases"] == 1
    assert payload["results"]["wiki_agent"]["aggregate"]["total_cases"] == 1
    assert len(payload["results"]["paired_per_case"]) == 1
    assert (tmp_path / "stage10_comparison_metrics.json").exists()
    assert (tmp_path / "stage10_comparison_run_info.json").exists()
    text = output.read_text(encoding="utf-8")
    assert "Authorization" not in text
    assert "Bearer " not in text

    with pytest.raises(FileExistsError):
        stage10.serialize_stage10(
            dataset, config, [(case, rag, agent)], output,
            dataset_sha256="a" * 64,
        )


def test_dataset_hash_is_verified(monkeypatch):
    dataset, pages, _, dataset_hash = stage10._load_index()
    assert dataset_hash == stage10.EXPECTED_DATASET_SHA256
    assert len(pages) == 9
    monkeypatch.setattr(stage10, "EXPECTED_DATASET_SHA256", "0" * 64)
    with pytest.raises(RuntimeError, match="does not match"):
        stage10._load_index()
