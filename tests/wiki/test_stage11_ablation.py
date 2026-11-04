"""Offline tests for the Stage 11 Follow Links ablation.

All model calls use a fake transport. No real API key or endpoint is touched.
"""

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "benchmark"))

import run_multihop_evaluation as m4  # noqa: E402
import run_stage11_ablation as stage11  # noqa: E402
from zmwiki.llm_generators import HttpResponse  # noqa: E402
from zmwiki.wiki.wiki_index import WikiIndex  # noqa: E402
from zmwiki.wiki.wiki_schema import WikiPage  # noqa: E402


def _page(page_id: str, content: str, links_to=()) -> WikiPage:
    return WikiPage(
        id=page_id,
        path=f"wiki/{page_id}",
        title=page_id,
        content=content,
        links_to=tuple(links_to),
    )


def _index() -> WikiIndex:
    index = WikiIndex()
    index.upsert(_page("p1", "Alice leads Project X.", ("p2",)))
    index.upsert(_page("p2", "Bob is an engineer."))
    return index


def _case(case_id="c1") -> dict:
    return {
        "id": case_id,
        "question": "Who is Bob?",
        "reference_answer": "Bob is an engineer.",
        "answer_keyword_groups": [["bob"]],
        "answerable": True,
        "required_page_ids": ["p2"],
        "hops": 2,
        "question_type": "two_page",
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
                },
            }).encode(),
        )


def _answer() -> str:
    return json.dumps({
        "action": "answer",
        "text": "Bob is an engineer.",
        "sources": ["p2"],
    })


def _full_responses() -> list[str]:
    return [
        json.dumps({"action": "search", "query": "Who is Bob?"}),
        json.dumps({"action": "read", "paths": ["wiki/p1"]}),
        json.dumps({"action": "follow_links", "from_page": "p1"}),
        _answer(),
    ]


def _no_links_responses() -> list[str]:
    return [
        json.dumps({"action": "search", "query": "Who is Bob?"}),
        json.dumps({"action": "read", "paths": ["wiki/p1"]}),
        json.dumps({"action": "follow_links", "from_page": "p1"}),
        json.dumps({"action": "read", "paths": ["wiki/p2"]}),
        _answer(),
    ]


def _payload(request) -> dict:
    return json.loads(request.body.decode("utf-8"))


def test_no_links_prompt_only_drops_follow_links():
    assert '"action": "follow_links"' in stage11.FULL_SYSTEM_PROMPT
    assert '"action": "follow_links"' not in stage11.NO_LINKS_SYSTEM_PROMPT
    for line in stage11.m10.SHARED_ANSWER_RULES.splitlines():
        assert line in stage11.FULL_SYSTEM_PROMPT
        assert line in stage11.NO_LINKS_SYSTEM_PROMPT
    assert stage11.FULL_SYSTEM_PROMPT != stage11.NO_LINKS_SYSTEM_PROMPT


def test_default_wiki_agent_still_allows_follow_links():
    from zmwiki.wiki.wiki_agent import _SYSTEM_PROMPT, _VALID_ACTIONS, WikiAgent

    agent = WikiAgent(
        _index(),
        base_url="https://api.deepseek.com",
        model="deepseek-flash",
        transport=FakeTransport([]),
    )
    assert agent.allowed_actions == _VALID_ACTIONS
    assert agent.system_prompt == _SYSTEM_PROMPT
    assert "follow_links" in agent.allowed_actions


def test_full_can_follow_links():
    case = _case()
    result = stage11.run_full_case(
        case, _index(), stage11.Stage11Config(), transport=FakeTransport(
            _full_responses()
        )
    )

    assert result.status == "answered"
    assert result.answer_correct is True
    assert result.follow_links_calls == 1
    assert result.follow_links_attempts == 1
    assert result.follow_links_pages == ["p2"]
    assert "follow_links" in [step["action"] for step in result.step_details]
    assert result.scoring_rule == "keyword_group_match"
    assert result.answer_correct is m4.score_answer(result.answer, case)


def test_no_links_blocks_follow_links_but_search_and_read_work():
    case = _case()
    result = stage11.run_no_links_case(
        case, _index(), stage11.Stage11Config(), transport=FakeTransport(
            _no_links_responses()
        )
    )

    actions = [step["action"] for step in result.step_details]
    assert actions == ["search", "read", "tool_disabled", "read", "answer"]
    assert result.status == "answered"
    assert result.answer_correct is True
    assert result.follow_links_calls == 0
    assert result.follow_links_attempts == 1
    assert result.follow_links_pages == []
    assert result.tool_disabled_attempts == 1
    assert result.read_page_ids == ["p1", "p2"]
    assert "follow_links" not in actions


def test_no_links_cannot_read_linked_page_without_explicit_read():
    case = _case()
    responses = [
        json.dumps({"action": "search", "query": "Who is Bob?"}),
        json.dumps({"action": "read", "paths": ["wiki/p1"]}),
        json.dumps({"action": "follow_links", "from_page": "p1"}),
        _answer(),
        _answer(),  # second attempt still cites the unread p2; agent then fails closed
    ]
    result = stage11.run_no_links_case(
        case, _index(), stage11.Stage11Config(),
        transport=FakeTransport(responses),
    )

    assert result.status == "insufficient_evidence"
    assert result.grounded_correct is False
    assert result.follow_links_calls == 0
    assert result.follow_links_attempts == 1
    assert result.follow_links_pages == []
    assert "p2" not in result.read_page_ids
    assert result.sources == []


def test_full_and_no_links_use_same_model_config_and_scorer():
    case = _case()
    config = stage11.Stage11Config()
    full_transport = FakeTransport(_full_responses())
    no_links_transport = FakeTransport(_no_links_responses())

    full = stage11.run_full_case(case, _index(), config, transport=full_transport)
    no_links = stage11.run_no_links_case(
        case, _index(), config, transport=no_links_transport
    )

    requests = list(full_transport.requests) + list(no_links_transport.requests)
    for request in requests:
        payload = _payload(request)
        assert payload["model"] == "deepseek-flash"
        assert payload["temperature"] == 0.0
        assert request.url.startswith("https://api.deepseek.com")

    assert full.scoring_rule == no_links.scoring_rule == "keyword_group_match"
    assert full.answer_correct is m4.score_answer(full.answer, case)
    assert no_links.answer_correct is m4.score_answer(no_links.answer, case)


def test_checkpoint_resume_skips_completed_case(tmp_path):
    case1 = _case("c1")
    case2 = _case("c2")
    config = stage11.Stage11Config()
    checkpoint = tmp_path / "stage11.checkpoint.jsonl"
    dataset_hash = "a" * 64

    first_full = FakeTransport(_full_responses())
    first_no = FakeTransport(_no_links_responses())
    first = stage11.run_stage11(
        [case1], _index(), config,
        dataset_sha256=dataset_hash,
        full_transport=first_full, no_links_transport=first_no,
        checkpoint_path=checkpoint, resume=False,
    )
    assert len(first) == 1
    assert len(first_full.requests) == 4
    assert len(first_no.requests) == 5
    with pytest.raises(FileExistsError):
        stage11.run_stage11(
            [case1], _index(), config,
            dataset_sha256=dataset_hash,
            full_transport=FakeTransport(_full_responses()),
            no_links_transport=FakeTransport(_no_links_responses()),
            checkpoint_path=checkpoint, resume=False,
        )

    second_full = FakeTransport(_full_responses())
    second_no = FakeTransport(_no_links_responses())
    second = stage11.run_stage11(
        [case1, case2], _index(), config,
        dataset_sha256=dataset_hash,
        full_transport=second_full, no_links_transport=second_no,
        checkpoint_path=checkpoint, resume=True,
    )
    assert [pair[0]["id"] for pair in second] == ["c1", "c2"]
    assert len(second_full.requests) == 4
    assert len(second_no.requests) == 5
    assert len(checkpoint.read_text(encoding="utf-8").splitlines()) == 2


def test_resume_rejects_different_config(tmp_path):
    case = _case()
    checkpoint = tmp_path / "stage11.checkpoint.jsonl"
    dataset_hash = "a" * 64
    stage11.run_stage11(
        [case], _index(), stage11.Stage11Config(),
        dataset_sha256=dataset_hash,
        full_transport=FakeTransport(_full_responses()),
        no_links_transport=FakeTransport(_no_links_responses()),
        checkpoint_path=checkpoint, resume=False,
    )
    with pytest.raises(RuntimeError, match="config hash mismatch"):
        stage11.run_stage11(
            [case], _index(), stage11.Stage11Config(top_k=2),
            dataset_sha256=dataset_hash,
            full_transport=FakeTransport(_full_responses()),
            no_links_transport=FakeTransport(_no_links_responses()),
            checkpoint_path=checkpoint, resume=True,
        )


def test_serialization_writes_both_groups_and_protects_files(tmp_path):
    case = _case()
    index = _index()
    config = stage11.Stage11Config()
    full = stage11.run_full_case(
        case, index, config, transport=FakeTransport(_full_responses())
    )
    no_links = stage11.run_no_links_case(
        case, index, config, transport=FakeTransport(_no_links_responses())
    )
    dataset = {
        "version": "1.4",
        "scoring_rule": "keyword_group_match",
        "cases": [case],
    }
    output = tmp_path / "stage11_ablation.json"

    payload = stage11.serialize_stage11(
        dataset, config, [(case, full, no_links)], output,
        dataset_sha256="a" * 64,
    )

    assert set(payload["results"]) == {
        "full", "no_links", "paired_per_case", "paired_summary",
    }
    assert payload["results"]["full"]["per_case"][0]["follow_links_calls"] == 1
    assert payload["results"]["no_links"]["per_case"][0]["follow_links_calls"] == 0
    assert len(payload["results"]["paired_per_case"]) == 1
    assert (tmp_path / "stage11_ablation_metrics.json").exists()
    assert (tmp_path / "stage11_ablation_run_info.json").exists()
    text = output.read_text(encoding="utf-8")
    assert "Authorization" not in text
    assert "Bearer " not in text

    with pytest.raises(FileExistsError):
        stage11.serialize_stage11(
            dataset, config, [(case, full, no_links)], output,
            dataset_sha256="a" * 64,
        )


def test_dry_run_dataset_is_complete_and_unique():
    dataset, pages, _, dataset_hash = stage11._load_index()
    assert dataset_hash == stage11.m10.EXPECTED_DATASET_SHA256
    assert len(dataset["cases"]) == 50
    assert len(pages) == 9
    assert len({case["id"] for case in dataset["cases"]}) == 50
