"""Headless TUI tests for ask-the-wiki through the real WikiAgent loop.

The Agent runs with a sequential fake model transport, so search/read/
follow_links/answer and every terminal status are exercised end to end with
no network and no real LLM.
"""

import asyncio
import json
import threading

from textual.widgets import Input, OptionList, Static

from zmwiki.llm_generators import HttpResponse
from zmwiki.tui.app import ZMWikiApp, main
from zmwiki.web.workspace import load_workspace
from zmwiki.wiki.wikipedia import WikipediaArticle
from zmwiki.wiki.wiki_agent import AgentResult, AgentStatus, WikiAgent
from zmwiki.wiki.wiki_index import WikiIndex
from zmwiki.wiki.wiki_schema import WikiPage


class SequentialFakeTransport:
    def __init__(self, responses):
        self._responses = list(responses)
        self._idx = 0
        self.requests = []

    def __call__(self, request):
        self.requests.append(request)
        if self._idx >= len(self._responses):
            raise RuntimeError("fake model transport exhausted")
        response = self._responses[self._idx]
        self._idx += 1
        if isinstance(response, Exception):
            raise response
        return response


def _completion(action: dict):
    return HttpResponse(
        status=200,
        body=json.dumps(
            {"choices": [{"message": {"role": "assistant",
                                      "content": json.dumps(action)}}]}
        ).encode("utf-8"),
    )


def _factory(responses, **kwargs):
    def build(index):
        return WikiAgent(
            index,
            base_url="https://api.example.com/v1",
            model="test-model",
            transport=SequentialFakeTransport(responses),
            **kwargs,
        )

    return build


def _page(page_id, path, title, content, links_to=(), links_from=()):
    return WikiPage(
        id=page_id, path=path, title=title, content=content,
        links_to=tuple(links_to), links_from=tuple(links_from),
    )


def _single_page_index() -> WikiIndex:
    index = WikiIndex()
    index.upsert(_page(
        "btree", "databases/btree", "B-Tree Index",
        "A B-tree is a balanced tree used for database indexes.",
    ))
    return index


def _app(index, responses, **agent_kwargs) -> ZMWikiApp:
    return ZMWikiApp(index, agent_factory=_factory(responses, **agent_kwargs))


async def _ask(pilot, app, question, *, press_key=True):
    if press_key:
        await pilot.press("a")
    else:
        app.query_one("#ask", Input).focus()
    await pilot.pause()
    ask = app.query_one("#ask", Input)
    assert ask.has_focus
    ask.value = question
    await pilot.pause()
    await pilot.press("enter")
    await app.workers.wait_for_complete()
    await pilot.pause()


def _run(scenario):
    asyncio.run(scenario())


# ---------------------------------------------------------------------------
# Successful answers
# ---------------------------------------------------------------------------


def test_question_triggers_agent_and_displays_answer_and_sources():
    async def scenario():
        responses = [
            _completion({"action": "search", "query": "b-tree"}),
            _completion({"action": "read", "paths": ["databases/btree"]}),
            _completion({
                "action": "answer",
                "text": "A B-tree is a balanced tree used for indexes.",
                "sources": ["btree"],
            }),
        ]
        app = _app(_single_page_index(), responses)
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            await _ask(pilot, app, "What is a B-tree?")

            result = app.last_agent_result
            assert result is not None
            assert result.status == AgentStatus.ANSWERED
            assert result.pages_read == 1
            assert result.read_page_ids == ("btree",)

            assert "What is a B-tree?" in app.query_one("#page-title", Static).content
            reader = app.query_one("#reader", Static).content
            assert "balanced tree" in reader
            assert "B-Tree Index" in reader
            assert "id=btree" in reader
            diagnostics = app.query_one("#page-meta", Static).content
            assert "已回答" in diagnostics
            assert "步骤" in diagnostics
            assert "已读页面" in diagnostics
            assert "来源" in diagnostics
            assert "01 search" in diagnostics

    _run(scenario)


def test_multihop_question_reads_linked_pages_and_cites_both():
    async def scenario():
        index = WikiIndex()
        index.upsert(_page(
            "alice", "people/alice", "Alice Smith",
            "Alice leads engineering.",
            links_to=("project/phoenix",),
        ))
        index.upsert(_page(
            "project/phoenix", "projects/phoenix", "Project Phoenix",
            "Phoenix budget is 500 thousand dollars.",
            links_from=("alice",),
        ))
        responses = [
            _completion({"action": "search", "query": "phoenix budget"}),
            _completion({"action": "read", "paths": ["people/alice"]}),
            _completion({"action": "follow_links", "from_page": "alice"}),
            _completion({
                "action": "answer",
                "text": "Alice leads engineering and Phoenix has a 500 thousand budget.",
                "sources": ["alice", "project/phoenix"],
            }),
        ]
        app = _app(index, responses)
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            await _ask(pilot, app, "Who leads the Phoenix project and its budget?")

            result = app.last_agent_result
            assert result.status == AgentStatus.ANSWERED
            assert result.pages_read == 2
            assert result.read_page_ids == ("alice", "project/phoenix")
            assert result.sources == ("people/alice", "projects/phoenix")

            reader = app.query_one("#reader", Static).content
            assert "Alice Smith (id=alice)" in reader
            assert "Project Phoenix (id=project/phoenix)" in reader
            assert "follow_links" in app.query_one("#page-meta", Static).content
            assert "02 read" in app.query_one("#page-meta", Static).content

    _run(scenario)


def test_cited_sources_that_were_not_read_are_not_displayed():
    async def scenario():
        responses = [
            _completion({"action": "search", "query": "b-tree"}),
            _completion({"action": "read", "paths": ["databases/btree"]}),
            _completion({
                "action": "answer",
                "text": "B-trees index databases.",
                "sources": ["btree", "ghost-page"],
            }),
        ]
        app = _app(_single_page_index(), responses)
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            await _ask(pilot, app, "What is a B-tree?")

            assert app.last_agent_result.sources == ("databases/btree",)
            assert "ghost-page" not in app.query_one("#reader", Static).content
            assert "ghost-page" not in app.query_one("#page-meta", Static).content

    _run(scenario)


def test_empty_workspace_reports_insufficient_evidence():
    async def scenario():
        responses = [
            _completion({"action": "search", "query": "anything"}),
            _completion({
                "action": "answer",
                "text": "I could not confirm this from the local wiki.",
                "sources": [],
            }),
        ]
        app = _app(WikiIndex(), responses)
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            await _ask(pilot, app, "What is anything?", press_key=False)

            result = app.last_agent_result
            assert result.status == AgentStatus.INSUFFICIENT_EVIDENCE
            assert result.pages_read == 0
            reader = app.query_one("#reader", Static).content
            assert "[证据不足]" in reader
            assert "模型备注（未经验证）" in reader
            assert "无经核实的来源" in reader
            assert result.sources == ()

    _run(scenario)


# ---------------------------------------------------------------------------
# Terminal failure statuses
# ---------------------------------------------------------------------------


def test_budget_exhaustion_is_reported():
    async def scenario():
        responses = [_completion({"action": "search", "query": "b-tree"})]
        app = _app(_single_page_index(), responses, max_steps=1)
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            await _ask(pilot, app, "What is a B-tree?")

            result = app.last_agent_result
            assert result.status == AgentStatus.BUDGET_EXHAUSTED
            reader = app.query_one("#reader", Static).content
            assert "[步骤预算耗尽]" in reader
            assert "max_steps=1" in reader
            assert "已回答" not in reader

    _run(scenario)


def test_protocol_error_is_reported():
    async def scenario():
        responses = [
            HttpResponse(status=200, body=json.dumps(
                {"choices": [{"message": {"content": "not json"}}]}
            ).encode("utf-8")),
            HttpResponse(status=200, body=json.dumps(
                {"choices": [{"message": {"content": "still not json"}}]}
            ).encode("utf-8")),
        ]
        app = _app(_single_page_index(), responses)
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            await _ask(pilot, app, "What is a B-tree?")

            result = app.last_agent_result
            assert result.status == AgentStatus.PROTOCOL_ERROR
            assert "协议错误" in app.query_one("#reader", Static).content
            assert "consecutive protocol errors" in app.query_one(
                "#page-meta", Static
            ).content

    _run(scenario)


def test_model_failure_is_reported():
    async def scenario():
        responses = [RuntimeError("connection refused")]
        app = _app(_single_page_index(), responses)
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            await _ask(pilot, app, "What is a B-tree?")

            result = app.last_agent_result
            assert result.status == AgentStatus.MODEL_ERROR
            reader = app.query_one("#reader", Static).content
            assert "[模型错误]" in reader
            assert "connection refused" in reader

    _run(scenario)


# ---------------------------------------------------------------------------
# Read-only guarantees and interaction
# ---------------------------------------------------------------------------


def test_agent_run_does_not_modify_pages_or_save_the_workspace(tmp_path):
    async def scenario():
        workspace = tmp_path / "workspace.json"
        pages = _single_page_index().list_pages()
        workspace.write_text(
            json.dumps({"version": 1, "pages": [p.to_dict() for p in pages]}),
            encoding="utf-8",
        )
        before = workspace.read_text(encoding="utf-8")
        index = _single_page_index()
        responses = [
            _completion({"action": "search", "query": "b-tree"}),
            _completion({"action": "read", "paths": ["databases/btree"]}),
            _completion({
                "action": "answer",
                "text": "A B-tree indexes databases.",
                "sources": ["btree"],
            }),
        ]
        app = ZMWikiApp(
            index,
            workspace_path=workspace,
            agent_factory=_factory(responses),
        )
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            await _ask(pilot, app, "What is a B-tree?")

            assert app.last_agent_result.status == AgentStatus.ANSWERED
            assert workspace.read_text(encoding="utf-8") == before
            assert index.get("btree").content == (
                "A B-tree is a balanced tree used for database indexes."
            )
            assert index.count() == 1

    _run(scenario)


def test_import_and_save_while_agent_is_running(tmp_path):
    """Boundary: the Agent worker reads the index while the UI imports and saves.

    The fake Agent blocks mid-answer. The UI imports a page and saves the
    workspace during that window; the Agent must then observe the new page and
    finish normally, and the saved file must contain both old and new pages.
    """
    async def scenario():
        workspace = tmp_path / "workspace.json"
        index = _single_page_index()
        workspace.write_text(json.dumps({
            "version": 1, "pages": [p.to_dict() for p in index.list_pages()],
        }), encoding="utf-8")

        entered = threading.Event()
        release = threading.Event()
        seen: dict[str, int] = {}

        class BlockingAgent:
            def __init__(self, live_index):
                self.index = live_index

            def answer(self, question):
                seen["before"] = self.index.count()
                entered.set()
                release.wait(timeout=5)
                page = self.index.get("wikipedia-42")
                seen["after"] = self.index.count()
                return AgentResult(
                    answer="The imported page was read.",
                    sources=(page.path,) if page is not None else (),
                    status=AgentStatus.ANSWERED,
                    steps=(),
                    pages_read=1 if page is not None else 0,
                    read_page_ids=(page.id,) if page is not None else (),
                )

        app = ZMWikiApp(
            index, workspace_path=workspace, agent_factory=BlockingAgent,
        )
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            try:
                ask = app.query_one("#ask", Input)
                ask.focus()
                ask.value = "What was imported?"
                await pilot.press("enter")
                await asyncio.wait_for(
                    asyncio.to_thread(entered.wait), timeout=5
                )

                app.query_one("#pages", OptionList).focus()
                app.preview_article = WikipediaArticle(
                    title="Imported", text="Imported body.",
                    url="https://en.wikipedia.org/wiki/Imported", page_id=42,
                )
                app.preview_links = ()
                await pilot.press("i")
                await pilot.pause()
                assert index.get("wikipedia-42") is not None

                await pilot.press("s")  # arm overwrite
                await pilot.press("s")  # save
                await pilot.pause()
            finally:
                release.set()

            await app.workers.wait_for_complete()
            await pilot.pause()

            assert app.last_agent_result.status == AgentStatus.ANSWERED
            assert app.last_agent_result.sources == ("wikipedia/42",)
            assert seen == {"before": 1, "after": 2}
            assert app.index.get("btree") is not None  # original page intact

            reloaded = load_workspace(workspace)
            assert reloaded.get("btree") is not None
            assert reloaded.get("wikipedia-42") is not None
            assert reloaded.check_links().is_clean

    _run(scenario)


def test_escape_returns_to_the_page_view():
    async def scenario():
        responses = [
            _completion({"action": "search", "query": "b-tree"}),
            _completion({"action": "read", "paths": ["databases/btree"]}),
            _completion({
                "action": "answer",
                "text": "A B-tree indexes databases.",
                "sources": ["btree"],
            }),
        ]
        app = _app(_single_page_index(), responses)
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            await _ask(pilot, app, "What is a B-tree?")

            await pilot.press("escape")
            await pilot.pause()

            assert app.mode == "pages"
            assert app.query_one("#page-title", Static).content == "B-Tree Index"
            assert "页面" in app.query_one("#statusbar", Static).content

    _run(scenario)


def test_ask_without_model_configuration_is_reported():
    async def scenario():
        app = ZMWikiApp(_single_page_index())
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            await _ask(pilot, app, "What is a B-tree?")

            status = app.query_one("#statusbar", Static).content
            assert "问答不可用" in status
            assert app.last_agent_result is None

    _run(scenario)


def test_cli_requires_base_url_and_model_together(tmp_path, capsys):
    workspace = tmp_path / "workspace.json"
    workspace.write_text('{"version": 1, "pages": []}', encoding="utf-8")

    assert main(["--workspace", str(workspace), "--base-url", "https://x/v1"]) == 2
    captured_err = capsys.readouterr().err
    assert "同时提供" in captured_err
    assert "--base-url" in captured_err and "--model" in captured_err
