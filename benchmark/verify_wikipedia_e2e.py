"""Live Wikipedia end-to-end acceptance check (manual, network required).

Deliberately NOT part of the pytest suite: every test in this repository is
offline and deterministic. This script exists to answer one question with
real data: do the existing connector, importer, link resolver, workspace
persistence and agent loop work together?

Model calls: none. The agent stage feeds WikiAgent a scripted fake transport,
so it exercises the real index and real imported pages with a deterministic
"model". That stage is NOT a real-LLM test.

Run:
    PYTHONPATH=src python benchmark/verify_wikipedia_e2e.py

Exit 0 only when every stage passes. Wikipedia network/API failures are
reported as external failures, never as code defects.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from zmwiki.llm_generators import HttpResponse
from zmwiki.web.workspace import load_workspace, save_workspace
from zmwiki.wiki.wikipedia import (
    WikipediaConnector,
    WikipediaError,
)
from zmwiki.wiki.wikipedia_import import import_wikipedia_article
from zmwiki.wiki.wiki_agent import AgentStatus, WikiAgent
from zmwiki.wiki.wiki_index import WikiIndex
from zmwiki.wiki.wiki_retriever import wiki_search

ARTICLE_URLS = (
    "https://en.wikipedia.org/wiki/Bletchley_Park",
    "https://en.wikipedia.org/wiki/Turing_machine",
    "https://en.wikipedia.org/wiki/Alan_Turing",  # imported last, on purpose
)
SEARCH_QUERY = "Alan Turing"


class StageFailure(AssertionError):
    pass


def _check(condition: bool, message: str) -> None:
    if not condition:
        raise StageFailure(message)


def _completion(content: str) -> dict:
    return {"choices": [{"message": {"role": "assistant", "content": content}}]}


def _json_response(payload: dict) -> HttpResponse:
    return HttpResponse(status=200, body=json.dumps(payload).encode("utf-8"))


class ScriptedTransport:
    """Deterministic stand-in for the LLM endpoint; never touches the network."""

    def __init__(self, responses: list[str]) -> None:
        self._responses = list(responses)
        self.calls = 0

    def __call__(self, request) -> HttpResponse:
        self.calls += 1
        if not self._responses:
            raise RuntimeError("scripted transport exhausted")
        return _json_response(_completion(self._responses.pop(0)))


def _agent_stage(index: WikiIndex, page_id: str) -> dict:
    matches = [r for r in wiki_search(index, SEARCH_QUERY, top_k=5)
               if r.page_id == page_id]
    _check(bool(matches), "imported Alan Turing page not found by local search")
    top = matches[0]
    linked = [tid for tid in index.get(top.page_id).links_to
              if index.get(tid) is not None]
    _check(bool(linked), "imported Alan Turing page has no resolved local links")

    script = [
        json.dumps({"action": "search", "query": SEARCH_QUERY}),
        json.dumps({"action": "read", "paths": [top.path]}),
        json.dumps({"action": "follow_links", "from_page": top.page_id}),
        json.dumps({
            "action": "answer",
            "text": "Alan Turing was an English mathematician and computer scientist.",
            "sources": [top.page_id, *linked],
        }),
    ]
    transport = ScriptedTransport(script)
    agent = WikiAgent(
        index, base_url="https://api.example.com/v1", model="scripted",
        api_key="unused", transport=transport,
    )
    result = agent.answer("Who was Alan Turing?")
    _check(result.status == AgentStatus.ANSWERED, f"agent status {result.status}")
    _check(transport.calls == len(script), "agent did not consume every scripted step")
    _check(top.page_id in result.read_page_ids, "agent never read the imported page")
    _check(len(result.sources) >= 2, "agent cited fewer than two read pages")
    return {
        "status": result.status,
        "model_calls": transport.calls,
        "pages_read": result.pages_read,
        "read_page_ids": list(result.read_page_ids),
        "sources": list(result.sources),
        "steps": [(s.action, s.result_summary) for s in result.steps],
    }


def main() -> int:
    conn = WikipediaConnector(timeout=20)
    evidence: dict = {}

    try:
        # 1. Real search.
        hits = conn.search(SEARCH_QUERY, limit=5)
        _check(bool(hits), "search returned no hits")
        _check(any(h.title == "Alan Turing" for h in hits),
               "search did not return the Alan Turing article")
        evidence["search"] = {
            "query": SEARCH_QUERY,
            "hits": [(h.page_id, h.title) for h in hits],
        }
        print(f"[PASS] search: {len(hits)} hits, top={hits[0].title!r}")

        # 2. Real article reads.
        articles = [conn.read_article(url=url) for url in ARTICLE_URLS]
        for article in articles:
            _check(isinstance(article.page_id, int) and article.page_id > 0,
                   "article has no usable page id")
            _check(len(article.text) > 1000, "article text is implausibly short")
        turing = next((a for a in articles if a.title == "Alan Turing"), None)
        _check(turing is not None, "Alan Turing article was not read")
        # Import Alan Turing last so its own links can resolve immediately.
        ordered = [a for a in articles if a is not turing] + [turing]
        evidence["articles"] = [
            {"page_id": a.page_id, "title": a.title, "chars": len(a.text),
             "revision_id": a.revision_id}
            for a in articles
        ]
        print(f"[PASS] read: {len(articles)} articles, "
              f"{sum(len(a.text) for a in articles)} chars")

        # 3. Real internal links.
        links_by_id = {}
        for article in articles:
            links = conn.read_links(article.page_id)
            ids = [link.page_id for link in links]
            _check(bool(links), f"no links returned for {article.title}")
            _check(len(ids) == len(set(ids)), "duplicate link ids returned")
            _check(all(pid > 0 and pid != article.page_id for pid in ids),
                   "invalid or self link returned")
            links_by_id[article.page_id] = links
        link_sets = {pid: {l.page_id for l in links}
                     for pid, links in links_by_id.items()}
        article_ids = set(link_sets)
        cross = {a: sorted(targets & article_ids)
                 for a, targets in link_sets.items()}
        _check(any(cross.values()), "no cross-links between the imported articles")
        evidence["links"] = {
            "counts": {pid: len(ids) for pid, ids in link_sets.items()},
            "cross_links": {str(k): v for k, v in cross.items()},
        }
        print(f"[PASS] links: {sum(len(v) for v in link_sets.values())} links, "
              f"cross-links={cross}")

        # 4. Import in a fixed order (Alan Turing last so its links can resolve).
        index = WikiIndex()
        imports = []
        for article in ordered:
            result = import_wikipedia_article(
                index, article, links=links_by_id[article.page_id]
            )
            imports.append({
                "page": result.page.id,
                "created": result.created,
                "links_resolved": result.links_resolved,
                "links_updated": result.links_updated,
            })
        _check(index.count() == len(articles), "import created a wrong page count")
        evidence["imports"] = imports
        print(f"[PASS] import: {index.count()} pages; "
              f"resolved={[i['links_resolved'] for i in imports]}")

        # 5. Link graph consistency.
        check = index.check_links()
        _check(check.is_clean, f"link check not clean: {check}")
        edges = sorted((p.id, t) for p in index.list_pages() for t in p.links_to)
        backlinks = sorted((p.id, s) for p in index.list_pages() for s in p.links_from)
        _check(len(edges) == len(backlinks),
               "links_to/links_from counts disagree")
        evidence["graph"] = {
            "edges": edges,
            "broken_links": list(check.broken_links),
            "inconsistent_links": list(check.inconsistent_links),
        }
        print(f"[PASS] graph: {len(edges)} directed edges, clean={check.is_clean}")

        # 6. Save / reload round-trip.
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "workspace.json"
            save_workspace(index, path)
            reloaded = load_workspace(path)
            before = [p.to_dict() for p in index.list_pages()]
            after = [p.to_dict() for p in reloaded.list_pages()]
            _check(before == after, "reloaded workspace differs from saved pages")
            _check(reloaded.check_links().is_clean, "reloaded graph is not clean")
            _check(bool(wiki_search(reloaded, SEARCH_QUERY, top_k=3)),
                   "reloaded workspace is not searchable")
            evidence["workspace"] = {
                "bytes": path.stat().st_size,
                "pages": reloaded.count(),
                "searchable": True,
            }
            print(f"[PASS] workspace: {path.stat().st_size} bytes, "
                  f"{reloaded.count()} pages round-tripped")

        # 7. Agent loop over real imported pages with a scripted model.
        agent_evidence = _agent_stage(reloaded, f"wikipedia-{turing.page_id}")
        evidence["agent_scripted"] = agent_evidence
        print(f"[PASS] agent (scripted model): status={agent_evidence['status']}, "
              f"pages_read={agent_evidence['pages_read']}")

    except WikipediaError as exc:
        print(f"[EXTERNAL FAILURE] Wikipedia network/API: "
              f"{type(exc).__name__}: {exc}")
        print("Not a code defect. Re-run later; stage evidence above is partial.")
        return 2
    except StageFailure as exc:
        print(f"[FAIL] acceptance check: {exc}")
        return 1

    evidence["result"] = "PASS"
    evidence["model_api_called"] = False
    print("RESULT: PASS (real network for Wikipedia; agent model scripted)")
    print(json.dumps(evidence, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
