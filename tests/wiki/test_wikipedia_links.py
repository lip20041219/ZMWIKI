"""Wikipedia link listing tests -- deterministic, no network."""

import json

import pytest

from zmwiki.llm_generators import GeneratorTransportError, HttpResponse
from zmwiki.wiki.wikipedia import (
    MAX_LINK_REQUESTS,
    WikipediaConnector,
    WikipediaResponseError,
    WikipediaTransportError,
)


def _connector(payloads, *, status=200):
    calls = []
    queue = list(payloads)

    def transport(request):
        calls.append(request)
        payload = queue.pop(0) if len(queue) > 1 else queue[0]
        return HttpResponse(status=status, body=json.dumps(payload).encode("utf-8"))

    return WikipediaConnector(transport=transport), calls


def _page(page_id, title, *, ns=0, **extra):
    return {"pageid": page_id, "ns": ns, "title": title, **extra}


def test_read_links_parses_and_filters_targets():
    connector, calls = _connector(
        [
            {
                "query": {
                    "pages": [
                        _page(2, "Beta"),
                        _page(2, "Beta"),  # duplicate
                        _page(10, "Source"),  # self link
                        _page(3, "Gamma"),
                        _page(14, "Category:Things", ns=14),
                        _page(4, "Missing", missing=True),
                        {"ns": 0, "title": "No id"},
                        {"ns": 0, "title": "Bad id", "pageid": -1},
                    ],
                    "redirects": [{"from": "Old Beta", "to": "Beta"}],
                }
            }
        ]
    )

    links = connector.read_links(10)

    assert [(link.page_id, link.title) for link in links] == [
        (2, "Beta"),
        (3, "Gamma"),
    ]
    request_url = calls[0].url
    assert "generator=links" in request_url
    assert "gplnamespace=0" in request_url
    assert "gpllimit=max" in request_url
    assert "pageids=10" in request_url
    assert "redirects=1" in request_url


def test_read_links_follows_pagination():
    connector, calls = _connector(
        [
            {
                "query": {"pages": [_page(2, "Beta")]},
                "continue": {"gplcontinue": "next-token", "continue": "||"},
            },
            {"query": {"pages": [_page(3, "Gamma")]}},
        ]
    )

    links = connector.read_links(10)

    assert [(link.page_id, link.title) for link in links] == [
        (2, "Beta"),
        (3, "Gamma"),
    ]
    assert len(calls) == 2
    assert "gplcontinue=next-token" in calls[1].url
    assert "continue=%7C%7C" in calls[1].url


def test_read_links_fails_loudly_when_budget_is_exhausted():
    looping = {
        "query": {"pages": [_page(2, "Beta")]},
        "continue": {"gplcontinue": "again"},
    }
    connector, calls = _connector([looping])

    with pytest.raises(WikipediaResponseError) as info:
        connector.read_links(10)

    assert "request budget" in str(info.value)
    assert len(calls) == MAX_LINK_REQUESTS


def test_read_links_without_pages_returns_empty():
    connector, _ = _connector([{"batchcomplete": True}])

    assert connector.read_links(10) == []


def test_read_links_http_error_is_reported():
    connector, _ = _connector([{"x": 1}], status=503)

    with pytest.raises(WikipediaResponseError):
        connector.read_links(10)


def test_read_links_transport_error_is_reported():
    def transport(request):
        raise GeneratorTransportError("timed out")

    connector = WikipediaConnector(transport=transport)

    with pytest.raises(WikipediaTransportError):
        connector.read_links(10)
