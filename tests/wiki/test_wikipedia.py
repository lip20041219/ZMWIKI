"""Wikipedia connector tests -- deterministic, no network.

Every exchange goes through an injected transport, so these tests assert the
exact request the connector builds and the exact failure it raises. Nothing
here contacts Wikipedia.
"""

import json

import pytest

from zmwiki.llm_generators import GeneratorTransportError, HttpResponse
from zmwiki.wiki import wikipedia
from zmwiki.wiki.wikipedia import (
    MAX_LIMIT,
    MAX_QUERY_LENGTH,
    WikipediaConnector,
    WikipediaNotFoundError,
    WikipediaResponseError,
    WikipediaTransportError,
)

SEARCH_PAYLOAD = {
    "batchcomplete": True,
    "query": {
        "search": [
            {
                "title": "Alan Turing",
                "pageid": 11675,
                "snippet": 'Mathematician <span class="searchmatch">Turing</span> &quot;test&quot;',
            },
            {
                "title": "Turing test",
                "pageid": 16048,
                "snippet": "A test of machine intelligence.",
            },
        ]
    },
}

ARTICLE_PAYLOAD = {
    "query": {
        "pages": [
            {
                "pageid": 11675,
                "ns": 0,
                "title": "Alan Turing",
                "lastrevid": 123456,
                "extract": "Alan Mathison Turing was an English mathematician.",
            }
        ]
    }
}


def _connector(payload=None, *, status=200, body=None, error=None):
    calls = []

    def transport(request):
        calls.append(request)
        if error is not None:
            raise error
        raw = body if body is not None else json.dumps(payload).encode("utf-8")
        return HttpResponse(status=status, body=raw)

    return WikipediaConnector(transport=transport), calls


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------


def test_search_parses_results_and_builds_a_safe_request():
    connector, calls = _connector(SEARCH_PAYLOAD)

    results = connector.search("turing machine", limit=2)

    assert [result.title for result in results] == ["Alan Turing", "Turing test"]
    assert results[0].page_id == 11675
    assert results[0].url == "https://en.wikipedia.org/wiki/Alan_Turing"
    assert results[0].summary == 'Mathematician Turing "test"'
    assert results[1].summary == "A test of machine intelligence."

    assert len(calls) == 1
    request = calls[0]
    assert request.url.startswith(wikipedia.API_URL + "?")
    assert "srsearch=turing+machine" in request.url
    assert "srlimit=2" in request.url
    assert request.headers["User-Agent"] == wikipedia.USER_AGENT
    assert request.timeout == wikipedia.DEFAULT_TIMEOUT


def test_search_with_no_hits_returns_an_empty_list():
    connector, _ = _connector(
        {"batchcomplete": True, "query": {"searchinfo": {"totalhits": 0}}}
    )

    assert connector.search("zzz-no-such-article") == []


@pytest.mark.parametrize("query", ["", "   ", "\n\t"])
def test_blank_query_is_rejected(query):
    connector, calls = _connector(SEARCH_PAYLOAD)

    with pytest.raises(ValueError):
        connector.search(query)

    assert calls == []


def test_non_string_query_is_rejected():
    connector, _ = _connector(SEARCH_PAYLOAD)

    with pytest.raises(TypeError):
        connector.search(123)


def test_overlong_query_is_rejected():
    connector, calls = _connector(SEARCH_PAYLOAD)

    with pytest.raises(ValueError):
        connector.search("x" * (MAX_QUERY_LENGTH + 1))

    assert calls == []


def test_result_limit_is_sent_and_validated():
    connector, calls = _connector(SEARCH_PAYLOAD)
    connector.search("tur", limit=1)
    assert "srlimit=1" in calls[0].url

    with pytest.raises(ValueError):
        connector.search("tur", limit=0)
    with pytest.raises(ValueError):
        connector.search("tur", limit=MAX_LIMIT + 1)
    with pytest.raises(TypeError):
        connector.search("tur", limit="5")


# ---------------------------------------------------------------------------
# Article reading
# ---------------------------------------------------------------------------


def test_read_article_by_page_id():
    connector, calls = _connector(ARTICLE_PAYLOAD)

    article = connector.read_article(page_id=11675)

    assert article.title == "Alan Turing"
    assert article.text == "Alan Mathison Turing was an English mathematician."
    assert article.url == "https://en.wikipedia.org/wiki/Alan_Turing"
    assert article.page_id == 11675
    assert article.revision_id == 123456

    request_url = calls[0].url
    assert "pageids=11675" in request_url
    assert "prop=extracts%7Cinfo" in request_url
    assert "explaintext=1" in request_url


def test_read_article_by_allowed_url_uses_its_title():
    connector, calls = _connector(ARTICLE_PAYLOAD)

    article = connector.read_article(url="https://en.wikipedia.org/wiki/Alan_Turing")

    assert article.title == "Alan Turing"
    assert "titles=Alan+Turing" in calls[0].url


@pytest.mark.parametrize(
    "url",
    [
        "http://en.wikipedia.org/wiki/Alan_Turing",  # not https
        "https://evil.example/wiki/Alan_Turing",  # foreign host
        "https://en.wikipedia.org.evil.example/wiki/x",  # suffix spoof
        "https://en.wikipedia.org/w/index.php?title=x",  # not /wiki/
        "https://en.wikipedia.org/wiki/",  # empty title
    ],
)
def test_disallowed_article_urls_are_rejected_without_a_request(url):
    connector, calls = _connector(ARTICLE_PAYLOAD)

    with pytest.raises(ValueError):
        connector.read_article(url=url)

    assert calls == []


def test_exactly_one_article_identifier_is_required():
    connector, calls = _connector(ARTICLE_PAYLOAD)

    with pytest.raises(ValueError):
        connector.read_article()
    with pytest.raises(ValueError):
        connector.read_article(
            page_id=1, url="https://en.wikipedia.org/wiki/Alan_Turing"
        )

    assert calls == []


def test_missing_page_raises_not_found():
    connector, _ = _connector(
        {"query": {"pages": [{"title": "Nope", "missing": True}]}}
    )

    with pytest.raises(WikipediaNotFoundError):
        connector.read_article(page_id=999999999)


def test_empty_extract_raises_not_found():
    connector, _ = _connector(
        {"query": {"pages": [{"title": "Empty", "pageid": 7, "extract": "   "}]}}
    )

    with pytest.raises(WikipediaNotFoundError):
        connector.read_article(page_id=7)


# ---------------------------------------------------------------------------
# Failures and information leaks
# ---------------------------------------------------------------------------


def test_transport_failure_becomes_a_transport_error():
    connector, _ = _connector(error=GeneratorTransportError("timed out"))

    with pytest.raises(WikipediaTransportError) as info:
        connector.search("turing")

    assert isinstance(info.value.__cause__, GeneratorTransportError)


def test_http_error_is_not_silently_parsed_and_body_is_not_leaked():
    connector, _ = _connector(status=503, body=b'{"detail":"SECRET-BODY"}')

    with pytest.raises(WikipediaResponseError) as info:
        connector.search("turing")

    assert "503" in str(info.value)
    assert "SECRET-BODY" not in str(info.value)


def test_invalid_json_raises_response_error_without_echoing_the_body():
    connector, _ = _connector(body=b"<html>SECRET-BODY</html>")

    with pytest.raises(WikipediaResponseError) as info:
        connector.search("turing")

    assert "SECRET-BODY" not in str(info.value)


def test_api_error_payload_is_reported():
    connector, _ = _connector(
        {"error": {"code": "badvalue", "info": "Unrecognized parameter value"}}
    )

    with pytest.raises(WikipediaResponseError) as info:
        connector.search("turing")

    assert "badvalue" in str(info.value)
    assert "Unrecognized parameter value" in str(info.value)


def test_non_object_json_is_rejected():
    connector, _ = _connector(body=b"[1, 2, 3]")

    with pytest.raises(WikipediaResponseError):
        connector.search("turing")


def test_invalid_connector_configuration_is_rejected():
    with pytest.raises(ValueError):
        WikipediaConnector(timeout=0)
    with pytest.raises(TypeError):
        WikipediaConnector(timeout=True)
    with pytest.raises(TypeError):
        WikipediaConnector(transport="not-callable")
