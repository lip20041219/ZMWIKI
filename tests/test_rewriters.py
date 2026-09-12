"""Query rewriter tests: QueryRewriter contract + DeterministicQueryRewriter.

The pipeline side of the contract -- that a rewriter only ever decides the
*retrieval* query, and that a broken one fails instead of falling back -- lives
in ``test_pipeline.py``; this file covers the stage in isolation.
"""

import pytest

from zmrag import DeterministicQueryRewriter, QueryRewriter


def _rewrite(query):
    return DeterministicQueryRewriter().rewrite(query)


# --- contract shape ---


def test_query_rewriter_is_abstract():
    with pytest.raises(TypeError):
        QueryRewriter()  # type: ignore[abstract]
    assert isinstance(DeterministicQueryRewriter(), QueryRewriter)


def test_rewriter_needs_no_configuration():
    # No tokenizer, no model source, no client: a deterministic rewriter has
    # nothing to inject, and an empty constructor is the honest shape for that.
    assert DeterministicQueryRewriter().rewrite("q") == "q"


# --- validation ---


@pytest.mark.parametrize("bad", ["", "   ", " \t\n\r ", " ", 42, None, ["q"]])
def test_invalid_query_rejected(bad):
    with pytest.raises(TypeError):
        _rewrite(bad)  # type: ignore[arg-type]


# --- normalization ---


@pytest.mark.parametrize(
    ("query", "expected"),
    [
        ("apple", "apple"),
        ("  apple  ", "apple"),
        ("apple   pie", "apple pie"),
        ("apple\tpie", "apple pie"),
        ("apple\n\npie", "apple pie"),
        ("apple \t\n pie", "apple pie"),
        ("apple pie", "apple pie"),  # non-breaking space is whitespace
        ("\t\n  spaced  out\tquery \n", "spaced out query"),
    ],
)
def test_whitespace_is_normalized(query, expected):
    assert _rewrite(query) == expected


@pytest.mark.parametrize(
    "query",
    [
        "How does Python work?",
        "Apple, Banana; pie!",
        "café crème",
        "苹果 派",
        "Hybrid retrieval fuses 2 signals.",
        "UPPER lower MiXeD",
    ],
)
def test_everything_but_whitespace_is_left_alone(query):
    # Case, punctuation, accents, CJK, word order and digits all survive: this
    # is a normalization stage, not a tokenizer or a paraphrase.
    assert _rewrite(query) == query
    assert _rewrite(f"  {query}  ") == query


def test_cjk_is_not_split_into_characters():
    # SimpleTokenizer would return ["苹", "果"] for this; a rewrite that did the
    # same would hand the retriever a different query, not a normalized one.
    assert _rewrite("苹果派 怎么做") == "苹果派 怎么做"


def test_output_is_never_empty_for_valid_input():
    assert _rewrite("   x   ").strip()


# --- determinism ---


def test_rewrite_is_deterministic_across_calls_and_instances():
    query = "  How   does Python work?  "
    rewriter = DeterministicQueryRewriter()
    assert rewriter.rewrite(query) == rewriter.rewrite(query)
    assert rewriter.rewrite(query) == DeterministicQueryRewriter().rewrite(query)
    assert rewriter.rewrite(query) == "How does Python work?"


def test_rewrite_is_idempotent():
    once = _rewrite("  spaced   out  ")
    assert _rewrite(once) == once


def test_original_query_string_is_never_modified():
    query = "  keep   me  "
    _rewrite(query)
    assert query == "  keep   me  "  # str is immutable; the rewriter holds no state


# --- not a model ---


def test_not_a_model_no_shape_beyond_the_contract():
    # The stand-in must not pretend to be an LLM: no sampling knobs, no model
    # source, no network client.
    rewriter = DeterministicQueryRewriter()
    for attr in ("model", "model_source", "temperature", "max_tokens", "client"):
        assert not hasattr(rewriter, attr)


# --- public API ---


def test_public_import():
    from zmrag import __all__

    assert {"QueryRewriter", "DeterministicQueryRewriter"} <= set(__all__)
    assert isinstance(DeterministicQueryRewriter(), QueryRewriter)
