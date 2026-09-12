"""Query transformation stage: the user's query -> the retrieval query.

A QueryRewriter sits in front of retrieval and does exactly one thing: turn the
query the user is asking with into the query the retriever searches with. It
never retrieves, embeds, fuses, reranks, packs context, generates, or
attributes -- it takes a string and returns a string. It is the only stage
allowed to change what is being searched *for*.

The *original* query is never overwritten. The rewriter's output is the
retrieval query used by the retrieval path; generation and attribution keep
working from the query the user actually asked (see ``pipeline.py``). A
rewriter can change how a question is looked up, never what was asked.

Only a deterministic, whitespace-normalizing rewriter ships here. It is **not**
a language model, does not understand the query, and must never be presented as
if it did. Real query rewriting -- an LLM rewriter, HyDE, multi-query expansion
-- belongs behind this same contract, carrying its own optional dependency the
way ``semantic.py`` carries sentence-transformers. None of it is part of the
core runtime.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from .schema import _nonempty_str

__all__ = ["QueryRewriter", "DeterministicQueryRewriter"]


class QueryRewriter(ABC):
    """Maps the user's query to the query the retriever searches with.

    One method, one direction: query in, retrieval query out. There is no
    second call, no state carried between queries, and no route from here to a
    retriever, store, or model -- a rewriter that could retrieve would be a
    different stage.
    """

    @abstractmethod
    def rewrite(self, query: str) -> str:
        """Return the retrieval query for ``query``.

        Implementations must not mutate anything, must be safe to call with the
        same query repeatedly, and must return a non-empty ``str``. A
        deterministic implementation returns the same string on every call and
        from every instance.
        """


class DeterministicQueryRewriter(QueryRewriter):
    """Test-only / local-development rewriter. NOT a model, and not "smart".

    Behavior is one line of normalization: trim the query, and collapse every
    run of internal whitespace to a single space (``str.split`` splits on all
    Unicode whitespace, so tabs, newlines and non-breaking spaces are covered
    too). Case, punctuation, and word order are left exactly as typed.

    That is deliberately the whole rewriter. Everything a semantic rewriter
    would do -- synonyms, expansion, hypothetical documents, several queries --
    changes the information need, and nothing in this package can do that
    without a model. Removing typing noise cannot change it, which is what makes
    whitespace normalization defensible as a deterministic baseline rather than
    a fake improvement.

    It stops here rather than at the tokenizer on purpose: SimpleTokenizer
    lowercases, drops punctuation, and splits CJK into single characters, so
    tokenize-then-rejoin would not normalize the query, it would replace it.
    The retrieval query is also what the dense embedder sees, and the lexical
    side tokenizes for itself further down.

    Stateless, so it is stable across calls, instances, and processes.
    """

    def rewrite(self, query: str) -> str:
        _nonempty_str(query, "query")
        return " ".join(query.split())


def _check_rewritten(rewritten: object) -> str:
    """Validate a QueryRewriter's return value at the pipeline boundary.

    A configured rewriter's contract is enforced, never repaired: a broken
    rewrite raises here rather than silently falling back to the original
    query, which would hide the bug behind plausible-looking results.
    """
    if not isinstance(rewritten, str) or not rewritten.strip():
        raise TypeError(
            f"query rewriter output must be a non-empty str, got {rewritten!r}"
        )
    return rewritten
