"""Generation stage: Context -> Answer, plus the prompt it is built from.

A Generator is the last stage of the chain and the only one that produces
user-facing text. It is separate from every retrieval stage: a Generator never
chunks, embeds, indexes, retrieves, fuses, reranks, packs context, or
evaluates. Its whole input is the query plus the ``Context`` a ContextBuilder
already assembled; its whole output is the answer ``str``. It never mutates
the ``Context`` it is given.

``build_prompt`` lives here because generation is the only stage that needs a
prompt: context and question are explicitly delimited sections. There is no
template engine, no prompt registry, and no dependency beyond the stdlib.

Only a deterministic, test-only generator ships here. It is **not** a language
model and must never be presented as one. Real LLM adapters (SDK-backed or
self-hosted) are deliberately not part of the core runtime: they belong behind
this same ``Generator`` contract, carrying their own optional dependency, like
``semantic.py`` / ``semantic_rerank.py`` do for models.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from .context import Context
from .schema import _nonempty_str
from .tokenizers import SimpleTokenizer, Tokenizer

__all__ = ["Generator", "DeterministicGenerator", "build_prompt"]

#: Instruction line of the minimal prompt. Constant so prompts stay greppable
#: and reproducible; a real adapter may replace the whole prompt if it wants.
_INSTRUCTIONS = (
    "Answer the question using only the context below. "
    "If the context does not contain the answer, say you do not know."
)
_CONTEXT_OPEN = "<context>"
_CONTEXT_CLOSE = "</context>"
_QUESTION_OPEN = "<question>"
_QUESTION_CLOSE = "</question>"


class Generator(ABC):
    """Turns a query plus its retrieved Context into the final answer.

    Inputs are the user's ``query`` and the packed :class:`Context` (already
    ordered by the reranker, already budget-bounded by the ContextBuilder); the
    output is the answer text. A Generator answers *only* from what it is
    given -- it never recalls documents, never re-ranks, never re-packs the
    context, and never queries a store/index/model of its own.
    """

    @abstractmethod
    def generate(self, query: str, context: Context) -> str:
        """Return the answer for ``query`` given ``context``."""


class DeterministicGenerator(Generator):
    """Test-only / local-development generator. NOT a language model.

    Extractive, not generative: it returns the ``chunk.text`` of the single
    best-matching context chunk, verbatim -- the scoring is the same
    interpretable lexical signal ``DeterministicReranker`` uses (count of
    distinct query tokens present in the chunk text), applied to pick a passage
    rather than to order a list. Ties keep the earliest chunk, i.e. the
    upstream ranking wins. No model, no network, no randomness: repeated calls
    and independent instances return the same string.

    Because it only ever copies text out of the context it cannot hallucinate;
    with an empty context there is nothing to copy, so the answer is ``""``
    (never a fabricated sentence, never an apology string).
    """

    def __init__(self, tokenizer: Tokenizer | None = None) -> None:
        self.tokenizer = tokenizer if tokenizer is not None else SimpleTokenizer()
        if not isinstance(self.tokenizer, Tokenizer):
            raise TypeError(
                f"tokenizer must be a Tokenizer, got {type(self.tokenizer).__name__}"
            )

    def generate(self, query: str, context: Context) -> str:
        _nonempty_str(query, "query")
        _check_context(context)
        if not context.chunks:
            return ""

        query_terms = set(self.tokenizer.tokenize(query))
        best_index = 0
        best_score = -1
        for index, chunk in enumerate(context.chunks):
            score = sum(1 for t in self.tokenizer.tokenize(chunk.text) if t in query_terms)
            if score > best_score:  # strict: ties keep the earlier (better-ranked) chunk
                best_index, best_score = index, score
        return context.chunks[best_index].text


def build_prompt(query: str, context: Context) -> str:
    """Build the minimal prompt for ``query`` + ``context``.

    Context and question are separate, explicitly delimited sections, so an
    answer can never be confused about where the evidence ends and the
    instruction begins. ``context.text`` is inserted verbatim (already packed
    and numbered by the ContextBuilder); an empty context yields an empty
    context section rather than an omitted one. This is the whole prompt
    system -- no templates, no roles, no variables.
    """
    _nonempty_str(query, "query")
    _check_context(context)
    return (
        f"{_INSTRUCTIONS}\n\n"
        f"{_CONTEXT_OPEN}\n{context.text}\n{_CONTEXT_CLOSE}\n\n"
        f"{_QUESTION_OPEN}\n{query}\n{_QUESTION_CLOSE}"
    )


def _check_context(context: object) -> None:
    if not isinstance(context, Context):
        raise TypeError(f"context must be a Context, got {type(context).__name__}")
