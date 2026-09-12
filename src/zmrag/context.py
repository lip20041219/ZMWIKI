"""Context building: ranked SearchResult[] -> Context (packed, within budget).

ContextBuilder is the adapter between retrieval/reranking and the Generator
(``generators.py``). It receives already-ordered SearchResults and assembles
the chunks that fit into a character budget into one structured Context. It
never retrieves, reranks, re-scores, summarizes, compresses, generates prompts,
or calls a model. The reranker decides the order; ContextBuilder only packs.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from .schema import Chunk
from .stores import SearchResult

__all__ = ["Context", "ContextBuilder", "DeterministicContextBuilder"]

#: Separator placed between packed chunk entries. It counts toward the budget.
_SEPARATOR = "\n\n"


@dataclass(frozen=True)
class Context:
    """The text handed to a Generator, plus the Chunks it came from.

    ``text`` is the assembled, budget-bounded, human-readable context string.
    ``chunks`` is the ordered tuple of the exact Chunk objects whose text
    entries appear in ``text`` -- so provenance is always recoverable (a chunk
    at ``chunks[i]`` corresponds to the ``[i+1]`` entry). ``Context`` is
    frozen and ``chunks`` is a tuple: no mutation, no aliased mutable list.
    """

    text: str
    chunks: tuple[Chunk, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.text, str):
            raise TypeError(f"text must be a str, got {type(self.text).__name__}")
        if not isinstance(self.chunks, tuple):
            raise TypeError(
                f"chunks must be a tuple of Chunk, got {type(self.chunks).__name__}"
            )
        for chunk in self.chunks:
            if not isinstance(chunk, Chunk):
                raise TypeError(
                    f"each chunk must be a Chunk, got {type(chunk).__name__}"
                )


class ContextBuilder(ABC):
    """Turns ordered SearchResults into one budget-bounded Context.

    Ordering, scoring, and duplicate-removal belong to the upstream retriever /
    fusion / reranker; this stage preserves input order and filters only by the
    character budget. It never sorts, never re-scores, never re-retrieves, and
    never produces a prompt.
    """

    @abstractmethod
    def build(self, results: list[SearchResult], max_chars: int) -> Context:
        """Pack ``results`` (in order) into a Context of at most ``max_chars``."""


class DeterministicContextBuilder(ContextBuilder):
    """Ordered greedy packing. Not a model, no summarization/compression.

    Chunk texts are emitted in input order, each as a ``[n] ``-numbered entry,
    joined by blank lines. ``max_chars`` is a budget on the *final*
    ``Context.text`` length in Python characters/code points (separators and
    numbering included) -- not an LLM token budget (no token counting exists
    in ZMRAG). If a chunk would push the text over budget the packing stops
    there; the only exception is the very first chunk, which is kept even when
    it alone exceeds ``max_chars``, so a semantic unit is never truncated.
    """

    def build(self, results: list[SearchResult], max_chars: int) -> Context:
        _check_results(results)
        _check_max_chars(max_chars)

        text = ""
        selected: list[Chunk] = []
        for result in results:
            entry = f"[{len(selected) + 1}] {result.chunk.text}"
            candidate = entry if not text else f"{text}{_SEPARATOR}{entry}"
            if text and len(candidate) > max_chars:
                break
            selected.append(result.chunk)
            text = candidate
        return Context(text=text, chunks=tuple(selected))


def _check_results(results: object) -> None:
    if not isinstance(results, list):
        raise TypeError(f"results must be a list, got {type(results).__name__}")
    seen: set[str] = set()
    for result in results:
        if not isinstance(result, SearchResult):
            raise TypeError(
                f"each result must be a SearchResult, got {type(result).__name__}"
            )
        if result.chunk.id in seen:
            raise ValueError(f"duplicate chunk.id in results: {result.chunk.id!r}")
        seen.add(result.chunk.id)


def _check_max_chars(max_chars: object) -> None:
    if isinstance(max_chars, bool) or not isinstance(max_chars, int):
        raise TypeError(f"max_chars must be an int, got {type(max_chars).__name__}")
    if max_chars < 1:
        raise ValueError(f"max_chars must be >= 1, got {max_chars}")
