"""Source attribution: the Context that grounded an answer -> its citations.

Attribution is derived, never generated. A Citation is built from the exact
``Chunk`` objects the ContextBuilder packed into the ``Context`` handed to the
Generator -- the same objects that went into the retrieval path -- so a
citation can only ever point at a chunk that really participated in the
answer's context. Nothing here parses the assembled ``Context.text`` to guess
where a passage came from, and nothing here asks a Generator (or any model) to
report its own sources: a model can hallucinate a source, a packing record
cannot.

Two identity levels are kept, because collapsing them loses information: a
Citation carries both its source document identity (``document_id``) and its
chunk identity (``chunk_id``, ``position``). Many chunks may share one
``document_id``; grouping by document is a caller-side ``groupby`` on a field
that is right there, not a nesting this module needs to invent.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .context import Context
from .schema import Chunk, Metadata, _metadata_copy, _nonempty_str

__all__ = ["Citation", "AnswerResult", "citations_from_context"]


@dataclass(frozen=True)
class Citation:
    """One retrieved chunk an answer is grounded in.

    Flat by design: the identifiers are copied out of the ``Chunk`` so a
    citation is self-contained (serializable, displayable) and never drags the
    Chunk -- let alone a Document tree -- along with it. ``text`` is the
    chunk's text *verbatim*; it is never truncated, summarized, or rewritten,
    because a citation that does not quote its source is worse than none.

    Fields
    ------
    document_id: ``Chunk.document_id`` -- source-level identity.
    chunk_id: ``Chunk.id`` -- chunk-level identity, unique per citation.
    position: 0-based index of the chunk within its document.
    text: the chunk's text, verbatim.
    metadata: the chunk's metadata, copied (shallowly) on construction.
    """

    document_id: str
    chunk_id: str
    position: int
    text: str
    metadata: Metadata = field(default_factory=dict)

    def __post_init__(self) -> None:
        _nonempty_str(self.document_id, "document_id")
        _nonempty_str(self.chunk_id, "chunk_id")
        _nonempty_str(self.text, "text")
        if (
            not isinstance(self.position, int)
            or isinstance(self.position, bool)
            or self.position < 0
        ):
            raise TypeError(f"position must be a non-negative int, got {self.position!r}")
        object.__setattr__(self, "metadata", _metadata_copy(self.metadata))

    @classmethod
    def from_chunk(cls, chunk: Chunk) -> "Citation":
        """Build the citation for ``chunk``. The Chunk is read, never mutated."""
        if not isinstance(chunk, Chunk):
            raise TypeError(f"chunk must be a Chunk, got {type(chunk).__name__}")
        return cls(
            document_id=chunk.document_id,
            chunk_id=chunk.id,
            position=chunk.position,
            text=chunk.text,
            metadata=chunk.metadata,
        )


@dataclass(frozen=True)
class AnswerResult:
    """A generated answer plus the sources it was grounded in.

    ``answer`` is the Generator's own return value, passed through verbatim --
    this type never edits, prefixes, or post-processes generated text.
    ``sources`` is the attribution for the very Context that answer was
    generated from, in that Context's order (best-ranked first), deduplicated
    by ``chunk_id``. An empty context yields an empty tuple: no answer, no
    sources, nothing fabricated to fill the gap.
    """

    answer: str
    sources: tuple[Citation, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.answer, str):
            raise TypeError(f"answer must be a str, got {type(self.answer).__name__}")
        if not isinstance(self.sources, tuple):
            raise TypeError(
                f"sources must be a tuple of Citation, got {type(self.sources).__name__}"
            )
        seen: set[str] = set()
        for source in self.sources:
            if not isinstance(source, Citation):
                raise TypeError(
                    f"each source must be a Citation, got {type(source).__name__}"
                )
            if source.chunk_id in seen:
                raise ValueError(f"duplicate chunk_id in sources: {source.chunk_id!r}")
            seen.add(source.chunk_id)


def citations_from_context(context: Context) -> tuple[Citation, ...]:
    """Return one Citation per chunk packed into ``context``, in Context order.

    The Context is the packing record itself -- ``context.chunks`` are the
    exact Chunk objects whose text entries make up ``context.text`` -- so
    attribution is a projection, not an inference: no text is parsed, no
    chunk is re-retrieved, and a chunk absent from the Context cannot appear
    here. Duplicate ``chunk_id``s are collapsed to the first occurrence, so the
    same chunk is never cited twice.
    """
    if not isinstance(context, Context):
        raise TypeError(f"context must be a Context, got {type(context).__name__}")
    citations: list[Citation] = []
    seen: set[str] = set()
    for chunk in context.chunks:
        if chunk.id in seen:
            continue
        seen.add(chunk.id)
        citations.append(Citation.from_chunk(chunk))
    return tuple(citations)
