"""ZMRAG core data contract.

Pipeline data relationship:

    External Input -> Document -> Chunk (-> index / retrieval in later phases)

Document is the normalized base unit; a Chunk is the smallest retrievable
unit a future Chunker cuts a Document into. Both are pure, immutable values
that later stages read from -- no chunking, embedding, indexing, retrieval,
ranking, LLM, or agent logic lives here.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

__all__ = ["Document", "Chunk", "Source", "Metadata"]


# Arbitrary user/metadata bag. Keys are expected to be str; values are any
# JSON-like data. No schema is enforced: it must stay usable for whatever a
# later stage (embedding, retrieval) needs, without binding to any store.
Metadata = dict[str, Any]


def _nonempty_str(value: Any, name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise TypeError(f"{name} must be a non-empty str, got {value!r}")


def _metadata_copy(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise TypeError(f"metadata must be a dict, got {type(value).__name__}")
    return dict(value)


class Source(ABC):
    """Typed origin of a Document.

    An abstraction only at this stage. Concrete source types -- file, URL,
    database, API, generated content -- are added in later phases and carry
    their own fields.
    """

    @property
    @abstractmethod
    def kind(self) -> str:
        """Short discriminator for the concrete source type (e.g. "file")."""


@dataclass(frozen=True)
class Document:
    """An immutable, normalized unit of retrievable text.

    Fields
    ------
    id: caller-supplied stable identifier. Never database-generated. Later
        phases (Chunk.document_id) and ingestion assume it is unique per
        source document. Required, non-empty.
    content: the document body as plain text. Required, non-empty.
    metadata: free-form dict for provenance and retrieval needs (e.g.
        filename, language, custom user fields). Optional; copied on
        construction so later mutation of the caller's dict never leaks in.
    source: typed origin, or None. See Source.
    content_type: open IANA-style media type; default "text/plain". Not an
        allowlist -- the registry is open.

    Frozen by design (see architecture.md): Documents are shared across
    pipeline stages and must not be silently mutated. The metadata dict is
    *shallowly* immutable -- its nested values stay mutable -- which is a
    deliberate trade-off (deep immutability would break user-supplied dicts).
    """

    id: str
    content: str
    metadata: Metadata = field(default_factory=dict)
    source: Source | None = None
    content_type: str = "text/plain"

    def __post_init__(self) -> None:
        _nonempty_str(self.id, "document id")
        _nonempty_str(self.content, "document content")
        _nonempty_str(self.content_type, "content_type")
        if self.source is not None and not isinstance(self.source, Source):
            raise TypeError(
                f"source must be a Source or None, got {type(self.source).__name__}"
            )
        # Copy so the caller's dict and this Document's metadata can't alias.
        object.__setattr__(self, "metadata", _metadata_copy(self.metadata))


@dataclass(frozen=True)
class Chunk:
    """An immutable, smallest retrievable unit cut from a Document.

    One Document yields many Chunks (1 -> N). A Chunk traces back to its
    Document through ``document_id`` only -- it never holds the Document
    instance, keeping the data layer low-coupling and the value objects
    independent.

    Fields
    ------
    id: caller-supplied stable identifier for this chunk, distinct within
        (and across) documents. Not database-generated. A future Chunker is
        expected to derive ids deterministically (e.g. from document_id +
        position), but Chunk itself does not compute one -- the producer owns
        id generation, so no single algorithm is forced here.
    document_id: the ``Document.id`` this chunk belongs to.
    text: non-empty chunk body.
    position: 0-based integer index of this chunk within its document.
    metadata: free-form dict. A future Chunker will *inherit* Document
        metadata into this dict; inheritance itself is not implemented here.

    Same immutability rationale as Document; the metadata dict is shallowly
    immutable only.
    """

    id: str
    document_id: str
    text: str
    position: int
    metadata: Metadata = field(default_factory=dict)

    def __post_init__(self) -> None:
        _nonempty_str(self.id, "chunk id")
        _nonempty_str(self.document_id, "document_id")
        _nonempty_str(self.text, "chunk text")
        if (
            not isinstance(self.position, int)
            or isinstance(self.position, bool)
            or self.position < 0
        ):
            raise TypeError(f"position must be a non-negative int, got {self.position!r}")
        object.__setattr__(self, "metadata", _metadata_copy(self.metadata))
