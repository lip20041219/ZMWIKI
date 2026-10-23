"""Raw-text ingestion: TextIngestor -> Document.

The pipeline's front door: converts raw external text into a normalized
``Document`` and nothing more. Text is preserved exactly -- never stripped,
normalized, lowercased, tokenized, or truncated. Identity is explicit and
deterministic: the caller supplies ``document_id``; no id is ever generated
(no UUID / random / timestamp / db id). No chunking, embedding, indexing,
retrieval, or LLM call happens here.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from .schema import Document, Metadata

__all__ = ["TextIngestor", "PlainTextIngestor"]


class TextIngestor(ABC):
    """Entry point converting raw text into a normalized Document.

    Contract: pure and deterministic -- the same ``document_id`` + text +
    metadata always yield the same Document. A TextIngestor never mutates its
    input, never generates an id, and never chunks, embeds, indexes,
    retrieves, or calls an LLM.
    """

    @abstractmethod
    def ingest(
        self,
        document_id: str,
        text: str,
        *,
        metadata: Metadata | None = None,
    ) -> Document:
        """Return a Document for ``text`` under the caller-supplied id."""


class PlainTextIngestor(TextIngestor):
    """Ingests plain text as-is into a ``Document`` of type ``text/plain``.

    ``content_type`` is always ``"text/plain"`` and ``source`` is always
    None -- provenance and other media types belong to later phases' dedicated
    ingestors. Optional caller ``metadata`` is passed straight to Document,
    which validates it is a dict and copies it. All other validation (non-empty
    str ``document_id`` / ``text``, whitespace-only rejected) is delegated to
    ``Document``'s own constructor, so no parallel rule set is maintained.
    """

    def ingest(
        self,
        document_id: str,
        text: str,
        *,
        metadata: Metadata | None = None,
    ) -> Document:
        kwargs = {"metadata": metadata} if metadata is not None else {}
        return Document(
            id=document_id,
            content=text,
            content_type="text/plain",
            **kwargs,
        )
