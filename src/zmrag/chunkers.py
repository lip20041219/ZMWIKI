"""Text chunking: Document -> Chunker -> list[Chunk].

Only a deterministic fixed-size chunker is implemented. Chunking is purely
lexical: it never strips or "cleans" the text, never consults a tokenizer,
and never calls an LLM. Same input + same parameters -> same chunks.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from .schema import Chunk, Document

__all__ = ["Chunker", "FixedSizeChunker"]


class Chunker(ABC):
    """Splits one Document into retrievable Chunks.

    Contract: pure and deterministic -- the same Document and parameters
    always yield the same chunks. A Chunker never mutates its input Document
    and never embeds, indexes, retrieves, or calls an LLM.
    """

    @abstractmethod
    def chunk(self, document: Document) -> list[Chunk]:
        """Return the document's text split into ordered Chunks."""


class FixedSizeChunker(Chunker):
    """Deterministic fixed-size chunker.

    Splits ``document.content`` into windows of at most ``chunk_size``
    **characters (Python code points)** -- not tokens; no tokenizer exists in
    ZMRAG yet. Adjacent windows share at most ``overlap`` characters.

    Windows begin every ``step = chunk_size - overlap`` characters and the
    last window may be shorter than ``chunk_size``. With ``overlap == 0`` the
    concatenation of all chunk texts equals the original content exactly:
    chunking slices, it never rewrites or strips text.

    chunk ids are deterministic and unique per document: ``"<id>::<pos>"``.
    ``pos`` is 0-based, emitted in order, and mirrors the Chunk.position.
    """

    def __init__(self, chunk_size: int, overlap: int = 0) -> None:
        self._check_size(chunk_size)
        self._check_overlap(overlap, chunk_size)
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk(self, document: Document) -> list[Chunk]:
        if not isinstance(document, Document):
            raise TypeError(
                f"document must be a Document, got {type(document).__name__}"
            )
        text = document.content
        n = len(text)
        if n == 0:  # defensive: Document already forbids empty content
            return []
        step = self.chunk_size - self.overlap
        chunks: list[Chunk] = []
        pos = 0
        start = 0
        while True:
            end = min(start + self.chunk_size, n)
            chunks.append(
                Chunk(
                    id=f"{document.id}::{pos}",
                    document_id=document.id,
                    text=text[start:end],
                    position=pos,
                    metadata=dict(document.metadata),  # independent per-chunk copy
                )
            )
            pos += 1
            if end == n:
                break
            start += step
        return chunks

    @staticmethod
    def _check_size(value: object) -> None:
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"chunk_size must be an int, got {type(value).__name__}")
        if value <= 0:
            raise ValueError(f"chunk_size must be > 0, got {value}")

    @staticmethod
    def _check_overlap(value: object, chunk_size: int) -> None:
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"overlap must be an int, got {type(value).__name__}")
        if value < 0:
            raise ValueError(f"overlap must be >= 0, got {value}")
        if value >= chunk_size:
            raise ValueError(f"overlap must be < chunk_size ({chunk_size}), got {value}")
