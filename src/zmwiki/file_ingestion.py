"""Local-file ingestion: FileIngestor -> Document.

The file front door for the pipeline: reads a local plain-text ``.txt`` file
into a normalized ``Document``. Reading only, never cleaning -- content is
returned exactly as the file decodes (no strip, no lower, no normalization,
no truncation). Only UTF-8; a decode failure raises rather than silently
falling back. Identity stays explicit and deterministic: the caller supplies
``document_id``. No directory / batch / PDF / Markdown / HTML / URL ingestion
lives here.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

from .schema import Document, Metadata, Source

__all__ = ["FileIngestor", "FileSource", "PlainTextFileIngestor"]


class FileIngestor(ABC):
    """Entry point converting a local file into a normalized Document.

    Contract: pure and deterministic -- the same file + ``document_id`` +
    metadata always yield the same Document. A FileIngestor never mutates its
    input, never generates an id, and never chunks, embeds, indexes,
    retrieves, or calls an LLM. Validation is delegated to ``Document``'s
    constructor and to the file system; no parallel rule set is maintained.
    """

    @abstractmethod
    def ingest(
        self,
        path: str | Path,
        *,
        document_id: str,
        metadata: Metadata | None = None,
    ) -> Document:
        """Return a Document built from the file at ``path``."""


@dataclass(frozen=True)
class FileSource(Source):
    """Minimal typed provenance: a Document read from a local file.

    Pure provenance only -- carries the ``path`` the Document came from and
    nothing else. It never reads, parses, or ingests; no database, no
    third-party dependency.
    """

    path: Path

    def __post_init__(self) -> None:
        # Normalize so FileSource("a.txt") == FileSource(Path("a.txt")).
        if not isinstance(self.path, Path):
            object.__setattr__(self, "path", Path(self.path))

    @property
    def kind(self) -> str:
        return "file"


class PlainTextFileIngestor(FileIngestor):
    """Ingests a ``.txt`` plain-text file into a Document.

    Accepts a ``str`` or ``Path``; extension is matched case-insensitively
    (``.txt`` / ``.TXT`` / ``.Txt``). The file is decoded as UTF-8; content is
    preserved exactly (equal to ``path.read_text(encoding="utf-8")``). Any
    other extension is refused -- nothing is parsed and no parsing library is
    pulled in. ``content_type`` is ``"text/plain"``; provenance is a
    ``FileSource(path)``. Optional caller ``metadata`` passes straight to
    Document, which validates and copies it.
    """

    def ingest(
        self,
        path: str | Path,
        *,
        document_id: str,
        metadata: Metadata | None = None,
    ) -> Document:
        p = self._path_of(path)
        if p.is_dir():
            raise IsADirectoryError(f"path is a directory, not a file: {p}")
        if p.suffix.lower() != ".txt":
            raise ValueError(
                f"unsupported extension {p.suffix!r}: only .txt plain-text files are supported"
            )
        content = p.read_text(encoding="utf-8")  # missing -> FileNotFoundError; bad bytes -> UnicodeDecodeError
        kwargs = {"metadata": metadata} if metadata is not None else {}
        return Document(
            id=document_id,
            content=content,
            content_type="text/plain",
            source=FileSource(p),
            **kwargs,
        )

    @staticmethod
    def _path_of(path: object) -> Path:
        if isinstance(path, bool) or not isinstance(path, (str, Path)):
            raise TypeError(f"path must be a str or Path, got {type(path).__name__}")
        return Path(path)
