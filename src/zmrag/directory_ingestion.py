"""Directory / batch ingestion: DirectoryIngestor -> list[Document].

The batch front door of the pipeline: discovers the ``.txt`` files in one
directory, orders them deterministically, and delegates reading to a
``FileIngestor`` (``PlainTextFileIngestor``) -- one Document per file.

This module only *discovers, orders, and delegates*. It never reads file bytes,
never constructs a Document, and never parses or cleans text. Only the current
directory is scanned (no recursion); non-``.txt`` files are ignored, not
errored on.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from .file_ingestion import FileIngestor, PlainTextFileIngestor
from .schema import Document, Metadata

__all__ = ["DirectoryIngestor", "PlainTextDirectoryIngestor"]


class DirectoryIngestor(ABC):
    """Entry point ingesting every ``.txt`` in a directory as Documents.

    Contract: pure and deterministic -- the same directory contents +
    ``document_id_prefix`` + metadata always yield the same Documents, in the
    same order, with the same ids. A DirectoryIngestor never mutates its
    input, never generates an id from discovery order or absolute paths, and
    never chunks, embeds, indexes, retrieves, or calls an LLM.
    """

    @abstractmethod
    def ingest(
        self,
        directory: str | Path,
        *,
        document_id_prefix: str = "",
        metadata: Metadata | None = None,
    ) -> list[Document]:
        """Return one Document per ``.txt`` file in ``directory``, sorted."""


class PlainTextDirectoryIngestor(DirectoryIngestor):
    """Ingests every ``.txt`` file in one directory as ``Document``s.

    Responsibilities are split strictly:

    - discovery + deterministic ordering + batch delegation happen here;
    - file reading + Document construction are delegated to the injected
      ``FileIngestor`` (``PlainTextFileIngestor`` by default) -- no reading or
      Document logic is duplicated here.

    Ordering sorts by ``(filename.casefold(), filename)`` so output is stable
    regardless of filesystem iteration or creation order. Document ids are
    ``"<prefix>/<filename>"`` (or just ``"<filename>"`` when ``prefix`` is
    empty) -- derived from the relative filename, never the absolute path, so
    they survive the directory being moved. Only the current directory is
    scanned; subdirectories and non-``.txt`` files are ignored. Per-file
    failures (decode errors, empty files) propagate unchanged.
    """

    def __init__(self, file_ingestor: FileIngestor | None = None) -> None:
        self._file_ingestor = (
            file_ingestor if file_ingestor is not None else PlainTextFileIngestor()
        )

    def ingest(
        self,
        directory: str | Path,
        *,
        document_id_prefix: str = "",
        metadata: Metadata | None = None,
    ) -> list[Document]:
        d = self._require_directory(directory)
        documents: list[Document] = []
        for name in self._discover(d):
            documents.append(
                self._file_ingestor.ingest(
                    d / name,
                    document_id=self._doc_id(document_id_prefix, name),
                    metadata=metadata,
                )
            )
        return documents

    @staticmethod
    def _require_directory(directory: object) -> Path:
        if isinstance(directory, bool) or not isinstance(directory, (str, Path)):
            raise TypeError(
                f"directory must be a str or Path, got {type(directory).__name__}"
            )
        p = Path(directory)
        if not p.exists():
            raise FileNotFoundError(f"directory does not exist: {p}")
        if not p.is_dir():
            raise NotADirectoryError(f"path is not a directory: {p}")
        return p

    @staticmethod
    def _discover(directory: Path) -> list[str]:
        """Non-recursive, case-insensitive ``.txt`` discovery, deterministically ordered."""
        names = [
            e.name
            for e in directory.iterdir()
            if e.is_file() and e.suffix.lower() == ".txt"
        ]
        # primary key: case-insensitive; tie-break: exact name. Never iterdir order.
        names.sort(key=lambda n: (n.casefold(), n))
        return names

    @staticmethod
    def _doc_id(prefix: str, filename: str) -> str:
        return f"{prefix}/{filename}" if prefix else filename
