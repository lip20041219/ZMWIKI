"""Evaluation dataset loader: local JSON file -> EvaluationCaseSet.

Phase 9C closes the input side of offline retrieval evaluation: an existing,
JSON-compatible evaluation-case file is loaded into an :class:`EvaluationCaseSet`
so the Phase 9B case contract can feed the Phase 9A runner unchanged.

```
JSON file
   ↓  JsonEvaluationDatasetLoader.load(path)
EvaluationCaseSet
   ↓  RetrievalEvaluationRunner -> EvaluationHarness -> RetrievalEvaluator
EvaluationResult
```

The loader is a thin, pure, offline adapter. It reads a local file (UTF-8,
no caching -- every ``load`` re-reads), parses one strict JSON schema, and
delegates to the value layer exactly as follows: raw ``query`` /
``relevant_chunk_ids`` fields are pulled out of the JSON, handed to
:class:`EvaluationCase` (which owns query non-emptiness and the canonical
sorted relevant-id tuple), and the constructed cases are collected into an
immutable :class:`EvaluationCaseSet`. The loader implements **no** second copy
of EvaluationCase's business validation and **no** second ordering rule; it
only asserts that the file is well-formed under the documented schema.

The loader is deliberately not a benchmark framework: no BEIR / MS MARCO /
HuggingFace datasets, no remote download, no registry / versioning / cache,
no database, no CLI. It depends only on the evaluation value layer and the
stdlib (``json`` / ``pathlib`` / ``abc``); it never imports or drives the
retrieval pipeline.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from pathlib import Path

from .evaluation import EvaluationCase
from .evaluation_dataset import EvaluationCaseSet

__all__ = ["EvaluationDatasetLoader", "JsonEvaluationDatasetLoader"]

#: The only keys the JSON top level and each case object may carry.
_TOP_KEYS = ("cases",)
_CASE_KEYS = ("query", "relevant_chunk_ids")


class EvaluationDatasetLoader(ABC):
    """Loads an EvaluationCaseSet from an evaluation-case data file."""

    @abstractmethod
    def load(self, path: str | Path) -> EvaluationCaseSet:
        """Load the ``path`` file into an immutable EvaluationCaseSet.

        ``path`` must be a str or pathlib.Path naming an existing regular
        UTF-8 JSON file. Missing file -> ``FileNotFoundError``, a directory ->
        ``IsADirectoryError``, a wrong ``path`` type -> ``TypeError``, invalid
        UTF-8 -> ``UnicodeDecodeError``, malformed JSON -> ``json.JSONDecodeError``,
        and schema violations -> ``ValueError``. ``EvaluationCase`` business
        constraints propagate untouched from the case constructor.
        """


class JsonEvaluationDatasetLoader(EvaluationDatasetLoader):
    """Loads the minimal local JSON case format into an EvaluationCaseSet.

    Stateless and constructor-free: nothing is cached, configured, or kept
    between calls, so two ``load`` calls (same loader or different loaders)
    over the same file content always return equal, immutable CaseSets.
    """

    def load(self, path: str | Path) -> EvaluationCaseSet:
        target = _require_path(path)
        if target.is_dir():
            raise IsADirectoryError(
                f"path is a directory, not an evaluation dataset file: {str(target)!r}"
            )
        # Explicit UTF-8 read: a non-UTF-8 file raises UnicodeDecodeError.
        raw = target.read_text(encoding="utf-8")
        data = json.loads(raw)  # malformed JSON -> json.JSONDecodeError (never swallowed)
        return self._parse(data)

    def _parse(self, data: object) -> EvaluationCaseSet:
        _check_object(data, "top level")
        _check_only_keys(data, _TOP_KEYS, "top level")
        cases_obj = data["cases"]
        if not isinstance(cases_obj, list):
            raise ValueError(
                "'cases' must be a JSON array, "
                f"got {_typename(cases_obj)}"
            )

        parsed: list[EvaluationCase] = []
        for i, item in enumerate(cases_obj):
            _check_object(item, f"cases[{i}]")
            _check_only_keys(item, _CASE_KEYS, f"cases[{i}]")
            query = item["query"]
            relevant = item["relevant_chunk_ids"]
            if not isinstance(query, str):
                raise ValueError(
                    f"cases[{i}].query must be a string, got {_typename(query)}"
                )
            if not isinstance(relevant, list):
                raise ValueError(
                    f"cases[{i}].relevant_chunk_ids must be a JSON array, "
                    f"got {_typename(relevant)}"
                )
            for j, cid in enumerate(relevant):
                if not isinstance(cid, str):
                    raise ValueError(
                        f"cases[{i}].relevant_chunk_ids[{j}] must be a string, "
                        f"got {_typename(cid)}"
                    )
            # EvaluationCase owns the business rules (empty/whitespace query,
            # non-empty ids, no empty/duplicate id, canonical sort) unchanged.
            parsed.append(EvaluationCase(query=query, relevant_chunk_ids=relevant))
        return EvaluationCaseSet(cases=tuple(parsed))


def _require_path(path: object) -> Path:
    if isinstance(path, bool) or not isinstance(path, (str, Path)):
        raise TypeError(
            f"path must be a str or pathlib.Path, got {_typename(path)}"
        )
    return Path(path)


def _check_object(value: object, where: str) -> None:
    if not isinstance(value, dict):
        raise ValueError(f"{where} must be a JSON object, got {_typename(value)}")


def _check_only_keys(mapping: dict[str, object], allowed: tuple[str, ...], where: str) -> None:
    unknown = [key for key in mapping if key not in allowed]
    if unknown:
        raise ValueError(
            f"{where} has unsupported field(s): {', '.join(map(str, unknown))}"
        )
    for key in allowed:
        if key not in mapping:
            raise ValueError(f"{where} is missing required field {key!r}")


def _typename(value: object) -> str:
    """Human name of a parsed-JSON value's Python type (None -> 'null')."""
    return "null" if value is None else type(value).__name__
