"""ZMWiki page data contract.

WikiPage is the structured knowledge unit compiled from raw Documents.
It is a pure, immutable value object -- no indexing, compilation, LLM,
or agent logic lives here.

Design choices mirror ``zmwiki.schema``: frozen dataclass, ``__post_init__``
validation with ``TypeError``, shallow-copied metadata, and caller-supplied
stable identifiers (never randomly regenerated).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from zmwiki.schema import Metadata, _metadata_copy, _nonempty_str

__all__ = ["WikiPage"]


def _as_tuple_of_str(value: Any, name: str) -> tuple[str, ...]:
    """Validate and normalize a sequence of non-empty strings to a tuple.

    Accepts any iterable of str (list, tuple, generator); rejects a bare
    str (which is iterable but semantically wrong here) and non-iterables.
    Each element must pass ``_nonempty_str``.
    """
    if isinstance(value, str):
        raise TypeError(f"{name} must be a sequence of str, not a single str")
    try:
        items = tuple(value)
    except TypeError:
        raise TypeError(
            f"{name} must be an iterable, got {type(value).__name__}"
        )
    for item in items:
        _nonempty_str(item, f"{name} element")
    return items


def _valid_page_id(value: Any) -> None:
    """Validate a wiki page identifier.

    A page id must be a non-empty str with no newline characters. Newlines
    are rejected because ids are single-line stable identifiers used in
    indexes, links, and file paths.
    """
    _nonempty_str(value, "wiki page id")
    if "\n" in value or "\r" in value:
        raise TypeError(
            f"wiki page id must not contain newline characters, got {value!r}"
        )


@dataclass(frozen=True)
class WikiPage:
    """An immutable, structured Wiki knowledge page.

    Fields
    ------
    id: stable, non-empty page identifier. Caller-supplied, never randomly
        regenerated on load. Must not contain newline characters. Required.
    path: logical page path (e.g. "databases/btree/overview"). Required,
        non-empty.
    title: human-readable page title. Required, non-empty.
    aliases: optional alternative names for this page. Tuple of non-empty
        str. Default empty tuple.
    tags: optional categorical tags. Tuple of non-empty str. Default empty.
    description: optional short summary. str or None. If provided, must be
        non-empty. Default None.
    content: page body text. Required, non-empty.
    sources: ``Document.id`` values this page was compiled from. Tuple of
        non-empty str. Empty tuple means no traceable source. Default empty.
    links_to: page ids this page links out to. Tuple of non-empty str.
        Default empty.
    links_from: page ids that link into this page. Tuple of non-empty str.
        Default empty.
    metadata: free-form extension dict. Copied on construction so caller
        mutation never leaks in. Default empty dict.

    Frozen by design (same rationale as Document/Chunk); metadata is
    shallowly immutable only.
    """

    id: str
    path: str
    title: str
    content: str
    aliases: tuple[str, ...] = field(default_factory=tuple)
    tags: tuple[str, ...] = field(default_factory=tuple)
    description: str | None = None
    sources: tuple[str, ...] = field(default_factory=tuple)
    links_to: tuple[str, ...] = field(default_factory=tuple)
    links_from: tuple[str, ...] = field(default_factory=tuple)
    metadata: Metadata = field(default_factory=dict)

    def __post_init__(self) -> None:
        _valid_page_id(self.id)
        _nonempty_str(self.path, "wiki page path")
        _nonempty_str(self.title, "wiki page title")
        _nonempty_str(self.content, "wiki page content")
        if self.description is not None:
            _nonempty_str(self.description, "wiki page description")
        # Normalize sequences to tuples (accept list/generator, reject bare str).
        object.__setattr__(self, "aliases", _as_tuple_of_str(self.aliases, "aliases"))
        object.__setattr__(self, "tags", _as_tuple_of_str(self.tags, "tags"))
        object.__setattr__(self, "sources", _as_tuple_of_str(self.sources, "sources"))
        object.__setattr__(
            self, "links_to", _as_tuple_of_str(self.links_to, "links_to")
        )
        object.__setattr__(
            self, "links_from", _as_tuple_of_str(self.links_from, "links_from")
        )
        object.__setattr__(self, "metadata", _metadata_copy(self.metadata))

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible dict.

        Tuples are converted to lists (JSON has no tuple type); metadata is
        shallow-copied. Round-trips with ``from_dict``.
        """
        return {
            "id": self.id,
            "path": self.path,
            "title": self.title,
            "content": self.content,
            "aliases": list(self.aliases),
            "tags": list(self.tags),
            "description": self.description,
            "sources": list(self.sources),
            "links_to": list(self.links_to),
            "links_from": list(self.links_from),
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> WikiPage:
        """Deserialize from a dict produced by ``to_dict``.

        Raises ``TypeError`` if ``data`` is not a dict or if any required
        field (id, path, title, content) is missing. Optional fields default
        to their empty/None values when absent. All validation from
        ``__post_init__`` applies.
        """
        if not isinstance(data, dict):
            raise TypeError(
                f"from_dict expects a dict, got {type(data).__name__}"
            )
        required = ("id", "path", "title", "content")
        for key in required:
            if key not in data:
                raise TypeError(f"missing required field: {key!r}")
        return cls(
            id=data["id"],
            path=data["path"],
            title=data["title"],
            content=data["content"],
            aliases=data.get("aliases", ()),
            tags=data.get("tags", ()),
            description=data.get("description"),
            sources=data.get("sources", ()),
            links_to=data.get("links_to", ()),
            links_from=data.get("links_from", ()),
            metadata=data.get("metadata", {}),
        )
