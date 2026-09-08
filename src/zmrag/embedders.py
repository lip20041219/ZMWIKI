"""Embedding contract: Text -> EmbeddingProvider -> Embedding.

ZMRAG core is embedding-provider-agnostic: no concrete API/model provider is
shipped. The only implementation here, ``DeterministicEmbedder``, is a
**test-only / local-development** stand-in -- it produces no semantic vectors
and must never be used to benchmark retrieval quality.
"""

from __future__ import annotations

import hashlib
import math
import struct
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Sequence

from .schema import _nonempty_str

__all__ = ["Embedding", "EmbeddingProvider", "DeterministicEmbedder"]


@dataclass(frozen=True)
class Embedding:
    """A fixed-dimension numeric vector.

    Deliberately independent of any provider library's return type, so a
    future VectorStore depends only on this value object.
    """

    vector: tuple[float, ...]

    @property
    def dimension(self) -> int:
        return len(self.vector)

    def __post_init__(self) -> None:
        if not isinstance(self.vector, (list, tuple)):
            raise TypeError(f"vector must be a sequence of floats, got {type(self.vector).__name__}")
        if not self.vector:
            raise ValueError("vector must not be empty")
        out: list[float] = []
        for value in self.vector:
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"vector entries must be int/float, got {value!r}")
            value = float(value)
            if not math.isfinite(value):
                raise ValueError(f"vector entries must be finite, got {value!r}")
            out.append(value)
        object.__setattr__(self, "vector", tuple(out))


class EmbeddingProvider(ABC):
    """Maps text to fixed-dimension Embeddings.

    Contract: same text -> same embedding (for a given provider+config);
    ``embed_many`` matches ``[embed(t) for t in texts]`` in content and order.
    """

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Dimensionality every returned embedding shares."""

    @abstractmethod
    def embed(self, text: str) -> Embedding:
        """Embed one non-empty text."""

    def embed_many(self, texts: Sequence[str]) -> list[Embedding]:
        """Embed an ordered sequence of texts.

        Order and count are preserved; an empty batch returns []. Subclasses
        may override to batch at a real provider, but must keep this contract.
        """
        if isinstance(texts, (str, bytes)) or not isinstance(texts, (list, tuple)):
            raise TypeError(
                f"texts must be a list/tuple of str, got {type(texts).__name__}"
            )
        return [self.embed(text) for text in texts]


class DeterministicEmbedder(EmbeddingProvider):
    """Test-only / local-development embedder. NOT a semantic model.

    Deterministic principle: the vector is derived purely from the text via
    sha256 + struct (stdlib). No randomness (``random``/``hash()``), so the
    result is byte-for-byte stable across calls, instances, and processes.
    Different texts yield different vectors in practice; same text + dimension
    always yields the exact same vector.
    """

    def __init__(self, dimension: int = 8) -> None:
        if isinstance(dimension, bool) or not isinstance(dimension, int):
            raise TypeError(f"dimension must be an int, got {type(dimension).__name__}")
        if dimension <= 0:
            raise ValueError(f"dimension must be > 0, got {dimension}")
        self._dimension = dimension

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed(self, text: str) -> Embedding:
        _nonempty_str(text, "text")
        return Embedding(_deterministic_vector(text, self._dimension))


def _deterministic_vector(text: str, dimension: int) -> tuple[float, ...]:
    """Map ``text`` to ``dimension`` floats in [-1, 1), deterministically.

    Each coordinate comes from 64 bits of sha256(text + counter), so inputs
    differing anywhere produce essentially always-different vectors. Stable
    across Python processes (no ``hash()`` salt, no global RNG).
    """
    out: list[float] = []
    counter = 0
    while len(out) < dimension:
        digest = hashlib.sha256(f"{counter}\n{text}".encode("utf-8")).digest()
        counter += 1
        for i in range(4):  # 4 x uint64 per 32-byte digest
            (word,) = struct.unpack_from("<Q", digest, i * 8)
            out.append(word / 2**63 - 1.0)
    return tuple(out[:dimension])
