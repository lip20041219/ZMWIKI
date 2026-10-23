"""Optional semantic embeddings behind the existing EmbeddingProvider contract.

zmwiki core ships only :class:`~zmwiki.embedders.DeterministicEmbedder` -- a
test / local-development stand-in that produces *no* semantic vectors. This
module adds a real, optional semantic provider for Phase 11's Semantic
Retrieval Profile. It is deliberately **not** part of the top-level public API
and is never imported by the default import graph.

Import safety: importing this module is always safe. No ML dependency appears
at module scope; ``torch`` / ``sentence-transformers`` are imported only inside
``_create_backend`` at instantiation time, so a plain ``import zmwiki`` (or
``from zmwiki.semantic import SemanticEmbeddingProvider``) never touches them.

Deterministic profile vs semantic profile (Phase 11 boundary):

- Deterministic profile (``DeterministicEmbedder``): stdlib-only,
  network-free, byte-reproducible across calls / instances / processes. It
  stays the canonical test/CI profile and its evidence is never regenerated.
- Semantic profile (this module): requires the ``semantic`` optional
  dependency (sentence-transformers + torch) and model weights. It is
  reproducible within one pinned model + dependency set, but is *not*
  byte-reproducible across model / torch / dependency versions and must never
  be labelled or merged as deterministic evidence.

Backend seam: :func:`_create_backend` lazily builds a narrow adapter with
``dimension`` and ``encode(list[str]) -> list[tuple[float, ...]]``. Tests
inject a fake backend by monkeypatching ``_create_backend``, so the provider
contract is exercised without any ML dependency, model, or network.
"""

from __future__ import annotations

import os
from typing import Sequence

from .embedders import Embedding, EmbeddingProvider
from .schema import _nonempty_str

__all__ = ["SemanticEmbeddingProvider"]


class SemanticEmbeddingProvider(EmbeddingProvider):
    """Real semantic embeddings behind the existing ``EmbeddingProvider`` contract.

    Backend: sentence-transformers (the ``semantic`` optional dependency),
    loaded lazily. Constructing a provider with the dependency absent raises an
    actionable ``ImportError``; merely importing this module never does.

    Exactly one model source is required (keyword-only):

    - ``model_name``: a sentence-transformers model id (weights are downloaded
      on first use unless already cached).
    - ``local_model_path``: a directory holding a locally stored
      sentence-transformers model, for offline use.

    ``device`` is passed through to the backend (``None`` = backend default).

    Every result is wrapped in the library's :class:`Embedding` value object
    (a finite ``tuple[float, ...]``); numpy arrays / torch tensors never leak
    into zmwiki. ``dimension`` always matches the loaded model's embedding
    dimension.
    """

    def __init__(
        self,
        *,
        model_name: str | None = None,
        local_model_path: str | None = None,
        device: str | None = None,
    ) -> None:
        if (model_name is None) == (local_model_path is None):
            raise ValueError(
                "exactly one of 'model_name' or 'local_model_path' is required"
            )
        if model_name is not None:
            _nonempty_str(model_name, "model_name")
            model_source: str = model_name
        else:
            model_source = os.fspath(local_model_path)
            _nonempty_str(model_source, "local_model_path")
        self._backend = _create_backend(model_source=model_source, device=device)
        self._dimension = self._backend.dimension

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed(self, text: str) -> Embedding:
        _nonempty_str(text, "text")
        return self._as_embedding(self._backend.encode([text])[0])

    def embed_many(self, texts: Sequence[str]) -> list[Embedding]:
        # Real backends batch cheaply, so override the ABC's per-item default
        # with a single backend call. Contract preserved: list/tuple input,
        # empty batch -> [], order and count preserved, every item an Embedding
        # with the provider's one dimension.
        if isinstance(texts, (str, bytes)) or not isinstance(texts, (list, tuple)):
            raise TypeError(
                f"texts must be a list/tuple of str, got {type(texts).__name__}"
            )
        if not texts:
            return []
        return [self._as_embedding(v) for v in self._backend.encode(list(texts))]

    @staticmethod
    def _as_embedding(vector) -> Embedding:
        # tolist() covers numpy.ndarray and torch.Tensor without importing them.
        if hasattr(vector, "tolist"):
            vector = vector.tolist()
        return Embedding(tuple(float(x) for x in vector))


def _create_backend(*, model_source: str, device: str | None) -> "_SentenceTransformerBackend":
    """Lazily build the sentence-transformers backend adapter.

    The heavy import happens *here*, never at module scope. With the optional
    dependency absent this raises a clear, actionable ImportError.
    """
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:  # optional dependency missing
        raise ImportError(
            "SemanticEmbeddingProvider requires the 'semantic' optional "
            'dependency (sentence-transformers + torch). Install it with: '
            'pip install "zmwiki[semantic]"'
        ) from exc
    return _SentenceTransformerBackend(
        SentenceTransformer(model_source, device=device)
    )


class _SentenceTransformerBackend:
    """Narrow adapter: ``dimension`` + ``encode(list[str]) -> list[tuple[float, ...]]``.

    The only place sentence-transformers' return types are touched. ``encode``
    emits unit vectors (``normalize_embeddings=True``), so the dense store's
    cosine similarity is a cosine in ``[-1, 1]``.
    """

    def __init__(self, model) -> None:
        self._model = model
        dim = model.get_sentence_embedding_dimension()
        if dim is None:  # some local models do not declare a dimension
            dim = int(model.encode([" "]).shape[1])
        self._dimension = int(dim)

    @property
    def dimension(self) -> int:
        return self._dimension

    def encode(self, texts: list[str]) -> list[tuple[float, ...]]:
        vectors = self._model.encode(texts, normalize_embeddings=True)
        return [tuple(float(x) for x in row) for row in vectors]
