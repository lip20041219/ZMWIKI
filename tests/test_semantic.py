"""Phase 11 Step 2: semantic provider contract + import safety.

This module pins what Phase 11 Step 2 adds without touching production code
beyond the new ``semantic.py`` and without any ML dependency, model, or network:

- importing ``zmrag`` must never load torch / transformers /
  sentence_transformers / fastembed (checked in a *fresh interpreter*, so an
  already-loaded dependency in the test process cannot cause a false failure);
- ``zmrag.semantic`` must import cleanly with the optional dependency absent;
- constructing ``SemanticEmbeddingProvider`` without the optional dependency
  must raise an actionable ImportError (skipped when the dependency *is*
  installed, so a dev environment never triggers a model download);
- the provider -> Embedding -> DenseRetriever -> InMemoryVectorStore seam works
  with a lightweight injected fake backend (monkeypatched ``_create_backend``);
- the deterministic profile is untouched: ``DeterministicEmbedder`` output is
  byte/value-stable, which every deterministic chain and its frozen evidence
  rests on.

The three frozen Phase 10 evidence files are never written by anything here;
no benchmark runner is invoked.
"""

import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import pytest

from zmrag import (
    Chunk,
    DenseRetriever,
    DeterministicEmbedder,
    Embedding,
    InMemoryVectorStore,
)
from zmrag.embedders import EmbeddingProvider
from zmrag.semantic import SemanticEmbeddingProvider

#: The ML modules whose auto-load after ``import zmrag`` would be a regression.
ML_MODULES = ("torch", "transformers", "sentence_transformers", "fastembed")

#: Literal pin of DeterministicEmbedder output (cross-process stable by design).
_PINNED_VECTOR = (
    0.6856080230568162,
    0.035514888438890946,
    -0.9642660458968345,
    -0.5073776120824016,
    -0.411777222489777,
    0.5403885225592844,
    0.6630629114272919,
    -0.23797857572408354,
)


def _requires_semantic_dependency() -> bool:
    """True when the optional ML dependency is importable in this environment."""
    return any(
        importlib.util.find_spec(name) is not None
        for name in ("sentence_transformers", "torch")
    )


class _FakeBackend:
    """Narrow backend seam stand-in: dimension + encode(list[str]).

    Deterministic and ML-free; reuses ``DeterministicEmbedder`` so identical
    text always yields an identical vector (exact-match queries win the scan).
    """

    def __init__(self, dimension: int = 8) -> None:
        self.dimension = dimension

    def encode(self, texts: list[str]) -> list[tuple[float, ...]]:
        embedder = DeterministicEmbedder(dimension=self.dimension)
        return [embedder.embed(text).vector for text in texts]


@pytest.fixture
def provider(monkeypatch):
    import zmrag.semantic as semantic

    monkeypatch.setattr(semantic, "_create_backend", lambda **kwargs: _FakeBackend())
    return SemanticEmbeddingProvider(model_name="unused/fake-model")


# --- A/B. import zmrag must not auto-load ML dependencies ----------------------


def test_import_zmrag_does_not_load_ml_dependencies():
    """A fresh interpreter imports zmrag; no ML module may appear in sys.modules.

    Run in a subprocess so the state of the current pytest process (which may
    already hold some dependency) cannot cause a false failure.
    """
    root = Path(__file__).resolve().parents[1]
    code = (
        "import sys\n"
        f"sys.path.insert(0, {str(root / 'src')!r})\n"
        "import zmrag\n"
        "loaded = sorted(m for m in "
        "('torch', 'transformers', 'sentence_transformers', 'fastembed') "
        "if m in sys.modules)\n"
        "print('LOADED:' + ','.join(loaded))\n"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code],
        cwd=root,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr
    last = [ln for ln in proc.stdout.splitlines() if ln.startswith("LOADED:")]
    assert last and last[-1] == "LOADED:", last


# --- C. zmrag.semantic imports cleanly without the optional dependency ---------


def test_semantic_module_imports_without_optional_dependency():
    import zmrag.semantic as semantic  # noqa: F401  (must not raise)

    assert isinstance(SemanticEmbeddingProvider, type)
    assert issubclass(SemanticEmbeddingProvider, EmbeddingProvider)


# --- D. instantiation without the dependency raises an actionable ImportError --


def test_instantiation_without_dependency_raises_actionable_importerror():
    if _requires_semantic_dependency():
        pytest.skip(
            "semantic optional dependency installed; "
            "ImportError path is not exercised in this environment"
        )
    with pytest.raises(ImportError) as excinfo:
        SemanticEmbeddingProvider(model_name="BAAI/bge-small-en-v1.5")
    message = str(excinfo.value)
    assert "zmrag[semantic]" in message
    assert "sentence-transformers" in message


def test_requires_exactly_one_model_source():
    # Validation happens before any backend is built, so this needs no dependency.
    with pytest.raises(ValueError):
        SemanticEmbeddingProvider()
    with pytest.raises(ValueError):
        SemanticEmbeddingProvider(model_name="m", local_model_path=".")


# --- E. provider -> Embedding -> DenseRetriever -> store seam (fake backend) ---


def test_provider_contract_seam_through_dense_retriever(provider):
    assert provider.dimension == 8

    texts = ["alpha beta gamma", "delta epsilon zeta"]
    chunks = [
        Chunk(id=f"c{i}", document_id="d", text=text, position=i)
        for i, text in enumerate(texts)
    ]

    embs = provider.embed_many([c.text for c in chunks])
    assert len(embs) == 2
    assert all(isinstance(e, Embedding) for e in embs)
    assert all(e.dimension == provider.dimension for e in embs)
    # Batch override preserves order/count and equals the per-item contract.
    assert [e.vector for e in embs] == [
        provider.embed(text).vector for text in texts
    ]

    store = InMemoryVectorStore()
    for chunk, embedding in zip(chunks, embs):
        store.upsert(chunk, embedding)
    assert store.count() == 2

    retriever = DenseRetriever(provider, store)
    hits = retriever.retrieve(texts[0], top_k=2)
    assert [r.chunk.id for r in hits] == ["c0", "c1"]


def test_provider_embed_many_contract_edges(provider):
    assert provider.embed_many([]) == []
    with pytest.raises(TypeError):
        provider.embed_many("not a list")
    with pytest.raises(TypeError):
        provider.embed_many(["ok", 1])
    with pytest.raises(TypeError):
        provider.embed("")  # noqa: PLC1901 (empty query is malformed)


# --- F. Deterministic profile protection --------------------------------------


def test_deterministic_embedder_output_is_byte_stable_after_semantic_import():
    # This module already imported zmrag.semantic; the deterministic profile
    # must be untouched: DeterministicEmbedder output stays exactly pinned.
    assert DeterministicEmbedder().embed("alpha beta gamma").vector == _PINNED_VECTOR
