"""Embedding contract unit tests (Embedding / EmbeddingProvider / DeterministicEmbedder)."""

import math
import subprocess
import sys

import pytest

from zmrag import (
    DeterministicEmbedder,
    Document,
    Embedding,
    EmbeddingProvider,
)


# --- Embedding schema ---


def test_embedding_creation_and_dimension():
    emb = Embedding(vector=(0.1, -0.2, 0.3))
    assert emb.vector == (0.1, -0.2, 0.3)
    assert emb.dimension == 3


def test_embedding_accepts_list_and_coerces_ints():
    emb = Embedding(vector=[1, 2, 3])
    assert emb.vector == (1.0, 2.0, 3.0)
    assert isinstance(emb.vector[0], float)
    assert emb.dimension == 3


def test_embedding_empty_vector_rejected():
    with pytest.raises(ValueError):
        Embedding(vector=())


@pytest.mark.parametrize("bad", [("a", 1.0), (None,), (True, 1.0), ("0.5",)])
def test_embedding_non_numeric_entry_rejected(bad):
    with pytest.raises(TypeError):
        Embedding(vector=bad)


@pytest.mark.parametrize("bad", [(float("nan"), 1.0), (float("inf"),), (1.0, float("-inf"))])
def test_embedding_non_finite_rejected(bad):
    with pytest.raises(ValueError):
        Embedding(vector=bad)


def test_embedding_non_sequence_rejected():
    with pytest.raises(TypeError):
        Embedding(vector=3.14)


def test_embedding_frozen_and_equal():
    a = Embedding(vector=(0.1, 0.2))
    b = Embedding(vector=(0.1, 0.2))
    assert a == b
    with pytest.raises(Exception):
        a.vector = (0.0,)  # type: ignore[misc]


# --- Provider abstraction ---


def test_provider_is_abstract():
    with pytest.raises(TypeError):
        EmbeddingProvider()  # type: ignore[abstract]


def test_single_embed_contract():
    emb = DeterministicEmbedder(dimension=4).embed("hello")
    assert isinstance(emb, Embedding)
    assert emb.dimension == 4


def test_batch_matches_individual():
    provider = DeterministicEmbedder(dimension=5)
    texts = ["alpha", "beta", "gamma"]
    batch = provider.embed_many(texts)
    indiv = [provider.embed(t) for t in texts]
    assert batch == indiv
    assert len(batch) == 3
    assert all(e.dimension == 5 for e in batch)


def test_empty_batch_returns_empty():
    assert DeterministicEmbedder().embed_many([]) == []


def test_batch_accepts_tuple():
    emb = DeterministicEmbedder(dimension=3).embed_many(("a", "b"))
    assert len(emb) == 2


def test_batch_rejects_single_string():
    with pytest.raises(TypeError):
        DeterministicEmbedder().embed_many("not a batch")


def test_batch_propagates_element_validation():
    with pytest.raises(TypeError):
        DeterministicEmbedder().embed_many(["ok", "   "])


# --- DeterministicEmbedder behaviour ---


def test_same_text_same_vector_repeated_calls():
    provider = DeterministicEmbedder(dimension=6)
    assert provider.embed("hello") == provider.embed("hello")


def test_different_texts_differ_in_practice():
    provider = DeterministicEmbedder(dimension=8)
    a = provider.embed("alpha")
    b = provider.embed("beta")
    assert a != b


def test_separate_instances_deterministic():
    a = DeterministicEmbedder(dimension=6).embed("hello")
    b = DeterministicEmbedder(dimension=6).embed("hello")
    assert a == b


def test_values_are_finite_and_bounded():
    provider = DeterministicEmbedder(dimension=16)
    for v in provider.embed("sample").vector:
        assert math.isfinite(v)
        assert -1.0 <= v < 1.0


def test_cross_process_stable():
    """Same text embeds identically in a fresh interpreter (no hash() salt)."""
    provider = DeterministicEmbedder(dimension=6)
    expected = provider.embed("hello world").vector
    code = (
        "import zmrag; "
        "print(zmrag.DeterministicEmbedder(dimension=6).embed('hello world').vector)"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    actual = eval(out.stdout)  # noqa: S307 - deterministic repr from our own code
    assert tuple(actual) == expected


# --- Validation ---


@pytest.mark.parametrize("bad", ["", "   \n\t"])
def test_empty_text_rejected(bad):
    with pytest.raises(TypeError):
        DeterministicEmbedder().embed(bad)


def test_non_string_text_rejected():
    with pytest.raises(TypeError):
        DeterministicEmbedder().embed(42)  # type: ignore[arg-type]


@pytest.mark.parametrize("bad", [0, -1])
def test_invalid_dimension_value_rejected(bad):
    with pytest.raises(ValueError):
        DeterministicEmbedder(dimension=bad)


@pytest.mark.parametrize("bad", [2.5, "8", True, None])
def test_invalid_dimension_type_rejected(bad):
    with pytest.raises(TypeError):
        DeterministicEmbedder(dimension=bad)  # type: ignore[arg-type]


def test_public_import():
    from zmrag import __all__

    assert {"Embedding", "EmbeddingProvider", "DeterministicEmbedder"} <= set(__all__)


def test_document_and_chunker_still_compose():
    # provider only needs text: existing pipeline objects are unaffected
    doc = Document(id="d", content="hello")
    emb = DeterministicEmbedder(dimension=4).embed(doc.content)
    assert emb.dimension == 4
