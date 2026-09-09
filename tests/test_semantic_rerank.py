"""Phase 12 Step 2: CrossEncoderReranker contract + import safety.

Pins what Phase 12 Step 2 adds without touching any production code beyond the
new ``semantic_rerank.py`` and without any ML dependency, model, or network:

- importing ``zmrag`` and ``zmrag.semantic_rerank`` must never load torch /
  transformers / sentence_transformers (fresh-interpreter check, so an
  already-loaded dependency in the test process cannot cause a false failure);
- constructing ``CrossEncoderReranker`` validates exactly-one model source
  *before* any backend is built;
- the reranker behaves like the existing ``Reranker`` contract: subset-only
  output (never re-recalls, never queries a store/retriever/index), empty
  ``[] -> []``, ``top_k`` int >= 1 (bool rejected), duplicate ``chunk.id`` and
  non-``SearchResult`` candidates rejected, inputs never mutated, one batched
  ``score_pairs(query, texts)`` backend call, deterministic ``(-score, chunk.id)``
  tie-break;
- Phase 12's new score validation: wrong score count, non-finite (NaN/Inf), and
  non-numeric scores are rejected (never silently truncated/padded/coerced);
- the reranker has no ``EmbeddingProvider`` inheritance / coupling -- it is a
  pure re-ranker over candidate ``SearchResult``s.

All of this is exercised through a fake backend injected by monkeypatching
``_create_backend`` -- no model is constructed, downloaded, or run.
"""

import subprocess
import sys
from pathlib import Path

import pytest

from zmrag.embedders import EmbeddingProvider
from zmrag.rerankers import Reranker
from zmrag.schema import Chunk

import zmrag.semantic_rerank as semantic_rerank
from zmrag.semantic_rerank import CrossEncoderReranker
from zmrag.stores import SearchResult


def _chunk(cid: str, text: str = "some text", pos: int = 0) -> Chunk:
    return Chunk(id=cid, document_id="doc", text=text, position=pos)


def _candidate(cid: str, text: str = "some text", pos: int = 0) -> SearchResult:
    return SearchResult(chunk=_chunk(cid, text, pos), score=0.0)


class _FakeScorer:
    """Deterministic fake for the ``score_pairs(query, texts) -> list[float]`` seam.

    Records every call so tests can assert the batching contract (one call with
    every candidate text, query preserved) and returns the caller-set scores.
    """

    def __init__(self, scores: list[float] | None = None) -> None:
        self.scores = list(scores) if scores is not None else []
        self.calls: list[tuple[str, tuple[str, ...]]] = []

    def score_pairs(self, query: str, texts: list[str]) -> list[float]:
        self.calls.append((query, tuple(texts)))
        return list(self.scores)


@pytest.fixture
def reranker(monkeypatch):
    """A CrossEncoderReranker whose backend is a _FakeScorer (no real model)."""
    fake = _FakeScorer()
    monkeypatch.setattr(
        semantic_rerank, "_create_backend", lambda **kwargs: fake
    )
    return CrossEncoderReranker(model_name="unused/fake-model"), fake


# --- A. import safety ---------------------------------------------------------


def test_import_zmrag_and_semantic_rerank_do_not_load_ml_dependencies():
    """A fresh interpreter imports zmrag + zmrag.semantic_rerank; no ML module
    may appear in sys.modules."""
    root = Path(__file__).resolve().parents[1]
    code = (
        "import sys\n"
        f"sys.path.insert(0, {str(root / 'src')!r})\n"
        "import zmrag\n"
        "import zmrag.semantic_rerank\n"
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


# --- B. constructor validation -------------------------------------------------


def test_constructor_requires_exactly_one_model_source(monkeypatch):
    """Validation happens before any backend is built, so invalid arguments
    must raise before _create_backend ever runs."""
    def _explode(**kwargs):  # pragma: no cover - must never be reached
        raise AssertionError("_create_backend called before argument validation")

    monkeypatch.setattr(semantic_rerank, "_create_backend", _explode)
    with pytest.raises(ValueError):
        CrossEncoderReranker()
    with pytest.raises(ValueError):
        CrossEncoderReranker(model_name="m", local_model_path=".")


def test_constructor_blank_model_name_rejected(monkeypatch):
    def _explode(**kwargs):  # pragma: no cover
        raise AssertionError("_create_backend called for invalid model_name")

    monkeypatch.setattr(semantic_rerank, "_create_backend", _explode)
    with pytest.raises(TypeError):
        CrossEncoderReranker(model_name="   ")


# --- C. subset-only ------------------------------------------------------------

def test_subset_only_top2(reranker):
    reranker_cls, fake = reranker
    fake.scores = [0.1, 0.9, 0.5]
    cands = [_candidate("a"), _candidate("b"), _candidate("c")]
    out = reranker_cls.rerank("q", cands, top_k=2)
    assert [r.chunk.id for r in out] == ["b", "c"]
    # Every output chunk is one of the input candidates -- never a new recall.
    assert {r.chunk.id for r in out} <= {c.chunk.id for c in cands}


# --- D. top_k -------------------------------------------------------------------

def test_top_k_variants(reranker):
    reranker_cls, fake = reranker
    cands = [_candidate("a"), _candidate("b"), _candidate("c")]
    fake.scores = [0.3, 0.1, 0.2]
    # top_k > len(candidates) -> everything (best first).
    out = reranker_cls.rerank("q", cands, top_k=10)
    assert [r.chunk.id for r in out] == ["a", "c", "b"]
    # top_k == len -> everything.
    assert len(reranker_cls.rerank("q", cands, top_k=3)) == 3
    # top_k == 1 -> single best.
    assert [r.chunk.id for r in reranker_cls.rerank("q", cands, top_k=1)] == ["a"]


def test_top_k_invalid(reranker):
    reranker_cls, fake = reranker
    cands = [_candidate("a")]
    fake.scores = [0.5]
    for bad in (0, -1):
        with pytest.raises(ValueError):
            reranker_cls.rerank("q", cands, top_k=bad)
    with pytest.raises(TypeError):
        reranker_cls.rerank("q", cands, top_k=True)
    with pytest.raises(TypeError):
        reranker_cls.rerank("q", cands, top_k=2.5)


# --- E. duplicate chunk.id -------------------------------------------------------

def test_duplicate_chunk_id_rejected(reranker):
    reranker_cls, fake = reranker
    fake.scores = [0.5, 0.5]
    dup = [_candidate("a"), SearchResult(chunk=_chunk("a", pos=1), score=0.0)]
    with pytest.raises(ValueError):
        reranker_cls.rerank("q", dup, top_k=8)


# --- F. empty candidates ---------------------------------------------------------

def test_empty_candidates_returns_empty(reranker):
    reranker_cls, fake = reranker
    assert reranker_cls.rerank("q", [], top_k=8) == []
    assert fake.calls == []  # backend never invoked for an empty pool


# --- G. order / tie-break --------------------------------------------------------

def test_equal_scores_tiebreak_by_chunk_id_ascending(reranker):
    reranker_cls, fake = reranker
    fake.scores = [0.5, 0.5, 0.5]
    # Input order is deliberately not sorted.
    cands = [_candidate("zeta"), _candidate("alpha"), _candidate("mid")]
    out = reranker_cls.rerank("q", cands, top_k=8)
    assert [r.chunk.id for r in out] == ["alpha", "mid", "zeta"]


# --- H. score count mismatch ------------------------------------------------------

def test_score_count_mismatch_rejected(reranker):
    reranker_cls, fake = reranker
    cands = [_candidate("a"), _candidate("b"), _candidate("c")]
    fake.scores = [0.5, 0.5]  # too few
    with pytest.raises(ValueError):
        reranker_cls.rerank("q", cands, top_k=8)
    fake.scores = [0.5, 0.5, 0.5, 0.5]  # too many
    with pytest.raises(ValueError):
        reranker_cls.rerank("q", cands, top_k=8)


# --- I. non-finite scores ---------------------------------------------------------

@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_non_finite_score_rejected(reranker, bad):
    reranker_cls, fake = reranker
    fake.scores = [bad]
    with pytest.raises(ValueError):
        reranker_cls.rerank("q", [_candidate("a")], top_k=8)


# --- J. invalid score type --------------------------------------------------------

@pytest.mark.parametrize("bad", ["0.5", None, object()])
def test_invalid_score_type_rejected(reranker, bad):
    reranker_cls, fake = reranker
    fake.scores = [bad]
    with pytest.raises(TypeError):
        reranker_cls.rerank("q", [_candidate("a")], top_k=8)


# --- K. input immutability ---------------------------------------------------------

def test_inputs_never_mutated(reranker):
    reranker_cls, fake = reranker
    fake.scores = [0.9, 0.1, 0.5]
    cands = [_candidate("a", text="alpha"), _candidate("b", text="beta"), _candidate("c", text="gamma")]
    before_ids = [c.chunk.id for c in cands]
    before_scores = [c.score for c in cands]
    before_chunks = [c.chunk for c in cands]
    before_texts = [c.chunk.text for c in cands]

    out = reranker_cls.rerank("q", cands, top_k=2)

    # Candidate list untouched.
    assert [c.chunk.id for c in cands] == before_ids
    assert [c.score for c in cands] == before_scores
    assert [c.chunk for c in cands] == before_chunks
    assert [c.chunk.text for c in cands] == before_texts
    # Output is a fresh list of fresh SearchResult objects.
    assert out != cands
    assert all(result not in cands for result in out)
    # Output chunks are the same Chunk objects (never copied/rewritten).
    assert all(any(r.chunk is c.chunk for c in cands) for r in out)


# --- L. batch contract -------------------------------------------------------------

def test_single_batched_backend_call(reranker):
    reranker_cls, fake = reranker
    fake.scores = [0.3, 0.1, 0.2]
    cands = [_candidate("a", text="alpha"), _candidate("b", text="beta"), _candidate("c", text="gamma")]
    reranker_cls.rerank("the query", cands, top_k=8)
    # Exactly one backend call, query preserved, every candidate text in order.
    assert len(fake.calls) == 1
    query, texts = fake.calls[0]
    assert query == "the query"
    assert texts == ("alpha", "beta", "gamma")


# --- M. no embedding coupling --------------------------------------------------------

def test_reranker_has_no_embedding_provider_coupling():
    """CrossEncoderReranker is a re-ranker: it does not extend EmbeddingProvider
    and requires no dimension / embedding concept."""
    assert issubclass(CrossEncoderReranker, Reranker)
    assert not issubclass(CrossEncoderReranker, EmbeddingProvider)
    assert not hasattr(CrossEncoderReranker, "dimension")
