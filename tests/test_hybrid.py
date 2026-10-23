"""Hybrid retrieval tests: fusion strategies + HybridRetriever orchestration."""

import pytest

from zmwiki import (
    Chunk,
    FusionStrategy,
    HybridRetriever,
    RRFFusion,
    Retriever,
    SearchResult,
    WeightedScoreFusion,
)


def _chunk(cid):
    return Chunk(id=cid, document_id="doc", text=cid, position=0)


def _hits(*pairs):
    """pairs of (chunk_id, score) -> ordered SearchResult list."""
    return [SearchResult(chunk=_chunk(cid), score=score) for cid, score in pairs]


class FakeRetriever(Retriever):
    def __init__(self, preset=None, error=None):
        self.preset = preset or []
        self.error = error
        self.calls = 0
        self.last_query = None
        self.last_top_k = None

    def retrieve(self, query, top_k=10):
        self.calls += 1
        self.last_query = query
        self.last_top_k = top_k
        if self.error is not None:
            raise self.error
        return list(self.preset)


def _hybrid(dense=None, lexical=None, fusion=None):
    return HybridRetriever(
        dense_retriever=FakeRetriever(preset=dense),
        lexical_retriever=FakeRetriever(preset=lexical),
        fusion=fusion or WeightedScoreFusion(),
    )


# --- WeightedScoreFusion math ---


def test_weighted_query_local_minmax_ranking():
    dense = _hits(("A", 0.9), ("B", 0.8), ("C", 0.1))
    lexical = _hits(("B", 5.0), ("C", 3.0), ("D", 1.0))
    fused = WeightedScoreFusion().fuse(dense, lexical, top_k=4)
    assert [r.chunk.id for r in fused] == ["B", "A", "C", "D"]
    # B = 0.5*((0.8-0.1)/0.8) + 0.5*((5-1)/4) = 0.5*0.875 + 0.5*1.0 = 0.9375
    by_id = {r.chunk.id: r.score for r in fused}
    assert by_id["B"] == pytest.approx(0.9375)


def test_weighted_normalization_formula():
    assert WeightedScoreFusion().fuse([], [], top_k=3) == []
    # min=0, max=10, score=5 -> 0.5 (single-source normalization sanity)
    single = WeightedScoreFusion(dense_weight=1.0, lexical_weight=0.0).fuse(
        _hits(("X", 5.0), ("Y", 0.0), ("Z", 10.0)), [], top_k=3
    )
    by_id = {r.chunk.id: r.score for r in single}
    assert by_id["Y"] == pytest.approx(0.0)
    assert by_id["X"] == pytest.approx(0.5)
    assert by_id["Z"] == pytest.approx(1.0)


def test_weighted_max_equals_min_normalizes_to_one():
    fusion = WeightedScoreFusion(dense_weight=1.0, lexical_weight=0.0)
    fused = fusion.fuse(_hits(("A", 0.7), ("B", 0.7)), [], top_k=2)
    assert all(r.score == pytest.approx(1.0) for r in fused)
    assert [r.chunk.id for r in fused] == ["A", "B"]  # equal -> tie by id


def test_weighted_weights_effect():
    dense = _hits(("Y", 0.9))  # single hit -> norm 1.0
    lexical = _hits(("X", 9.0))
    only_dense = WeightedScoreFusion(dense_weight=1.0, lexical_weight=0.0).fuse(
        dense, lexical, top_k=2
    )
    assert [r.chunk.id for r in only_dense] == ["Y", "X"]
    only_lexical = WeightedScoreFusion(dense_weight=0.0, lexical_weight=1.0).fuse(
        dense, lexical, top_k=2
    )
    assert [r.chunk.id for r in only_lexical] == ["X", "Y"]


def test_weighted_score_range_under_equal_weights():
    fused = WeightedScoreFusion().fuse(
        _hits(("A", 0.9), ("B", 0.2)), _hits(("B", 7.0), ("C", 3.0)), top_k=3
    )
    assert all(0.0 <= r.score <= 1.0 for r in fused)


@pytest.mark.parametrize("bad", [True, -1, "1"])
def test_weighted_invalid_weight_rejected(bad):
    with pytest.raises((TypeError, ValueError)):
        WeightedScoreFusion(dense_weight=bad)  # type: ignore[arg-type]


def test_weighted_both_weights_zero_rejected():
    with pytest.raises(ValueError):
        WeightedScoreFusion(dense_weight=0.0, lexical_weight=0.0)


# --- RRFFusion math ---


def test_rrf_single_source_rank_formula():
    k = 60
    fused = RRFFusion(rrf_k=k).fuse(_hits(("X", 0.1)), [], top_k=1)
    assert fused[0].score == pytest.approx(1.0 / (k + 1))


def test_rrf_doc_in_both_sources_scores_higher():
    k = 60
    dense = _hits(("A", 1.0), ("B", 1.0), ("C", 1.0))
    lexical = _hits(("B", 1.0), ("C", 1.0), ("D", 1.0))
    fused = RRFFusion(rrf_k=k).fuse(dense, lexical, top_k=4)
    by_id = {r.chunk.id: r.score for r in fused}
    assert by_id["B"] == pytest.approx(1 / (k + 2) + 1 / (k + 1))
    assert by_id["C"] == pytest.approx(1 / (k + 3) + 1 / (k + 2))
    assert by_id["A"] == pytest.approx(1 / (k + 1))
    assert by_id["D"] == pytest.approx(1 / (k + 3))
    # present in both -> B, C above single-source A, D
    assert by_id["B"] > by_id["A"] and by_id["C"] > by_id["D"]
    assert [r.chunk.id for r in fused] == ["B", "C", "A", "D"]


@pytest.mark.parametrize("bad", [0, -1, 2.5, "60", True])
def test_rrf_k_validation(bad):
    with pytest.raises((ValueError, TypeError)):
        RRFFusion(rrf_k=bad)  # type: ignore[arg-type]


# --- deduplication ---


def test_dedupe_by_chunk_id():
    dense = _hits(("A", 0.9), ("B", 0.8), ("C", 0.1))
    lexical = _hits(("B", 5.0), ("C", 3.0), ("D", 1.0))
    fused = WeightedScoreFusion().fuse(dense, lexical, top_k=10)
    ids = [r.chunk.id for r in fused]
    assert len(ids) == len(set(ids)) == 4
    assert sorted(ids) == ["A", "B", "C", "D"]


# --- HybridRetriever orchestration ---


def test_hybrid_injection_and_calls():
    dense, lexical = FakeRetriever(preset=_hits(("A", 1.0))), FakeRetriever(
        preset=_hits(("A", 9.0))
    )
    hybrid = HybridRetriever(dense, lexical, RRFFusion())
    out = hybrid.retrieve("q", top_k=1, candidate_k=3)
    assert dense.calls == 1 and lexical.calls == 1
    assert dense.last_top_k == 3 and lexical.last_top_k == 3
    assert dense.last_query == lexical.last_query == "q"
    assert len(out) == 1


def test_hybrid_rejects_wrong_types():
    with pytest.raises(TypeError):
        HybridRetriever("d", FakeRetriever(), RRFFusion())  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        HybridRetriever(FakeRetriever(), "l", RRFFusion())  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        HybridRetriever(FakeRetriever(), FakeRetriever(), "fusion")  # type: ignore[arg-type]


def test_hybrid_empty_both():
    assert _hybrid([], []).retrieve("q") == []


def test_hybrid_dense_only():
    hybrid = _hybrid(dense=_hits(("A", 0.9), ("B", 0.1)), lexical=[], fusion=RRFFusion())
    assert [r.chunk.id for r in hybrid.retrieve("q", top_k=5)] == ["A", "B"]


def test_hybrid_lexical_only():
    hybrid = _hybrid(dense=[], lexical=_hits(("X", 9.0), ("Y", 1.0)), fusion=RRFFusion())
    assert [r.chunk.id for r in hybrid.retrieve("q", top_k=5)] == ["X", "Y"]


def test_hybrid_final_top_k_truncates_candidates():
    dense = _hits(*[(c, float(i)) for i, c in enumerate(["A", "B", "C", "D", "E"])])
    lexical = _hits(("A", 1.0), ("C", 1.0))
    hybrid = HybridRetriever(FakeRetriever(dense), FakeRetriever(lexical), RRFFusion())
    out = hybrid.retrieve("q", top_k=2, candidate_k=5)
    assert len(out) == 2


def test_hybrid_candidate_k_lt_top_k_raises():
    hybrid = _hybrid([], [])
    with pytest.raises(ValueError):
        hybrid.retrieve("q", top_k=10, candidate_k=5)


@pytest.mark.parametrize("kw, bad", [("top_k", 0), ("top_k", -1), ("top_k", True), ("top_k", 2.5), ("candidate_k", 0), ("candidate_k", -1), ("candidate_k", True), ("candidate_k", "50")])
def test_hybrid_invalid_top_and_candidate_k(kw, bad):
    hybrid = _hybrid([], [])
    with pytest.raises((ValueError, TypeError)):
        hybrid.retrieve("q", **{kw: bad})  # type: ignore[arg-type]


@pytest.mark.parametrize("bad", ["", "   \n\t", 42])
def test_hybrid_invalid_query(bad):
    hybrid = _hybrid([], [])
    with pytest.raises(TypeError):
        hybrid.retrieve(bad)  # type: ignore[arg-type]


def test_hybrid_error_propagation():
    dense_boom = FakeRetriever(error=RuntimeError("dense boom"))
    hybrid = HybridRetriever(dense_boom, FakeRetriever(), RRFFusion())
    with pytest.raises(RuntimeError):
        hybrid.retrieve("q")

    lex_boom = FakeRetriever(preset=_hits(("A", 1.0)), error=ValueError("lex boom"))
    dense_ok = FakeRetriever(preset=_hits(("A", 0.9)))
    hybrid2 = HybridRetriever(dense_ok, lex_boom, RRFFusion())
    with pytest.raises(ValueError):
        hybrid2.retrieve("q")
    assert dense_ok.calls == 1  # dense still ran once before lexical failed


def test_hybrid_repeated_query_deterministic():
    dense = FakeRetriever(_hits(("A", 0.9), ("B", 0.8), ("C", 0.1)))
    lexical = FakeRetriever(_hits(("B", 5.0), ("C", 3.0)))
    hybrid = HybridRetriever(dense, lexical, WeightedScoreFusion())
    first = [(r.chunk.id, r.score) for r in hybrid.retrieve("q", top_k=3)]
    second = [(r.chunk.id, r.score) for r in hybrid.retrieve("q", top_k=3)]
    assert first == second


# --- strategy injection ---


class FakeFusion(FusionStrategy):
    def __init__(self):
        self.calls = 0
        self.last_top_k = None

    def fuse(self, dense_results, lexical_results, top_k):
        self.calls += 1
        self.last_top_k = top_k
        return _hits(("FUSED", 1.0))


def test_hybrid_injects_fake_fusion_strategy():
    dense = FakeRetriever(_hits(("A", 1.0)))
    lexical = FakeRetriever(_hits(("B", 1.0)))
    fusion = FakeFusion()
    hybrid = HybridRetriever(dense, lexical, fusion)
    out = hybrid.retrieve("q", top_k=4, candidate_k=6)
    assert [r.chunk.id for r in out] == ["FUSED"]
    assert fusion.calls == 1
    assert fusion.last_top_k == 4
    assert dense.calls == lexical.calls == 1


def test_public_import():
    from zmwiki import __all__

    assert {"HybridRetriever", "FusionStrategy", "WeightedScoreFusion", "RRFFusion"} <= set(__all__)
    assert isinstance(WeightedScoreFusion(), FusionStrategy)
    assert isinstance(RRFFusion(), FusionStrategy)
