"""Phase 12 Step 3: real CrossEncoder rerank profile evidence + runner pins.

This module verifies the committed
``benchmark/results/retrieval_semantic_rerank_profile.json`` and the runner
that produced it, and it *never* makes the default suite hit the network, the
CrossEncoder model, or torch.

Default-suite tests (offline, no ML dependency / model / network):

- the rerank evidence is a valid JSON dict with the fixed top-level schema:
  ``profile == "semantic_rerank"``, locked embedding identity
  (``embedding_model_id`` == the Phase 11 MiniLM model, ``embedding_dimension``
  == 384), the locked reranker identity
  (``reranker_model_id`` == ``cross-encoder/ms-marco-MiniLM-L-6-v2``),
  ``deterministic`` false at the top level and on every config row, candidate_k
  33 / top_k 8, and ``configs`` exactly the locked R1 -> R2 -> R3 matrix in
  order (dense / hybrid+weighted / hybrid+rrf, reranker always
  "cross_encoder");
- every config row carries the six Phase 8 metrics plus ``delta_vs_phase11``
  (only mrr/map/ndcg deltas) and, re-reading the Phase 11 evidence read-only,
  each recorded delta equals ``rerank_metric - phase11_metric`` within float
  tolerance (R1->S1, R2->S2, R3->S3);
- the runner source wires ``CrossEncoderReranker`` (the real reranker) as a
  true 33 -> 8 two-stage stage and contains no ``DeterministicEmbedder`` /
  ``DeterministicReranker`` / pipeline internals;
- the three frozen deterministic Phase 10 evidence files are byte-identical to
  their pinned SHA-256, and the Phase 11 semantic evidence is byte-identical
  to its pinned SHA-256 (none modified).

The runner is only *imported* here -- its heavy work happens in ``main()``,
which constructs the models. No offline test calls it.

Integration tests (real sentence-transformers embedding + CrossEncoder model)
live at the bottom and are gated behind ``ZMRAG_SEMANTIC_INTEGRATION=1`` so
the default suite skips them and never downloads/loads a model. Run them
explicitly:

    set ZMRAG_SEMANTIC_INTEGRATION=1
    python -m pytest tests/test_semantic_rerank_profile.py::\
        test_integration_real_semantic_rerank_profile -q

The integration test re-runs the real runner to a fresh output path and checks
that (a) it reproduces the committed profile under this pinned environment and
(b) the two-stage R1/R2/R3 rows all complete with valid metrics and honest
Phase 11 deltas.
"""

import hashlib
import json
import math
import os
import sys
from pathlib import Path

import pytest

# Make benchmark/ (and the sibling baseline runner) importable under pytest.
BENCH_DIR = Path(__file__).resolve().parents[1] / "benchmark"
sys.path.insert(0, str(BENCH_DIR))

import run_semantic_rerank_profile as rerank_profile  # noqa: E402  (needs benchmark path)

RESULTS_DIR = BENCH_DIR / "results"
EVIDENCE_PATH = RESULTS_DIR / "retrieval_semantic_rerank_profile.json"
PHASE11_PATH = RESULTS_DIR / "retrieval_semantic_profile.json"

#: SHA-256 of the three frozen Phase 10 deterministic evidence files at HEAD
#: 03c093f (unchanged) and the committed Phase 11 semantic evidence at HEAD
#: f6a2c52. Byte-identical re-runs are the deterministic profile's contract;
#: the Phase 11 file must also not be touched by the rerank runner.
FROZEN_EVIDENCE_SHA256 = {
    "retrieval_small_baseline.json":
        "7a5e9b8f141b96c39b149a5842dda994d85c21ab388c9581e1d76e19ced622f9",
    "retrieval_small_experiments.json":
        "6aa510cc7dce4ba31793e0fd9899d532d5945de5c3d2409f746f1eafc908cb17",
    "retrieval_truncation.json":
        "159c697297a1c3991adedddf01bad61bbc2e976cf8e9cb6f78617306b7d520c6",
    "retrieval_semantic_profile.json":
        "fb4e6f4af405e02c05a82559225e3ce768b6e9c5e18ef7fc07af87943aa65a0f",
}

EMBEDDING_MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
RERANKER_MODEL_ID = "cross-encoder/ms-marco-MiniLM-L-6-v2"
METRIC_KEYS = ("hit_rate", "recall", "precision", "mrr", "map", "ndcg")
DELTA_KEYS = ("mrr_delta", "map_delta", "ndcg_delta")
COUNT_KEYS = ("total_cases", "evaluated_cases")
EXPECTED_ROWS = (("R1", "dense", None), ("R2", "hybrid", "weighted"), ("R3", "hybrid", "rrf"))
BASELINE_ID_BY_ROW = {"R1": "S1", "R2": "S2", "R3": "S3"}


def _evidence() -> dict:
    data = json.loads(EVIDENCE_PATH.read_text(encoding="utf-8"))
    assert isinstance(data, dict)
    return data


def _phase11() -> dict:
    data = json.loads(PHASE11_PATH.read_text(encoding="utf-8"))
    assert isinstance(data, dict)
    return data


# --- A. profile / experiment identity ---------------------------------------


def test_rerank_evidence_top_level_profile():
    ev = _evidence()
    assert ev["profile"] == "semantic_rerank"
    assert ev["experiment"] == "retrieval_semantic_rerank_profile"
    assert ev["candidate_k"] == 33
    assert ev["top_k"] == 8
    assert ev["corpus_documents"] == 6
    assert ev["corpus_chunks"] == 33
    assert ev["total_cases"] == 8 and ev["evaluated_cases"] == 8


# --- B. model identity -------------------------------------------------------


def test_rerank_evidence_model_identity():
    ev = _evidence()
    assert ev["embedding_model_id"] == EMBEDDING_MODEL_ID
    assert ev["embedding_dimension"] == 384
    assert ev["reranker_model_id"] == RERANKER_MODEL_ID


def test_rerank_evidence_environment_facts():
    ev = _evidence()
    for dep in ("python", "sentence_transformers", "torch", "numpy", "transformers"):
        assert dep in ev["environment"], f"environment missing {dep!r}"
    assert "same_environment_reproducible" in ev
    assert isinstance(ev["same_environment_reproducible"], bool)


# --- C. matrix ---------------------------------------------------------------


def test_rerank_evidence_matrix_locked():
    ev = _evidence()
    rows = ev["configs"]
    assert [r["id"] for r in rows] == ["R1", "R2", "R3"]
    for row, (rid, retriever, fusion) in zip(rows, EXPECTED_ROWS):
        assert row["retriever"] == retriever
        assert row["fusion"] == fusion
        assert row["reranker"] == "cross_encoder"
        assert row["candidate_k"] == 33
        assert row["top_k"] == 8
        for key in COUNT_KEYS + METRIC_KEYS:
            assert key in row, f"row {row['id']} missing {key!r}"
        assert set(row["delta_vs_phase11"]) == set(DELTA_KEYS)
        assert row["total_cases"] == 8 and row["evaluated_cases"] == 8


def test_rerank_evidence_metrics_valid():
    ev = _evidence()
    for row in ev["configs"]:
        for key in METRIC_KEYS:
            value = row[key]
            assert isinstance(value, (int, float)), f"{key} not numeric: {value!r}"
            assert 0.0 <= value <= 1.0, f"{key} out of [0,1]: {value!r}"


# --- D. deterministic flags ---------------------------------------------------


def test_rerank_evidence_deterministic_false():
    ev = _evidence()
    assert ev["deterministic"] is False
    for row in ev["configs"]:
        assert row["deterministic"] is False


# --- F. delta consistency vs the Phase 11 baseline -----------------------------


def test_rerank_evidence_delta_consistency():
    """Each row's delta_vs_phase11 equals rerank_metric - phase11_metric (R->S)."""
    phase11 = _phase11()
    baseline_by_id = {row["id"]: row for row in phase11["configs"]}
    for row in _evidence()["configs"]:
        base = baseline_by_id[BASELINE_ID_BY_ROW[row["id"]]]
        delta = row["delta_vs_phase11"]
        assert set(delta) == set(DELTA_KEYS), f"unexpected delta keys: {sorted(delta)}"
        for metric in ("mrr", "map", "ndcg"):
            actual = row[metric] - base[metric]
            assert math.isclose(delta[f"{metric}_delta"], actual,
                                rel_tol=1e-12, abs_tol=1e-12), (
                f"{row['id']} {metric}: recorded {delta[f'{metric}_delta']} "
                f"!= computed {actual}"
            )


# --- runner source wiring: real CrossEncoder, true two-stage -------------------

def test_rerank_runner_locked_configuration():
    """Constants pin the two models and the R1/R2/R3 two-stage matrix exactly."""
    rp = rerank_profile
    assert rp.EMBEDDING_MODEL_ID == EMBEDDING_MODEL_ID
    assert rp.RERANKER_MODEL_ID == RERANKER_MODEL_ID
    assert rp.TOP_K == 8
    assert rp.CANDIDATE_K == 33
    assert [c["id"] for c in rp.CONFIGS] == ["R1", "R2", "R3"]
    assert all(c["reranker"] == "cross_encoder" for c in rp.CONFIGS)


def test_rerank_runner_is_real_two_stage_and_ml_core_free():
    """The runner wires CrossEncoderReranker over a 33-pool; no deterministic
    embedder/reranker, no pipeline internals, no semantic-provider substitution."""
    source = Path(rerank_profile.__file__).read_text(encoding="utf-8")
    code = source[source.index("from __future__") :]
    # Real optional providers -- both are the Phase 12 boundary classes.
    assert "SemanticEmbeddingProvider" in code
    assert "CrossEncoderReranker" in code
    # No deterministic stand-ins and no core modification.
    assert "DeterministicEmbedder" not in code
    assert "DeterministicReranker" not in code
    assert "DeterministicRAGPipeline" not in code
    # The two-stage shape is explicit in the code: first stage asks for the full
    # 33 candidate pool (never 8), then reranks that pool down to top_k=8.
    assert "top_k=CANDIDATE_K" in code
    assert "reranker.rerank(query, pool, top_k=TOP_K)" in code


# --- E. frozen evidence byte-identical to HEAD --------------------------------

@pytest.mark.parametrize("name", sorted(FROZEN_EVIDENCE_SHA256))
def test_frozen_evidence_unchanged(name):
    path = RESULTS_DIR / name
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    assert digest == FROZEN_EVIDENCE_SHA256[name], (
        f"{name} changed vs its committed HEAD -- frozen evidence must not be modified"
    )


# ---------------------------------------------------------------------------
# Integration: real sentence-transformers embedding + CrossEncoder model.
# Skipped by default; enable with ZMRAG_SEMANTIC_INTEGRATION=1 (requires the
# `semantic` optional dep and a model cache or network). The default suite
# must never construct a model.
# ---------------------------------------------------------------------------

def _integration_enabled() -> bool:
    return bool(os.environ.get("ZMRAG_SEMANTIC_INTEGRATION"))


@pytest.mark.skipif(
    not _integration_enabled(),
    reason="real-model integration; set ZMRAG_SEMANTIC_INTEGRATION=1 to run",
)
def test_integration_real_semantic_rerank_profile(tmp_path):
    """Re-run the real rerank profile to a fresh path; compare to committed.

    Under this pinned environment the model output is reproducible (the runner
    itself verifies two fresh passes are identical), so the fresh evidence must
    match the committed one on identity/matrix/determinism/deltas, and the
    recorded reproducibility flag must agree with the committed evidence.
    """
    out = tmp_path / "retrieval_semantic_rerank_profile.json"
    fresh = rerank_profile.main(out)
    committed = _evidence()

    assert fresh["same_environment_reproducible"] is True
    assert fresh["same_environment_reproducible"] == committed["same_environment_reproducible"]
    for key in (
        "profile", "embedding_model_id", "embedding_dimension", "reranker_model_id",
        "deterministic", "candidate_k", "top_k",
    ):
        assert fresh[key] == committed[key], f"{key} drifted on re-run"

    # Every config row re-ran end to end (retrieval -> CrossEncoder rerank ->
    # evaluation) with valid metrics, honest Phase 11 deltas, deterministic=false.
    fresh_by_id = {row["id"]: row for row in fresh["configs"]}
    phase11 = _phase11()
    baseline_by_id = {row["id"]: row for row in phase11["configs"]}
    assert [row["id"] for row in fresh["configs"]] == ["R1", "R2", "R3"]
    for row_id, committed_row in zip(
        ["R1", "R2", "R3"], committed["configs"]
    ):
        row = fresh_by_id[row_id]
        assert row["reranker"] == "cross_encoder"
        assert row["deterministic"] is False
        for key in METRIC_KEYS:
            assert 0.0 <= row[key] <= 1.0
        # Fresh deltas match the committed deltas and the read-only Phase 11 rows.
        assert row["delta_vs_phase11"] == committed_row["delta_vs_phase11"]
        base = baseline_by_id[BASELINE_ID_BY_ROW[row_id]]
        assert math.isclose(row["mrr"] - base["mrr"], row["delta_vs_phase11"]["mrr_delta"],
                            rel_tol=1e-12, abs_tol=1e-12)
