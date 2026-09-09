"""Phase 11 Step 3: real semantic retrieval profile evidence + runner pins.

This module verifies the committed ``benchmark/results/retrieval_semantic_profile.json``
and the runner that produced it, and it *never* makes the default suite hit the
network, the model, or torch.

Default-suite tests (offline, no ML dependency / model / network):

- the semantic evidence is a valid JSON dict with the fixed top-level schema:
  ``profile == "semantic"``, a stable model identity
  (``model_id`` / ``model_name`` == the pinned MiniLM model), ``deterministic``
  is false at the top level and on every config row, the six Phase 8 metrics
  are present and well-typed, and ``configs`` are exactly the locked
  S1 -> S2 -> S3 matrix in order (dense / hybrid+weighted / hybrid+rrf,
  reranker always null, top-level candidate_k=33 / top_k=8);
- the runner source wires ``SemanticEmbeddingProvider`` (the real provider)
  and contains no ``DeterministicEmbedder``, no reranker, and no pipeline
  components -- the semantic dense path is a plain real ``DenseRetriever``;
- the three frozen deterministic Phase 10 evidence files are byte-identical
  to their pinned SHA-256 (unchanged from the committed corpus).

The runner is only *imported* here -- its heavy work happens in ``main()``,
which constructs the model. No offline test calls it.

Integration tests (real sentence-transformers model) live at the bottom and
are gated behind ``ZMRAG_SEMANTIC_INTEGRATION=1`` so the default suite skips
them and never downloads/loads a model. Run them explicitly:

    set ZMRAG_SEMANTIC_INTEGRATION=1
    python -m pytest -m "" tests/test_semantic_profile.py -q   # (or by node id)

    # cross-platform one-liner: ZMRAG_SEMANTIC_INTEGRATION=1 python -m pytest
    # tests/test_semantic_profile.py::test_integration_real_semantic_profile

The integration test re-runs the real runner to a fresh output path and
checks that (a) it reproduces the committed metrics under this pinned
environment, and (b) real semantic dense retrieval clears the synthetic
floor: hit_rate / recall above zero on the 7 corpus-grounded cases.
"""

import hashlib
import json
import os
import sys
from pathlib import Path

import pytest

# Make benchmark/ (and the sibling baseline runner) importable under pytest.
BENCH_DIR = Path(__file__).resolve().parents[1] / "benchmark"
sys.path.insert(0, str(BENCH_DIR))

import run_semantic_profile as semantic_profile  # noqa: E402  (needs benchmark path)

RESULTS_DIR = BENCH_DIR / "results"
EVIDENCE_PATH = RESULTS_DIR / "retrieval_semantic_profile.json"

#: SHA-256 of the three frozen Phase 10 deterministic evidence files at HEAD
#: 03c093f. Byte-identical re-runs are the deterministic profile's contract.
FROZEN_EVIDENCE_SHA256 = {
    "retrieval_small_baseline.json":
        "7a5e9b8f141b96c39b149a5842dda994d85c21ab388c9581e1d76e19ced622f9",
    "retrieval_small_experiments.json":
        "6aa510cc7dce4ba31793e0fd9899d532d5945de5c3d2409f746f1eafc908cb17",
    "retrieval_truncation.json":
        "159c697297a1c3991adedddf01bad61bbc2e976cf8e9cb6f78617306b7d520c6",
}

METRIC_KEYS = ("hit_rate", "recall", "precision", "mrr", "map", "ndcg")
COUNT_KEYS = ("total_cases", "evaluated_cases")
MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"


def _evidence() -> dict:
    data = json.loads(EVIDENCE_PATH.read_text(encoding="utf-8"))
    assert isinstance(data, dict)
    return data


# --- 1/7/8. runner wiring: real provider, no DeterministicEmbedder -----------


def test_semantic_runner_uses_real_provider_and_no_deterministic_embedder():
    """The runner code wires SemanticEmbeddingProvider and never builds on
    DeterministicEmbedder / a reranker / pipeline internals."""
    source = Path(semantic_profile.__file__).read_text(encoding="utf-8")
    # Code only -- the module docstring may name the deterministic classes in
    # prose, but no *code* may import or construct them.
    code = source[source.index("from __future__") :]
    assert "SemanticEmbeddingProvider" in code
    assert "DeterministicEmbedder" not in code
    assert "DeterministicRAGPipeline" not in code
    assert "DeterministicReranker" not in code
    # reranking appears only as the locked null config field, never a class use.
    assert "Reranker" not in code


def test_semantic_runner_locked_configuration():
    """Constants pin the model and the minimal S1/S2/S3 matrix exactly."""
    sp = semantic_profile
    assert sp.MODEL_NAME == MODEL_ID
    assert sp.CANDIDATE_K == 33
    assert sp.TOP_K == 8
    assert [c["id"] for c in sp.CONFIGS] == ["S1", "S2", "S3"]
    assert all(c["reranker"] is None for c in sp.CONFIGS)
    assert [c["retriever"] for c in sp.CONFIGS] == ["dense", "hybrid", "hybrid"]
    assert [c["fusion"] for c in sp.CONFIGS] == [None, "weighted", "rrf"]


# --- 2/3/4. evidence schema: profile / model identity / deterministic=false ---


def test_semantic_evidence_top_level_schema():
    ev = _evidence()
    assert ev["profile"] == "semantic"
    assert ev["experiment"] == "retrieval_semantic_profile"
    assert ev["model_id"] == MODEL_ID
    assert ev["model_name"] == MODEL_ID
    assert isinstance(ev["embedding_dimension"], int) and ev["embedding_dimension"] > 0
    assert ev["candidate_k"] == 33
    assert ev["top_k"] == 8
    assert ev["corpus_documents"] == 6
    assert ev["corpus_chunks"] == 33
    assert ev["deterministic"] is False
    assert "same_environment_reproducible" in ev  # present, bool, honest value
    assert isinstance(ev["same_environment_reproducible"], bool)
    for dep in ("python", "sentence_transformers", "torch", "numpy"):
        assert dep in ev["environment"], f"environment missing {dep!r}"
    # The semantic profile must never be labelled deterministic evidence.
    assert "deterministic" not in ev or ev["deterministic"] is False


def test_semantic_evidence_rows_deterministic_false_in_order():
    ev = _evidence()
    rows = ev["configs"]
    assert [r["id"] for r in rows] == ["S1", "S2", "S3"]
    for row in rows:
        assert row["deterministic"] is False
        assert row["reranker"] is None
    assert rows[0]["retriever"] == "dense" and rows[0]["fusion"] is None
    assert rows[1]["retriever"] == "hybrid" and rows[1]["fusion"] == "weighted"
    assert rows[2]["retriever"] == "hybrid" and rows[2]["fusion"] == "rrf"


# --- 5/6. evidence metrics valid, all 8 cases evaluated -----------------------


def test_semantic_evidence_metrics_valid():
    ev = _evidence()
    rows = ev["configs"]
    for row in rows:
        for key in COUNT_KEYS + METRIC_KEYS:
            assert key in row, f"row {row['id']} missing {key!r}"
        assert row["total_cases"] == 8
        assert row["evaluated_cases"] == 8
        for key in METRIC_KEYS:
            value = row[key]
            assert isinstance(value, (int, float)), f"{key} not numeric: {value!r}"
            assert 0.0 <= value <= 1.0, f"{key} out of [0,1]: {value!r}"
    assert ev["total_cases"] == 8 and ev["evaluated_cases"] == 8


def test_semantic_dense_evidence_is_off_the_synthetic_floor():
    """S1 (real semantic dense) must not sit at the DeterministicEmbedder floor.

    Synthetic deterministic dense at the same top_k=8 measures hit_rate 0.25 /
    recall 0.25 / mrr ~0.04 (near-random ranks). Real semantic dense here
    reaches the full 7-of-7 corpus-grounded ceiling at top_k=8, so its
    hit_rate / recall must exceed that floor.
    """
    s1 = _evidence()["configs"][0]
    assert s1["id"] == "S1"
    assert s1["hit_rate"] > 0.0
    assert s1["recall"] > 0.0
    assert s1["hit_rate"] > 0.25 and s1["recall"] > 0.25


# --- 9. frozen deterministic evidence byte-identical to HEAD ------------------


@pytest.mark.parametrize("name", sorted(FROZEN_EVIDENCE_SHA256))
def test_frozen_deterministic_evidence_unchanged(name):
    path = RESULTS_DIR / name
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    assert digest == FROZEN_EVIDENCE_SHA256[name], (
        f"{name} changed vs HEAD 03c093f -- frozen evidence must not be modified"
    )


# ---------------------------------------------------------------------------
# Integration: real sentence-transformers model. Skipped by default; enable
# with ZMRAG_SEMANTIC_INTEGRATION=1 (requires the `semantic` optional dep and
# a model cache or network). The default suite must never construct a model.
# ---------------------------------------------------------------------------

def _integration_enabled() -> bool:
    return bool(os.environ.get("ZMRAG_SEMANTIC_INTEGRATION"))


@pytest.mark.skipif(
    not _integration_enabled(),
    reason="real-model integration; set ZMRAG_SEMANTIC_INTEGRATION=1 to run",
)
def test_integration_real_semantic_profile(tmp_path):
    """Re-run the real semantic profile to a fresh path; compare to committed.

    Under this pinned environment the model output is reproducible (the runner
    itself verifies two fresh passes are identical), so the fresh evidence must
    equal the committed one, and S1 real dense must clear zero.
    """
    out = tmp_path / "retrieval_semantic_profile.json"
    fresh = semantic_profile.main(out)
    committed = _evidence()

    assert fresh["same_environment_reproducible"] is True
    for key in (
        "profile", "model_id", "model_name", "embedding_dimension",
        "deterministic", "candidate_k", "top_k", "configs",
    ):
        assert fresh[key] == committed[key], f"{key} drifted on re-run"
    s1 = fresh["configs"][0]
    assert s1["id"] == "S1"
    assert s1["hit_rate"] > 0.0 and s1["recall"] > 0.0
