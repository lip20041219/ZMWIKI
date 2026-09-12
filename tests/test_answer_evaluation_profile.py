"""Answer evaluation profile tests: dataset, evidence schema, demonstrations.

Pins what the answer-side profile adds, without re-implementing a metric:

- the committed evidence has a fixed schema with aggregates AND per-query rows;
- every case carries the real numbers for its kind, and the negative cases show
  the intended contrasts (grounded-but-wrong, correct-but-ungrounded,
  cited-vs-uncited, and the negation the baselines cannot see);
- the profile is reproducible from the runner and deterministic;
- the answer dataset, the retrieval dataset, the corpus and the frozen
  retrieval evidence are byte-identical to what they were before this ran;
- the runner reaches no network and no model, and computes no metric itself.
"""

import ast
import hashlib
import json
import pathlib
import sys

import pytest

# Make benchmark/ (and its sibling frozen runners) importable under pytest.
BENCH_DIR = pathlib.Path(__file__).resolve().parents[1] / "benchmark"
sys.path.insert(0, str(BENCH_DIR))

import run_answer_evaluation as answer_profile  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]
DATASET_PATH = BENCH_DIR / "datasets" / "answer_small.json"
EVIDENCE_PATH = BENCH_DIR / "results" / "answer_evaluation_profile.json"
RUNNER_SCRIPT = BENCH_DIR / "run_answer_evaluation.py"

#: Inputs and evidence this round must not touch. A change means someone edited
#: frozen data or history.
FROZEN_SHA256 = {
    "benchmark/datasets/answer_small.json":
        "e569efa315cd9436ef0aaefb586371d1c030b2e050a0ac5006e376aca4f3bdb7",
    "benchmark/datasets/retrieval_small.json":
        "5112ba3c580ddc448a7ef96031930c7a9614c0264808af9d648ecce246c46f81",
    "benchmark/results/retrieval_small_baseline.json":
        "7a5e9b8f141b96c39b149a5842dda994d85c21ab388c9581e1d76e19ced622f9",
    "benchmark/results/retrieval_small_experiments.json":
        "6aa510cc7dce4ba31793e0fd9899d532d5945de5c3d2409f746f1eafc908cb17",
    "benchmark/results/retrieval_strategy_comparison.json":
        "4c45d26508839fa5a7575e6e5cb4d243ca1190721cb34b3c272d783ab2fa8328",
    "benchmark/corpus/doc_a.txt":
        "576c6b4f7ccf621bfa16efd955d7bdf48a9ea7dd4c7f11b335e1915c8c20c287",
    "benchmark/corpus/doc_b.txt":
        "511a8b4df39a2bdd0603e3dad2fd333322a81fc27dc35a05d8019e7338a465b2",
    "benchmark/corpus/doc_c.txt":
        "f86c275d73a9e31d7efd9c8c40903ef4752981058942c585749e9e1849163104",
    "benchmark/corpus/doc_d.txt":
        "d02b1e485f53f8577c5702f154bce7e64c12a9e50599f975c7a352e3711f251f",
    "benchmark/corpus/doc_e.txt":
        "22a51d4e96d584295ff9b1bd79f89bc7789d1e528873926ea1a3eaba90baaadb",
    "benchmark/corpus/doc_f.txt":
        "09d114ab7c8adbd14ba1e24cdee886ef014be446218cc83569185634dc314fc5",
}

TOP_KEYS = [
    "experiment",
    "dataset",
    "corpus_documents",
    "corpus_chunks",
    "chunking",
    "embedding",
    "retriever",
    "reranker",
    "generator",
    "evaluator",
    "tokenizer",
    "groundedness_threshold",
    "top_k",
    "candidate_k",
    "total_cases",
    "evaluated_cases",
    "aggregate",
    "by_kind",
    "retrieval_answer_join",
    "per_query",
    "deterministic",
]

AGGREGATE_KEYS = [
    "mean_reference_coverage",
    "mean_groundedness",
    "mean_citation_coverage",
    "answer_rate",
    "source_presence_rate",
    "mean_source_count",
]

PER_QUERY_KEYS = [
    "id",
    "kind",
    "query",
    "answer_source",
    "citation_source",
    "answer",
    "reference_answer",
    "relevant_chunk_ids",
    "retrieved_chunk_ids",
    "retrieval_hit_at_5",
    "query_id",
    "reference_coverage",
    "groundedness",
    "citation_coverage",
    "has_answer",
    "has_sources",
    "source_count",
    "answer_sentences",
    "grounded_sentences",
    "cited_sentences",
    "reference_tokens",
    "matched_reference_tokens",
]

#: The case kinds the dataset must cover (spec: the eight answer shapes plus the
#: five negative probes).
REQUIRED_KINDS = {
    "single_fact",
    "multi_fact",
    "different_document",
    "context_partial",
    "unanswerable",
    "multi_chunk_support",
    "negative_grounded_wrong",
    "negative_correct_ungrounded",
    "negative_correct_cited",
    "negative_correct_uncited",
    "negative_negation",
}


def _evidence():
    return json.loads(EVIDENCE_PATH.read_text(encoding="utf-8"))


def _dataset():
    return json.loads(DATASET_PATH.read_text(encoding="utf-8"))


def _row(case_id):
    return next(row for row in _evidence()["per_query"] if row["id"] == case_id)


def _digest(relative_path):
    return hashlib.sha256((ROOT / relative_path).read_bytes()).hexdigest()


# --- A. Frozen inputs and retrieval evidence are untouched ---------------------


@pytest.mark.parametrize("relative_path", sorted(FROZEN_SHA256))
def test_frozen_inputs_and_retrieval_evidence_are_unchanged(relative_path):
    assert _digest(relative_path) == FROZEN_SHA256[relative_path], (
        f"{relative_path} changed -- the corpus, datasets and retrieval evidence "
        "must not be modified by this profile"
    )


def test_rerun_writes_only_to_the_given_path(tmp_path):
    answer_profile.main(tmp_path / "fresh.json")
    assert _digest("benchmark/results/retrieval_small_baseline.json") == FROZEN_SHA256[
        "benchmark/results/retrieval_small_baseline.json"
    ]


# --- B. Dataset -----------------------------------------------------------------


def test_dataset_schema_is_complete_and_kinds_are_covered():
    cases = _dataset()["cases"]
    assert len(cases) == answer_profile.EXPECTED_CASES
    ids = [case["id"] for case in cases]
    assert len(set(ids)) == len(ids)
    for case in cases:
        for field in answer_profile.REQUIRED_CASE_FIELDS:
            assert case[field], (case["id"], field)
        assert case["relevant_chunk_ids"]
    assert REQUIRED_KINDS <= {case["kind"] for case in cases}


def test_dataset_is_separate_from_the_retrieval_benchmark():
    retrieval = json.loads(
        (BENCH_DIR / "datasets" / "retrieval_small.json").read_text(encoding="utf-8")
    )
    answer = _dataset()
    # Same corpus and query vocabulary, but a different file with its own cases.
    assert [case["query"] for case in retrieval["cases"]][:3] == [
        "database B-tree index query",
        "cache eviction memory",
        "producer consumer message queue",
    ]
    assert "reference_answer" not in retrieval["cases"][0]
    assert "relevant_chunk_ids" in answer["cases"][0]


def test_loader_rejects_a_malformed_dataset(monkeypatch, tmp_path):
    broken = tmp_path / "broken.json"
    broken.write_text(json.dumps({"cases": []}), encoding="utf-8")
    monkeypatch.setattr(answer_profile, "DATASET_PATH", broken)
    with pytest.raises(RuntimeError, match="no cases list"):
        answer_profile._load_cases()


# --- C. Evidence schema, aggregates, per-query completeness --------------------


def test_evidence_schema_is_fixed():
    evidence = _evidence()
    assert list(evidence) == TOP_KEYS
    assert evidence["experiment"] == answer_profile.EXPERIMENT_ID
    assert evidence["dataset"] == "benchmark/datasets/answer_small.json"
    assert evidence["corpus_documents"] == 6
    assert evidence["corpus_chunks"] == 33
    assert evidence["generator"] == "DeterministicGenerator"
    assert evidence["evaluator"] == "AnswerEvaluator"
    assert evidence["groundedness_threshold"] == 0.5
    assert evidence["total_cases"] == evidence["evaluated_cases"] == 16
    assert evidence["deterministic"] is True


def test_aggregate_block_is_complete_and_in_range():
    aggregate = _evidence()["aggregate"]
    assert list(aggregate) == AGGREGATE_KEYS
    for key in AGGREGATE_KEYS[:5]:
        assert 0.0 <= aggregate[key] <= 1.0, key
    assert aggregate["mean_source_count"] >= 0.0


def test_aggregate_matches_the_per_query_rows():
    evidence = _evidence()
    rows = evidence["per_query"]
    aggregate = evidence["aggregate"]
    assert aggregate["mean_reference_coverage"] == pytest.approx(
        sum(r["reference_coverage"] for r in rows) / len(rows)
    )
    assert aggregate["mean_groundedness"] == pytest.approx(
        sum(r["groundedness"] for r in rows) / len(rows)
    )
    assert aggregate["answer_rate"] == pytest.approx(
        sum(1 for r in rows if r["has_answer"]) / len(rows)
    )
    assert aggregate["source_presence_rate"] == pytest.approx(
        sum(1 for r in rows if r["has_sources"]) / len(rows)
    )


def test_per_query_evidence_is_complete():
    rows = _evidence()["per_query"]
    assert len(rows) == 16
    assert all(list(row) == PER_QUERY_KEYS for row in rows)
    for row in rows:
        assert row["query_id"] == row["id"]
        assert 0.0 <= row["reference_coverage"] <= 1.0
        assert 0.0 <= row["groundedness"] <= 1.0
        assert 0.0 <= row["citation_coverage"] <= 1.0
        assert row["grounded_sentences"] <= row["answer_sentences"]
        assert row["matched_reference_tokens"] <= row["reference_tokens"]
        assert row["has_sources"] == (row["source_count"] > 0)
        assert row["answer_source"] in ("pipeline", "override")
        assert row["citation_source"] in ("context", "pinned")
        assert isinstance(row["retrieval_hit_at_5"], bool)
        assert row["retrieved_chunk_ids"]


def test_by_kind_matches_the_per_query_rows():
    rows = _evidence()["per_query"]
    by_kind = _evidence()["by_kind"]
    for kind, summary in by_kind.items():
        group = [r for r in rows if r["kind"] == kind]
        assert summary["cases"] == len(group)
        assert summary["mean_reference_coverage"] == pytest.approx(
            sum(r["reference_coverage"] for r in group) / len(group)
        )


# --- D. The negative cases carry the intended contrasts ------------------------


def test_grounded_answer_that_does_not_answer_the_question():
    row = _row("n01_grounded_but_off_topic")
    assert row["groundedness"] == 1.0
    assert row["reference_coverage"] < 0.3


def test_correct_answer_that_the_context_does_not_support():
    row = _row("n02_correct_but_ungrounded")
    assert row["reference_coverage"] == 1.0
    assert row["groundedness"] == 0.0


def test_citation_presence_is_independent_of_answer_quality():
    cited = _row("n03_correct_with_sources")
    uncited = _row("n04_correct_without_sources")
    # Same answer and reference: only the sources differ.
    assert cited["answer"] == uncited["answer"]
    assert cited["reference_coverage"] == uncited["reference_coverage"]
    assert cited["groundedness"] == uncited["groundedness"]
    assert (cited["has_sources"], cited["source_count"]) == (True, 5)
    assert (uncited["has_sources"], uncited["source_count"]) == (False, 0)
    assert uncited["citation_coverage"] == 0.0


def test_negation_is_invisible_to_both_baselines():
    row = _row("n05_negation_grounded")
    assert row["groundedness"] == 1.0
    assert row["reference_coverage"] == 1.0


def test_unanswerable_case_shows_grounded_but_uncovered():
    row = _row("a07_unanswerable")
    assert row["retrieval_hit_at_5"] is False
    assert row["reference_coverage"] == 0.0
    assert row["groundedness"] == 1.0  # it quotes the context, it just doesn't answer
    assert row["has_answer"] is True  # the extractive generator never says "I don't know"


# --- E. Retrieval outcome joined to answer quality -----------------------------


def test_retrieval_join_is_computed_over_real_pipeline_rows_only():
    join = _evidence()["retrieval_answer_join"]
    rows = _evidence()["per_query"]
    real = [r for r in rows if r["answer_source"] == "pipeline"]
    assert join["cases_considered"] == len(real) == 7
    assert join["cases_excluded_as_probes"] == len(rows) - len(real) == 9
    hits = [r for r in real if r["retrieval_hit_at_5"]]
    misses = [r for r in real if not r["retrieval_hit_at_5"]]
    assert join["cases_with_retrieval_hit"] == len(hits)
    assert join["cases_with_retrieval_miss"] == len(misses)
    if hits:
        assert join["hit_mean_reference_coverage"] == pytest.approx(
            sum(r["reference_coverage"] for r in hits) / len(hits)
        )
    if misses:
        assert join["miss_mean_reference_coverage"] == pytest.approx(
            sum(r["reference_coverage"] for r in misses) / len(misses)
        )


# --- F. Reproducibility and hygiene --------------------------------------------


def test_profile_is_reproducible_and_deterministic(tmp_path):
    fresh = answer_profile.main(tmp_path / "fresh.json")
    assert fresh == _evidence()
    assert answer_profile._run_profile() == answer_profile._run_profile()


def test_evidence_has_no_leakage():
    raw = EVIDENCE_PATH.read_text(encoding="utf-8")
    for leaked in ("timestamp", "hostname", "username", "datetime", "uuid", "api_key"):
        assert leaked not in raw.lower()
    assert "\\" not in raw  # no Windows absolute path
    assert "://" not in raw  # no URL
    assert "sk-" not in raw  # no credential-shaped string


def test_runner_uses_no_network_no_model_and_no_metric_of_its_own():
    source = RUNNER_SCRIPT.read_text(encoding="utf-8")
    tree = ast.parse(source)
    absolute = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            absolute.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and not node.level:
            absolute.add((node.module or "").split(".")[0])
    assert absolute <= {
        "__future__",
        "json",
        "sys",
        "pathlib",
        "zmrag",
        "run_retrieval_small_baseline",
    }
    for banned in ("requests", "urllib", "httpx", "socket", "sentence_transformers"):
        assert banned not in absolute
    # Metrics come from AnswerEvaluator; the harness never recomputes one.
    assert "AnswerEvaluator" in source
    for banned in ("def _ndcg", "math.log2", "def _coverage("):
        assert banned not in source


def test_evaluator_is_not_wired_into_the_pipeline():
    # Evaluation stays a separate stage: the pipeline has no evaluator attribute.
    from zmrag import DeterministicRAGPipeline

    assert "answer_evaluator" not in dir(DeterministicRAGPipeline)
    source = (ROOT / "src" / "zmrag" / "pipeline.py").read_text(encoding="utf-8")
    assert "answer_evaluation" not in source
