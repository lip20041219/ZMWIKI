"""Claim attribution profile tests: evidence schema, funnel, reproducibility.

Pins what the attribution benchmark adds, without re-implementing a metric:

- the committed evidence has a fixed schema, per-claim rows for every case, and
  a funnel whose arithmetic agrees with those rows;
- the two rates stay separate in the evidence (supported-but-uncited claims are
  recorded, not folded away), and the limitation probes are present rather than
  smoothed over;
- the runner is byte-identical across two independent runs;
- the corpus, the retrieval dataset and the frozen retrieval evidence are
  untouched, and the runner writes only where it is told to;
- the runner reaches no network, no model, and no key, and computes no metric of
  its own.
"""

import ast
import json
import pathlib
import sys

import pytest

from _frozen_evidence import frozen_digest_matches, normalized_lf_bytes

# Make benchmark/ (and its sibling runners) importable under pytest.
BENCH_DIR = pathlib.Path(__file__).resolve().parents[1] / "benchmark"
sys.path.insert(0, str(BENCH_DIR))

import run_attribution_profile as attribution_profile  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]
EVIDENCE_PATH = BENCH_DIR / "results" / "answer_attribution_profile.json"
RUNNER_SCRIPT = BENCH_DIR / "run_attribution_profile.py"

#: The frozen inputs this round must not touch (spec: corpus, retrieval dataset,
#: retrieval evidence). A change here means someone edited data or history.
FROZEN_SHA256 = {
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
    "generator",
    "extractor",
    "attributor",
    "tokenizer",
    "attribution_threshold",
    "top_k",
    "candidate_k",
    "total_cases",
    "aggregate",
    "funnel",
    "per_claim",
    "deterministic",
]

AGGREGATE_KEYS = [
    "supported_claim_rate",
    "attributed_claim_rate",
    "mean_support_score",
    "total_claims",
    "supported_claims",
    "attributed_claims",
]

FUNNEL_KEYS = [
    "cases",
    "claims",
    "claims_supported",
    "claims_attributed",
    "claims_unsupported",
    "claims_supported_but_uncited",
]

CLAIM_ROW_KEYS = [
    "query_id",
    "kind",
    "claim_id",
    "claim",
    "sentence_index",
    "supported",
    "attributed",
    "supporting_citation_ids",
    "support_score",
    "citation_source",
    "answer_source",
]


def _evidence():
    return json.loads(EVIDENCE_PATH.read_text(encoding="utf-8"))


def _claims_of(case_id):
    return [row for row in _evidence()["per_claim"] if row["query_id"] == case_id]


# --- A. Frozen inputs ----------------------------------------------------------


@pytest.mark.parametrize("relative_path", sorted(FROZEN_SHA256))
def test_frozen_inputs_are_unchanged(relative_path):
    assert frozen_digest_matches(
        ROOT / relative_path, FROZEN_SHA256[relative_path]
    ), (
        f"{relative_path} changed -- corpus, retrieval dataset and retrieval "
        "evidence must not be modified by this profile"
    )


def test_rerun_writes_only_to_the_given_path(tmp_path):
    attribution_profile.main(tmp_path / "fresh.json")
    assert frozen_digest_matches(
        ROOT / "benchmark/results/retrieval_small_baseline.json",
        FROZEN_SHA256["benchmark/results/retrieval_small_baseline.json"],
    )


# --- B. Evidence schema --------------------------------------------------------


def test_evidence_schema_is_fixed():
    evidence = _evidence()
    assert list(evidence) == TOP_KEYS
    assert evidence["experiment"] == attribution_profile.EXPERIMENT_ID
    assert evidence["dataset"] == "benchmark/datasets/answer_small.json"
    assert evidence["corpus_documents"] == 6
    assert evidence["corpus_chunks"] == 33
    assert evidence["extractor"] == "DeterministicClaimExtractor"
    assert evidence["attributor"] == "DeterministicClaimAttributor"
    assert evidence["attribution_threshold"] == 0.5
    assert evidence["deterministic"] is True
    assert list(evidence["aggregate"]) == AGGREGATE_KEYS
    assert list(evidence["funnel"]) == FUNNEL_KEYS


def test_aggregate_agrees_with_the_per_claim_rows():
    evidence = _evidence()
    rows = evidence["per_claim"]
    aggregate = evidence["aggregate"]
    total = len(rows)
    supported = sum(1 for row in rows if row["supported"])
    attributed = sum(1 for row in rows if row["attributed"])
    assert aggregate["total_claims"] == total
    assert aggregate["supported_claims"] == supported
    assert aggregate["attributed_claims"] == attributed
    assert aggregate["supported_claim_rate"] == pytest.approx(supported / total)
    assert aggregate["attributed_claim_rate"] == pytest.approx(attributed / total)
    assert aggregate["mean_support_score"] == pytest.approx(
        sum(row["support_score"] for row in rows) / total
    )


def test_funnel_arithmetic_holds():
    funnel = _evidence()["funnel"]
    assert funnel["claims"] == funnel["claims_supported"] + funnel["claims_unsupported"]
    assert (
        funnel["claims_supported_but_uncited"]
        == funnel["claims_supported"] - funnel["claims_attributed"]
    )
    assert funnel["cases"] == len({row["query_id"] for row in _evidence()["per_claim"]})


def test_every_claim_row_is_complete_and_in_range():
    rows = _evidence()["per_claim"]
    assert rows
    for row in rows:
        assert list(row) == CLAIM_ROW_KEYS
        assert 0.0 <= row["support_score"] <= 1.0
        assert isinstance(row["supported"], bool)
        assert isinstance(row["attributed"], bool)
        assert isinstance(row["supporting_citation_ids"], list)
        assert row["attributed"] == bool(row["supporting_citation_ids"])
        assert row["claim"]
        assert row["citation_source"] in ("context", "pinned")
        assert row["answer_source"] in ("pipeline", "override")


def test_every_dataset_case_contributes_claims():
    dataset = json.loads(
        (BENCH_DIR / "datasets" / "answer_small.json").read_text(encoding="utf-8")
    )
    evidence = _evidence()
    assert evidence["total_cases"] == len(dataset["cases"])
    covered = {row["query_id"] for row in evidence["per_claim"]}
    assert covered == {case["id"] for case in dataset["cases"]}
    for case in dataset["cases"]:
        claims = _claims_of(case["id"])
        assert [row["claim_id"] for row in claims] == [
            f"claim-{index}" for index in range(len(claims))
        ]


# --- C. The separation the layer exists for ------------------------------------


def test_supported_and_attributed_are_recorded_separately():
    # The uncited probe: supported by the context, no citation at all.
    claim = _claims_of("n04_correct_without_sources")[0]
    assert claim["supported"] is True
    assert claim["attributed"] is False
    assert claim["supporting_citation_ids"] == []


def test_wrong_citation_is_supported_but_unattributed():
    claim = _claims_of("p01_correct_wrong_citation")[0]
    assert claim["supported"] is True
    assert claim["attributed"] is False
    assert claim["citation_source"] == "pinned"


def test_claim_covered_by_two_citations_lists_both_in_order():
    claims = _claims_of("p03_claim_with_two_sources")
    assert len(claims) == 1
    assert claims[0]["supporting_citation_ids"] == ["doc_b::1", "doc_b::2"]


def test_supported_but_uncited_claims_exist_in_the_funnel():
    # The middle cell of the two-rate story is counted, not folded away.
    funnel = _evidence()["funnel"]
    assert funnel["claims_supported_but_uncited"] >= 2
    second = _claims_of("p02_claim_supported_but_uncited")[1]
    assert second["supported"] is True
    assert second["attributed"] is False


def test_unsupported_claim_is_recorded():
    claim = _claims_of("n02_correct_but_ungrounded")[0]
    assert claim["supported"] is False
    assert claim["attributed"] is False
    assert claim["support_score"] == 0.0


def test_negation_probe_is_kept_as_limitation_evidence():
    # One flipped word: supported AND attributed to the source it contradicts.
    claim = _claims_of("n05_negation_grounded")[0]
    assert "never" in claim["claim"]
    assert claim["supported"] is True
    assert claim["attributed"] is True
    assert claim["support_score"] < 1.0  # "never" is the only miss


def test_attributed_never_exceeds_supported_in_the_evidence():
    rows = _evidence()["per_claim"]
    attributed_but_unsupported = [
        row for row in rows if row["attributed"] and not row["supported"]
    ]
    # A citation's text is a slice of the context it was projected from, so a
    # citation cannot support a claim the context does not.
    assert attributed_but_unsupported == []


# --- D. Reproducibility and hygiene --------------------------------------------


def test_two_independent_runs_are_byte_identical(tmp_path):
    first = tmp_path / "one.json"
    second = tmp_path / "two.json"
    attribution_profile.main(first)
    attribution_profile.main(second)
    assert normalized_lf_bytes(first) == normalized_lf_bytes(second)
    assert normalized_lf_bytes(first) == normalized_lf_bytes(EVIDENCE_PATH)


def test_evidence_has_no_leakage():
    raw = EVIDENCE_PATH.read_text(encoding="utf-8")
    for leaked in ("timestamp", "hostname", "username", "datetime", "uuid", "api_key"):
        assert leaked not in raw.lower()
    assert "\\" not in raw  # no Windows absolute path
    assert "://" not in raw  # no URL
    assert ".cache" not in raw
    assert "sk-" not in raw


def test_runner_uses_no_network_no_model_and_no_metric_of_its_own():
    source = RUNNER_SCRIPT.read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported = {
        alias.name.split(".")[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    } | {
        (node.module or "").split(".")[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and not node.level
    }
    # stdlib + the library + the two sibling harnesses it reuses.
    assert imported <= {
        "__future__",
        "json",
        "sys",
        "pathlib",
        "zmwiki",
        "run_answer_evaluation",
        "run_retrieval_small_baseline",
    }
    for banned in ("requests", "urllib", "httpx", "socket", "sentence_transformers", "os"):
        assert banned not in imported
    # The rates come from zmwiki.attribution; the runner only pools them.
    assert "DeterministicClaimAttributor" in source
    for banned in ("def _ndcg", "def _coverage", "math."):
        assert banned not in source


def test_attribution_is_not_wired_into_the_pipeline_or_the_generator():
    # Prose may mention attribution (the citation stage is named after it); the
    # check is that neither module *depends* on the claim-attribution layer.
    for relative in ("src/zmwiki/pipeline.py", "src/zmwiki/generators.py"):
        source = (ROOT / relative).read_text(encoding="utf-8")
        imports = [
            line
            for line in source.splitlines()
            if line.startswith(("import ", "from "))
        ]
        assert not any("attribution" in line for line in imports), (relative, imports)
