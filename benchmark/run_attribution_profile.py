"""Claim attribution profile: answer -> claims -> supported / attributed.

The claim-level counterpart of the answer evaluation profile, over the *same*
dataset, corpus and pipeline:

    answer_small.json
       ↓  per case: Pipeline.answer_with_sources / query  (reused from the
       ↓             answer evaluation harness -- one wiring, not two)
       ↓  DeterministicClaimExtractor.extract(answer)      -> Claim[]
       ↓  DeterministicClaimAttributor.attribute(claims, context, citations)
       ↓  AttributionResult[]                              -> per-claim evidence
    aggregate + funnel

Two rates are reported and never merged: **supported** (the context covers the
claim) and **attributed** (a cited source covers it). The interesting cell is
the one between them -- claims the context supports but no citation covers.

Offline, dependency-free, model-free, and deterministic: the runner executes the
whole profile twice and refuses to write evidence unless the two runs are
identical. It reads no key, opens no socket, and computes no metric of its own
-- every number comes from ``zmrag.attribution``.

Run from the repository root:

    python benchmark/run_attribution_profile.py

An optional positional output path overrides the default evidence location (so
tests never overwrite the committed evidence file).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Bootstrapping for the uninstalled src/ layout (mirrors the pyproject
# ``pythonpath = ["src"]`` pytest setting), so the script runs as-is from the
# repo root without ``pip install``. Stdlib only.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from zmrag import DeterministicClaimAttributor, DeterministicClaimExtractor

#: Sibling harnesses: the answer profile owns the dataset loading, the canonical
#: pipeline wiring, and the citation pinning; the retrieval runner owns the
#: corpus. Both are reused as-is -- never modified, never re-implemented.
import run_answer_evaluation as _answer_profile
import run_retrieval_small_baseline as _baseline

#: Repo root, resolved from this file so the script runs from any cwd.
ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EVIDENCE_PATH = ROOT / "benchmark" / "results" / "answer_attribution_profile.json"

#: Stable identifiers used in the evidence (never an absolute path).
DATASET_ID = _answer_profile.DATASET_ID
EXPERIMENT_ID = "answer_attribution_profile"
EXTRACTOR_ID = "DeterministicClaimExtractor"
ATTRIBUTOR_ID = "DeterministicClaimAttributor"
TOKENIZER_ID = _answer_profile.TOKENIZER_ID

#: The aggregate keys, in their fixed order.
AGGREGATE_FIELDS = (
    "supported_claim_rate",
    "attributed_claim_rate",
    "mean_support_score",
    "total_claims",
    "supported_claims",
    "attributed_claims",
)

#: The funnel keys, in their fixed order: where claims are lost on the way from
#: "the answer said it" to "a cited source carries it".
FUNNEL_FIELDS = (
    "cases",
    "claims",
    "claims_supported",
    "claims_attributed",
    "claims_unsupported",
    "claims_supported_but_uncited",
)

#: Each per-claim row's fixed keys, in their fixed order.
CLAIM_ROW_FIELDS = (
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
)


def _aggregate(results) -> dict:
    """Pooled claim-level aggregates over every case.

    Pooled, not a mean of per-case means: a claim is the unit here, so an answer
    with three claims weighs three times one with a single claim. Every rate is
    0.0 when there are no claims at all.
    """
    results = list(results)
    total = sum(result.total_claims for result in results)
    supported = sum(result.supported_claims for result in results)
    attributed = sum(result.attributed_claims for result in results)
    score_sum = sum(
        claim.support_score for result in results for claim in result.claims
    )
    return {
        "supported_claim_rate": supported / total if total else 0.0,
        "attributed_claim_rate": attributed / total if total else 0.0,
        "mean_support_score": score_sum / total if total else 0.0,
        "total_claims": total,
        "supported_claims": supported,
        "attributed_claims": attributed,
    }


def _funnel(rows: list[dict]) -> dict:
    """Claim counts along the path from a produced answer to a cited claim."""
    total = len(rows)
    supported = sum(1 for row in rows if row["supported"])
    attributed = sum(1 for row in rows if row["attributed"])
    return {
        "cases": len({row["query_id"] for row in rows}),
        "claims": total,
        "claims_supported": supported,
        "claims_attributed": attributed,
        "claims_unsupported": total - supported,
        "claims_supported_but_uncited": supported - attributed,
    }


def _run_profile() -> dict:
    """Run every case through a freshly built pipeline; nothing is cached."""
    documents = _baseline.ingest_corpus()
    _baseline.verify_corpus(documents)
    cases = _answer_profile._load_cases()
    pipeline = _answer_profile.build_pipeline(documents)

    extractor = DeterministicClaimExtractor()
    attributor = DeterministicClaimAttributor()
    kwargs = {
        "top_k": _answer_profile.TOP_K,
        "candidate_k": _answer_profile.CANDIDATE_K,
        "max_chars": _answer_profile.MAX_CHARS,
    }

    rows: list[dict] = []
    results = []
    for case in cases:
        query = case["query"]
        answer_result = pipeline.answer_with_sources(query, **kwargs)
        context = pipeline.query(query, **kwargs)
        override = case.get("answer_override")
        answer = override if override is not None else answer_result.answer
        sources, citation_source = _answer_profile._resolve_sources(
            case, context, answer_result
        )

        claims = extractor.extract(answer)
        result = attributor.attribute(claims, context=context, citations=sources)
        results.append(result)

        by_id = {claim.claim_id: claim for claim in claims}
        for attribution in result.claims:
            claim = by_id[attribution.claim_id]
            rows.append(
                {
                    "query_id": case["id"],
                    "kind": case["kind"],
                    "claim_id": attribution.claim_id,
                    "claim": claim.text,
                    "sentence_index": claim.sentence_index,
                    "supported": attribution.supported,
                    "attributed": attribution.attributed,
                    "supporting_citation_ids": list(attribution.supporting_citation_ids),
                    "support_score": attribution.support_score,
                    "citation_source": citation_source,
                    "answer_source": "override" if override is not None else "pipeline",
                }
            )

    aggregate = _aggregate(results)
    if list(aggregate) != list(AGGREGATE_FIELDS):
        raise RuntimeError("aggregate keys drifted from the locked field list")

    return {
        "experiment": EXPERIMENT_ID,
        "dataset": DATASET_ID,
        "corpus_documents": _answer_profile.EXPECTED_DOCUMENTS,
        "corpus_chunks": _answer_profile.EXPECTED_TOTAL_CHUNKS,
        "chunking": f"FixedSizeChunker(chunk_size={_answer_profile.CHUNK_SIZE}, "
        f"overlap={_answer_profile.OVERLAP})",
        "embedding": _answer_profile.EMBEDDING_ID,
        "generator": _answer_profile.GENERATOR_ID,
        "extractor": EXTRACTOR_ID,
        "attributor": ATTRIBUTOR_ID,
        "tokenizer": TOKENIZER_ID,
        "attribution_threshold": attributor.threshold,
        "top_k": _answer_profile.TOP_K,
        "candidate_k": _answer_profile.CANDIDATE_K,
        "total_cases": len(cases),
        "aggregate": aggregate,
        "funnel": _funnel(rows),
        "per_claim": rows,
        "deterministic": True,
    }


def main(out_path: str | Path = DEFAULT_EVIDENCE_PATH) -> dict:
    """Run the profile twice on fresh pipelines and write the evidence."""
    output = Path(out_path)
    first = _run_profile()
    second = _run_profile()
    if first != second:
        raise RuntimeError(
            "attribution profile is not deterministic: two fresh runs differ"
        )
    payload = json.dumps(first, indent=2) + "\n"
    if json.dumps(second, indent=2) + "\n" != payload:  # pragma: no cover - guard
        raise RuntimeError("attribution profile is not byte-identical across runs")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(payload, encoding="utf-8")
    return first


if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_EVIDENCE_PATH
    evidence = main(target)
    print(f"Evidence written to: {target}")
    print(json.dumps(evidence, indent=2))
