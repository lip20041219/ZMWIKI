"""Temporary M4.4 diagnostic: run only the two protocol-error cases.

Reproduces the historical protocol errors for:
  - mh02_project_budget
  - mh10_direct_reports

All evaluation logic is reused from run_multihop_evaluation.py (dataset
loading, WikiIndex building, run_experiment, serialize_results). This script
only filters the two cases and picks fresh output paths.

The API key is read from ZMWIKI_API_KEY only — never from the command line
and never printed.

Usage:
    python benchmark/tmp_m4_4_diagnose_cases.py \
        --base-url https://api.example.com/v1 --model MODEL

Refuses to overwrite any existing output or checkpoint file. Exits non-zero
unless every case ends with AgentStatus.ANSWERED.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

_BENCHMARK_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_BENCHMARK_DIR))

import run_multihop_evaluation as m4  # noqa: E402

CASE_IDS = ("mh02_project_budget", "mh10_direct_reports")
DEFAULT_OUTPUT = m4.ROOT / "benchmark" / "results" / "m4_4_diagnosis_mh02_mh10.json"


def main() -> None:
    parser = argparse.ArgumentParser(description="M4.4 two-case protocol-error diagnostic")
    parser.add_argument("--base-url", required=True, help="OpenAI-compatible API base URL")
    parser.add_argument("--model", required=True, help="Model identifier")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT),
                        help="New output JSON path (must not exist)")
    args = parser.parse_args()

    api_key = os.environ.get("ZMWIKI_API_KEY", "").strip()
    if not api_key:
        parser.error("ZMWIKI_API_KEY environment variable is required "
                     "(keys are not accepted on the command line).")

    output_path = Path(args.output)
    checkpoint_path = output_path.with_name(output_path.stem + ".checkpoint.jsonl")
    for path in (output_path, checkpoint_path):
        if path.exists():
            parser.error(f"Refusing to overwrite existing file: {path}")

    # Filter the dataset to exactly the two requested cases.
    dataset = m4.load_dataset()
    cases = [c for c in dataset["cases"] if c["id"] in CASE_IDS]
    found = {c["id"] for c in cases}
    missing = [case_id for case_id in CASE_IDS if case_id not in found]
    if missing:
        raise SystemExit(f"Requested cases not found in dataset: {missing}")
    dataset = {**dataset, "cases": cases}
    print(f"Selected cases ({len(cases)}): {[c['id'] for c in cases]}")

    documents = m4.load_documents()
    pages = m4.build_wiki_pages(documents)
    wiki_index = m4.build_wiki_index(pages)

    # Reuse the deterministic offline baseline for the same two cases.
    baseline_results = m4.run_baseline(cases, m4.build_baseline_pipeline(documents))

    experiment_results = m4.run_experiment(
        cases,
        wiki_index,
        base_url=args.base_url,
        model=args.model,
        api_key=api_key,
        max_steps=m4.AGENT_MAX_STEPS,
        link_source="manual",
        checkpoint_path=checkpoint_path,
    )

    for result in experiment_results:
        print(f"\n== {result.id} status={result.status} error={result.error!r}")
        for detail in result.step_details:
            print(f"  step {detail['step']} {detail['action']}: "
                  f"{detail['result_summary']} error={detail['error']!r}")

    m4.serialize_results(
        dataset,
        baseline_results,
        experiment_results,
        live=True,
        model=args.model,
        output_path=output_path,
        link_source="manual",
        base_url=args.base_url,
    )
    print(f"\nResults written to {output_path}")

    failed = [r.id for r in experiment_results if r.status != m4.AgentStatus.ANSWERED]
    if failed:
        raise SystemExit(f"FAILED: cases did not answer successfully: {failed}")


if __name__ == "__main__":
    main()
