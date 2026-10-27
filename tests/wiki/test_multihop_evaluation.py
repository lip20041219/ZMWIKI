"""Offline tests for the multi-hop evaluation infrastructure.

These tests verify data loading, metric computation, answer scoring,
leakage detection, result serialization, and failure handling. They do
NOT run a real LLM — the WikiAgent experiment path is validated via
infrastructure checks only.
"""

import json
import sys
from pathlib import Path

import pytest

# Ensure benchmark/ is importable (the script uses sys.path insertion for src/,
# but tests need both).
_BENCHMARK_DIR = Path(__file__).resolve().parents[2] / "benchmark"
sys.path.insert(0, str(_BENCHMARK_DIR))

import run_multihop_evaluation as m4  # noqa: E402
from zmwiki.wiki.wiki_index import WikiIndex  # noqa: E402
from zmwiki.wiki.wiki_schema import WikiPage  # noqa: E402


# ---------------------------------------------------------------------------
# Dataset loading & validation
# ---------------------------------------------------------------------------


class TestDatasetLoading:
    def test_loads_26_cases(self):
        dataset = m4.load_dataset()
        assert len(dataset["cases"]) == 26

    def test_all_cases_have_required_fields(self):
        dataset = m4.load_dataset()
        required = ("id", "question", "reference_answer", "answerable",
                     "required_page_ids", "hops", "question_type")
        for case in dataset["cases"]:
            for field in required:
                assert field in case, f"case {case.get('id')} missing {field}"

    def test_case_ids_unique(self):
        dataset = m4.load_dataset()
        ids = [c["id"] for c in dataset["cases"]]
        assert len(ids) == len(set(ids))

    def test_has_unanswerable_case(self):
        dataset = m4.load_dataset()
        unanswerable = [c for c in dataset["cases"] if not c["answerable"]]
        assert len(unanswerable) >= 1

    def test_has_multi_hop_cases(self):
        dataset = m4.load_dataset()
        multi = [c for c in dataset["cases"] if c["hops"] >= 2]
        assert len(multi) >= 5

    def test_answerable_cases_have_keyword_groups(self):
        dataset = m4.load_dataset()
        for case in dataset["cases"]:
            if case["answerable"]:
                assert case["answer_keyword_groups"], (
                    f"answerable case {case['id']} has no keyword groups"
                )


# ---------------------------------------------------------------------------
# Corpus & WikiIndex building
# ---------------------------------------------------------------------------


class TestCorpusAndIndex:
    def test_loads_9_documents(self):
        docs = m4.load_documents()
        assert len(docs) == 9
        for doc in docs:
            assert doc.id
            assert doc.content.strip()

    def test_document_ids_match_page_ids(self):
        docs = m4.load_documents()
        doc_ids = {d.id for d in docs}
        page_ids = set(m4._PAGE_LINKS.keys())
        assert doc_ids == page_ids

    def test_build_wiki_pages(self):
        docs = m4.load_documents()
        pages = m4.build_wiki_pages(docs)
        assert len(pages) == 9
        for page in pages:
            assert isinstance(page, WikiPage)
            assert page.id
            assert page.path
            assert page.title
            assert page.content

    def test_wiki_index_links_clean(self):
        docs = m4.load_documents()
        pages = m4.build_wiki_pages(docs)
        idx = m4.build_wiki_index(pages)
        assert idx.count() == 9
        report = idx.check_links()
        assert report.is_clean, f"broken={report.broken_links} inconsistent={report.inconsistent_links}"

    def test_bidirectional_links_consistent(self):
        """If A links_to B, then B should links_from A."""
        docs = m4.load_documents()
        pages = {p.id: p for p in m4.build_wiki_pages(docs)}
        for page in pages.values():
            for target_id in page.links_to:
                assert target_id in pages, f"page {page.id} links to missing {target_id}"
                assert page.id in pages[target_id].links_from, (
                    f"page {page.id} links_to {target_id} but {target_id} missing links_from {page.id}"
                )


# ---------------------------------------------------------------------------
# Answer scoring
# ---------------------------------------------------------------------------


class TestAnswerScoring:
    def test_correct_answer_matches_all_keyword_groups(self):
        case = {
            "answerable": True,
            "answer_keyword_groups": [["director"], ["engineering"]],
        }
        assert m4.score_answer("Bob Jones is the Director of Engineering.", case) is True

    def test_partial_match_is_incorrect(self):
        case = {
            "answerable": True,
            "answer_keyword_groups": [["director"], ["engineering"]],
        }
        assert m4.score_answer("Bob Jones is a Director.", case) is False

    def test_alternative_keyword_in_group(self):
        case = {
            "answerable": True,
            "answer_keyword_groups": [["500", "five"], ["thousand"]],
        }
        assert m4.score_answer("The budget is five hundred thousand dollars.", case) is True
        assert m4.score_answer("The budget is 500 thousand dollars.", case) is True
        # "$500,000" does not contain the word "thousand", so it does not match
        # the second keyword group. This is intentional: the scorer is lexical,
        # not semantic.
        assert m4.score_answer("The budget is $500,000.", case) is False

    def test_empty_answer_is_incorrect(self):
        case = {"answerable": True, "answer_keyword_groups": [["test"]]}
        assert m4.score_answer("", case) is False
        assert m4.score_answer(None, case) is False

    def test_unanswerable_refusal_detected(self):
        case = {
            "answerable": False,
            "refusal_keywords": ["cannot", "not mentioned", "unknown"],
        }
        assert m4.score_answer("I cannot find this information in the wiki.", case) is True
        assert m4.score_answer("The CEO is John Doe.", case) is False

    def test_case_insensitive_matching(self):
        case = {"answerable": True, "answer_keyword_groups": [["DIRECTOR"]]}
        assert m4.score_answer("the director", case) is True

    def test_punctuation_ignored(self):
        case = {"answerable": True, "answer_keyword_groups": [["500"]]}
        assert m4.score_answer("$500,000.00", case) is True


# ---------------------------------------------------------------------------
# M4.3.1: false-negative fixes and legal alternative evidence paths
# ---------------------------------------------------------------------------


class TestM431ScoringFixes:
    @staticmethod
    def _case(case_id: str) -> dict:
        dataset = m4.load_dataset()
        return next(c for c in dataset["cases"] if c["id"] == case_id)

    def test_pronoun_subject_matches_mh20(self):
        case = self._case("mh20_phoenix_deadline_office")
        answer = (
            "Project Phoenix has a target completion date of December 31, 2026. "
            "It is led by Alice Smith. Alice Smith's office is on the second "
            "floor of Building A."
        )
        assert m4.score_answer(answer, case) is True

    def test_clean_refusals_still_score(self):
        answers = {
            "mh08_ceo_unanswerable":
                "I could not confirm the CEO; no page names the CEO.",
            "mh28_first_floor_office_premise":
                "The first floor of Building B does not contain an office; it "
                "is a gym. There is no such person or project.",
            "mh29_alice_salary":
                "The corpus does not contain any information about Alice "
                "Smith's salary; no compensation details are provided.",
            "mh30_david_university":
                "The page does not state which university David Kim attended; "
                "I could not confirm it.",
        }
        for case_id, answer in answers.items():
            assert m4.score_answer(answer, self._case(case_id)) is True, case_id

    def test_refusal_plus_fabrication_rejected(self):
        answers = {
            "mh08_ceo_unanswerable":
                "I could not confirm the CEO; the CEO is Bob Jones.",
            "mh28_first_floor_office_premise":
                "I cannot find the first-floor office; maybe it belongs to "
                "David Kim, who leads Project Orion.",
            "mh29_alice_salary":
                "Alice Smith's salary is not stated; it is probably 150,000 "
                "dollars.",
            "mh30_david_university":
                "David Kim attended MIT; the wiki is unclear.",
        }
        for case_id, answer in answers.items():
            assert m4.score_answer(answer, self._case(case_id)) is False, case_id

    def test_alternative_evidence_path_satisfies_recall(self):
        case = self._case("mh21_team_intersection")
        found = {"bob", "project_phoenix", "project_orion"}
        recall, precision, numerator, denominator = m4._compute_evidence_metrics(
            found, case["required_page_ids"], sorted(found),
            case["alternative_evidence_sets"],
        )
        assert recall == 1.0
        assert precision == 1.0
        assert (numerator, denominator) == (3, 3)

    def test_partial_alternative_path_not_full_recall(self):
        case = self._case("mh21_team_intersection")
        recall, _, _ = m4._compute_recall(
            {"bob"}, case["required_page_ids"],
            case["alternative_evidence_sets"],
        )
        assert recall is not None and recall < 1.0

    def test_hops_matches_primary_required_pages(self):
        dataset = m4.load_dataset()
        for case in dataset["cases"]:
            assert case["hops"] == len(case["required_page_ids"]), case["id"]

    def test_every_reference_answer_scores(self):
        dataset = m4.load_dataset()
        for case in dataset["cases"]:
            assert m4.score_answer(case["reference_answer"], case), case["id"]

    def test_alternative_evidence_sets_use_known_pages(self):
        dataset = m4.load_dataset()
        for case in dataset["cases"]:
            for path in case.get("alternative_evidence_sets", []):
                assert path
                assert set(path) <= set(m4._PAGE_LINKS), case["id"]


# ---------------------------------------------------------------------------
# Metric computation
# ---------------------------------------------------------------------------


class TestMetrics:
    def _make_result(self, case_id, accessed, required, sources, correct=True,
                      status="answered", steps=1, pages_read=1):
        return m4.CaseResult(
            id=case_id, question="q", answer="a" if correct else "",
            answer_correct=correct, status=status, sources=sources,
            accessed_pages=accessed, required_pages=required,
            evidence_recall=(len(set(accessed) & set(required)) / len(required)
                              if required else None),
            evidence_recall_numerator=len(set(accessed) & set(required)) if required else 0,
            evidence_recall_denominator=len(required),
            source_precision=len(set(sources) & set(required)) / len(sources) if sources else 0.0,
            steps=steps, pages_read=pages_read,
            answerable=bool(required),
        )

    def test_evidence_recall_all_found(self):
        recall, _, num, den = m4._compute_evidence_metrics({"a", "b"}, ["a", "b"], ["a"])
        assert recall == 1.0
        assert num == 2
        assert den == 2

    def test_evidence_recall_partial(self):
        recall, _, num, den = m4._compute_evidence_metrics({"a"}, ["a", "b"], ["a"])
        assert recall == 0.5
        assert num == 1
        assert den == 2

    def test_evidence_recall_empty_required_is_none(self):
        """Unanswerable cases (required empty) get evidence_recall=None, NOT 1.0."""
        recall, precision, num, den = m4._compute_evidence_metrics(set(), [], [])
        assert recall is None
        assert precision == 0.0
        assert num == 0
        assert den == 0

    def test_source_precision(self):
        _, precision, _, _ = m4._compute_evidence_metrics({"a", "b", "c"}, ["a", "b"], ["a", "c"])
        assert precision == 0.5  # 1 of 2 sources is required

    def test_source_precision_empty_sources(self):
        _, precision, _, _ = m4._compute_evidence_metrics({"a"}, ["a"], [])
        assert precision == 0.0

    def test_aggregate_counts(self):
        cases = [
            {"id": "c1", "answerable": True, "hops": 1},
            {"id": "c2", "answerable": True, "hops": 2},
            {"id": "c3", "answerable": False, "hops": 0},
        ]
        results = [
            self._make_result("c1", ["a"], ["a"], ["a"], correct=True),
            self._make_result("c2", ["a"], ["a", "b"], ["a"], correct=False),
            self._make_result("c3", [], [], [], correct=True, status="answered"),
        ]
        agg = m4.aggregate_results(cases, results)
        assert agg["total_cases"] == 3
        assert agg["answerable_cases"] == 2
        assert agg["unanswerable_cases"] == 1
        assert agg["correct_count"] == 2
        assert agg["answer_accuracy"] == pytest.approx(2 / 3)
        assert agg["multi_hop_cases"] == 1
        assert agg["multi_hop_success_rate"] == 0.0  # c2 not correct with full recall

    def test_failure_rate(self):
        cases = [{"id": "c1", "answerable": True, "hops": 1}]
        results = [
            self._make_result("c1", [], ["a"], [], correct=False,
                              status="model_error", steps=0, pages_read=0),
        ]
        agg = m4.aggregate_results(cases, results)
        assert agg["failure_rate"] == 1.0


# ---------------------------------------------------------------------------
# Answer leakage check
# ---------------------------------------------------------------------------


class TestLeakageCheck:
    def test_no_leakage_in_real_dataset(self):
        dataset = m4.load_dataset()
        docs = m4.load_documents()
        pages = m4.build_wiki_pages(docs)
        warnings = m4.check_answer_leakage(dataset, pages)
        assert warnings == [], f"leakage detected: {warnings}"

    def test_detects_verbatim_leak(self):
        dataset = {
            "cases": [{
                "id": "leak",
                "reference_answer": "The secret answer is 42",
                "answerable": True,
                "answer_keyword_groups": [["42"]],
                "required_page_ids": ["p1"],
                "hops": 1,
                "question_type": "test",
                "question": "What is the answer?",
            }]
        }
        pages = [WikiPage(
            id="p1", path="wiki/p1", title="Page",
            content="The secret answer is 42 and more text.",
        )]
        warnings = m4.check_answer_leakage(dataset, pages)
        assert len(warnings) == 1
        assert "leak" in warnings[0]


# ---------------------------------------------------------------------------
# Baseline pipeline integration
# ---------------------------------------------------------------------------


class TestBaselineIntegration:
    def test_baseline_runs_all_cases(self):
        dataset = m4.load_dataset()
        docs = m4.load_documents()
        pipeline = m4.build_baseline_pipeline(docs)
        results = m4.run_baseline(dataset["cases"], pipeline)
        assert len(results) == 26
        for r in results:
            assert r.id
            assert r.status in ("answered", "no_answer", "error")
            assert isinstance(r.answer_correct, bool)

    def test_baseline_evidence_recall_high(self):
        """The baseline should retrieve most relevant documents."""
        dataset = m4.load_dataset()
        docs = m4.load_documents()
        pipeline = m4.build_baseline_pipeline(docs)
        results = m4.run_baseline(dataset["cases"], pipeline)
        answerable = [r for r, c in zip(results, dataset["cases"]) if c["answerable"]]
        mean_recall = sum(r.evidence_recall for r in answerable) / len(answerable)
        assert mean_recall >= 0.5, f"baseline recall too low: {mean_recall}"

    def test_baseline_does_not_crash_on_unanswerable(self):
        dataset = m4.load_dataset()
        docs = m4.load_documents()
        pipeline = m4.build_baseline_pipeline(docs)
        results = m4.run_baseline(dataset["cases"], pipeline)
        unanswerable = [r for r, c in zip(results, dataset["cases"]) if not c["answerable"]]
        assert len(unanswerable) >= 1
        for r in unanswerable:
            assert r.error is None  # should not crash


# ---------------------------------------------------------------------------
# MonitoringWikiIndex
# ---------------------------------------------------------------------------


class TestMonitoringWikiIndex:
    def test_tracks_get_accesses(self):
        idx = WikiIndex()
        idx.upsert(WikiPage(id="p1", path="wiki/p1", title="P1", content="c1"))
        monitor = m4.MonitoringWikiIndex(idx)
        assert monitor.get("p1") is not None
        assert "p1" in monitor.accessed_page_ids

    def test_tracks_get_by_path_accesses(self):
        idx = WikiIndex()
        idx.upsert(WikiPage(id="p1", path="wiki/p1", title="P1", content="c1"))
        monitor = m4.MonitoringWikiIndex(idx)
        assert monitor.get_by_path("wiki/p1") is not None
        assert "p1" in monitor.accessed_page_ids

    def test_missing_access_not_recorded(self):
        idx = WikiIndex()
        monitor = m4.MonitoringWikiIndex(idx)
        assert monitor.get("missing") is None
        assert monitor.accessed_page_ids == set()

    def test_delegates_other_methods(self):
        idx = WikiIndex()
        idx.upsert(WikiPage(id="p1", path="wiki/p1", title="P1", content="c1"))
        monitor = m4.MonitoringWikiIndex(idx)
        assert monitor.count() == 1
        assert monitor.check_links().is_clean


# ---------------------------------------------------------------------------
# Result serialization
# ---------------------------------------------------------------------------


class TestSerialization:
    def test_serializes_baseline_only(self, tmp_path):
        dataset = m4.load_dataset()
        docs = m4.load_documents()
        pipeline = m4.build_baseline_pipeline(docs)
        results = m4.run_baseline(dataset["cases"], pipeline)
        output = tmp_path / "result.json"
        m4.serialize_results(dataset, results, None, live=False, model=None, output_path=output)
        assert output.exists()
        payload = json.loads(output.read_text(encoding="utf-8"))
        assert payload["experiment_id"] == "multihop_wiki_comparison"
        assert "baseline" in payload["results"]
        assert "experiment" in payload["results"]
        assert payload["results"]["experiment"]["aggregate"] is None
        assert payload["config"]["experiment"]["live"] is False

    def test_metadata_contains_git_and_python(self, tmp_path):
        dataset = m4.load_dataset()
        docs = m4.load_documents()
        pipeline = m4.build_baseline_pipeline(docs)
        results = m4.run_baseline(dataset["cases"], pipeline)
        output = tmp_path / "result.json"
        m4.serialize_results(dataset, results, None, live=False, model=None, output_path=output)
        payload = json.loads(output.read_text(encoding="utf-8"))
        assert "git_sha" in payload
        assert "python_version" in payload
        assert "timestamp" in payload
        assert "dataset_sha256_16" in payload
        assert payload["dataset_sha256_16"]  # non-empty

    def test_per_case_results_complete(self, tmp_path):
        dataset = m4.load_dataset()
        docs = m4.load_documents()
        pipeline = m4.build_baseline_pipeline(docs)
        results = m4.run_baseline(dataset["cases"], pipeline)
        output = tmp_path / "result.json"
        m4.serialize_results(dataset, results, None, live=False, model=None, output_path=output)
        payload = json.loads(output.read_text(encoding="utf-8"))
        per_case = payload["results"]["baseline"]["per_case"]
        assert len(per_case) == 26
        for row in per_case:
            for key in ("id", "answer_correct", "status", "evidence_recall",
                        "source_precision", "steps", "pages_read"):
                assert key in row, f"per-case row missing {key}"


# ---------------------------------------------------------------------------
# Failure handling
# ---------------------------------------------------------------------------


class TestFailureHandling:
    def test_failed_case_retained_in_results(self):
        """A case that errors must still appear in per-case results, not dropped."""
        cases = [{"id": "err", "question": "q", "answerable": True,
                  "required_page_ids": ["a"], "hops": 1, "question_type": "t",
                  "reference_answer": "a", "answer_keyword_groups": [["a"]]}]

        class BrokenPipeline:
            def answer_with_sources(self, *a, **kw):
                raise RuntimeError("simulated failure")
            def query(self, *a, **kw):
                raise RuntimeError("simulated failure")

        results = m4.run_baseline(cases, BrokenPipeline())
        assert len(results) == 1  # not dropped
        assert results[0].status == "error"
        assert results[0].error is not None
        assert "simulated failure" in results[0].error

    def test_failed_case_counted_in_aggregate(self):
        cases = [{"id": "err", "answerable": True, "hops": 1}]
        results = [m4.CaseResult(
            id="err", question="q", answer=None, answer_correct=False,
            status="error", sources=[], accessed_pages=[], required_pages=["a"],
            evidence_recall=0.0, source_precision=0.0, steps=0, pages_read=0,
            error="boom",
        )]
        agg = m4.aggregate_results(cases, results)
        assert agg["failure_rate"] == 1.0
        assert agg["correct_count"] == 0


# ---------------------------------------------------------------------------
# M4.2: Evidence recall separation (answerable vs unanswerable)
# ---------------------------------------------------------------------------


class TestEvidenceRecallSeparation:
    """M4.2: Unanswerable cases must NOT inflate mean_evidence_recall."""

    def _make(self, case_id, answerable, required, accessed, correct=True):
        recall = (len(set(accessed) & set(required)) / len(required)
                  if required else None)
        return m4.CaseResult(
            id=case_id, question="q", answer="a" if correct else "",
            answer_correct=correct, status="answered", sources=[],
            accessed_pages=accessed, required_pages=required,
            evidence_recall=recall,
            evidence_recall_numerator=len(set(accessed) & set(required)) if required else 0,
            evidence_recall_denominator=len(required),
            source_precision=0.0, steps=1, pages_read=len(accessed),
            answerable=answerable,
        )

    def test_mean_evidence_recall_excludes_unanswerable(self):
        cases = [
            {"id": "a1", "answerable": True, "hops": 1},
            {"id": "a2", "answerable": True, "hops": 1},
            {"id": "u1", "answerable": False, "hops": 0},
        ]
        results = [
            self._make("a1", True, ["p1"], ["p1"]),       # recall 1.0
            self._make("a2", True, ["p1", "p2"], ["p1"]),  # recall 0.5
            self._make("u1", False, [], []),                 # recall None
        ]
        agg = m4.aggregate_results(cases, results)
        # Mean over answerable only: (1.0 + 0.5) / 2 = 0.75
        assert agg["mean_evidence_recall"] == pytest.approx(0.75)
        re_metrics = agg["retrieval_evidence_metrics"]
        assert re_metrics["mean_evidence_recall_answerable"]["numerator_sum"] == 2
        assert re_metrics["mean_evidence_recall_answerable"]["denominator_sum"] == 3
        assert re_metrics["unanswerable_cases"]["count"] == 1
        assert "excluded" in re_metrics["unanswerable_cases"]["evidence_recall_treatment"]

    def test_all_unanswerable_mean_recall_is_zero(self):
        """If no answerable cases, mean_evidence_recall should be 0.0 (not crash)."""
        cases = [{"id": "u1", "answerable": False, "hops": 0}]
        results = [self._make("u1", False, [], [])]
        agg = m4.aggregate_results(cases, results)
        assert agg["mean_evidence_recall"] == 0.0

    def test_per_case_evidence_recall_none_for_unanswerable(self):
        dataset = m4.load_dataset()
        docs = m4.load_documents()
        pipeline = m4.build_baseline_pipeline(docs)
        results = m4.run_baseline(dataset["cases"], pipeline)
        unanswerable = [r for r, c in zip(results, dataset["cases"]) if not c["answerable"]]
        assert len(unanswerable) == 4
        for result in unanswerable:
            assert result.evidence_recall is None
            assert result.evidence_recall_denominator == 0


# ---------------------------------------------------------------------------
# M4.2: Link resolution diagnostic (uses real resolver)
# ---------------------------------------------------------------------------


class TestLinkResolutionDiagnostic:
    """M4.2: run_link_resolution() must use the real resolver and report metrics."""

    def test_uses_real_resolver_flag(self):
        docs = m4.load_documents()
        pages = m4.build_wiki_pages(docs, use_candidate_links=True)
        report = m4.run_link_resolution(pages)
        assert report["uses_real_resolver"] is True

    def test_resolution_counts_match_real_report(self):
        docs = m4.load_documents()
        pages = m4.build_wiki_pages(docs, use_candidate_links=True)
        report = m4.run_link_resolution(pages)
        # All 9 pages have candidate_links; candidates should resolve by title.
        assert report["pages_processed"] == 9
        assert report["candidates_total"] > 0
        assert report["resolved_count"] > 0
        # No self-links in our candidate data.
        assert report["self_link_skipped"] == 0
        # resolution_rate denominator excludes self-links.
        assert report["resolution_rate_denominator"] == report["candidates_total"]

    def test_resolved_links_have_match_method(self):
        docs = m4.load_documents()
        pages = m4.build_wiki_pages(docs, use_candidate_links=True)
        report = m4.run_link_resolution(pages)
        for rl in report["resolved_links"]:
            assert rl["match_method"] in ("page_id", "path", "title", "alias")
            assert rl["source"] != rl["target"]  # no self-links

    def test_resolved_pages_pass_link_check(self):
        from zmwiki.wiki.wiki_link_resolver import resolve_candidate_links
        docs = m4.load_documents()
        pages = m4.build_wiki_pages(docs, use_candidate_links=True)
        resolved, _ = resolve_candidate_links(pages)
        idx = m4.build_wiki_index(resolved)
        link_report = idx.check_links()
        assert link_report.is_clean


# ---------------------------------------------------------------------------
# M4.2: Manual vs auto link source marking
# ---------------------------------------------------------------------------


class TestLinkSourceMarking:
    """M4.2: manual and auto links must be clearly distinguished, not confused."""

    def test_manual_pages_have_no_candidate_links(self):
        docs = m4.load_documents()
        pages = m4.build_wiki_pages(docs, use_candidate_links=False)
        for p in pages:
            assert "candidate_links" not in p.metadata
            assert len(p.links_to) > 0  # manual links present

    def test_auto_pages_have_candidate_links_and_empty_direct_links(self):
        docs = m4.load_documents()
        pages = m4.build_wiki_pages(docs, use_candidate_links=True)
        for p in pages:
            assert "candidate_links" in p.metadata
            assert len(p.links_to) == 0  # not yet resolved

    def test_baseline_link_source_is_none(self):
        dataset = m4.load_dataset()
        docs = m4.load_documents()
        pipeline = m4.build_baseline_pipeline(docs)
        results = m4.run_baseline(dataset["cases"], pipeline)
        for r in results:
            assert r.link_source == "none"

    def test_serialize_records_link_source(self, tmp_path):
        dataset = m4.load_dataset()
        docs = m4.load_documents()
        pipeline = m4.build_baseline_pipeline(docs)
        results = m4.run_baseline(dataset["cases"], pipeline)
        output = tmp_path / "r.json"
        m4.serialize_results(dataset, results, None, live=False, model=None,
                             output_path=output, link_source="manual")
        payload = json.loads(output.read_text(encoding="utf-8"))
        assert payload["link_source"] == "manual"
        assert payload["config"]["experiment"]["link_source"] == "manual"

    def test_serialize_with_auto_link_resolution(self, tmp_path):
        dataset = m4.load_dataset()
        docs = m4.load_documents()
        pipeline = m4.build_baseline_pipeline(docs)
        results = m4.run_baseline(dataset["cases"], pipeline)
        pages = m4.build_wiki_pages(docs, use_candidate_links=True)
        link_res = m4.run_link_resolution(pages)
        output = tmp_path / "r.json"
        m4.serialize_results(dataset, results, None, live=False, model=None,
                             output_path=output, link_source="auto",
                             link_resolution=link_res)
        payload = json.loads(output.read_text(encoding="utf-8"))
        assert payload["link_source"] == "auto"
        assert "link_resolution" in payload
        assert payload["link_resolution"]["uses_real_resolver"] is True


# ---------------------------------------------------------------------------
# M4.2: Per-case recomputation and comparability notes
# ---------------------------------------------------------------------------


class TestPerCaseRecomputation:
    """M4.2: aggregate metrics must be recomputable from per-case results."""

    def test_answer_accuracy_recomputable(self):
        dataset = m4.load_dataset()
        docs = m4.load_documents()
        pipeline = m4.build_baseline_pipeline(docs)
        results = m4.run_baseline(dataset["cases"], pipeline)
        agg = m4.aggregate_results(dataset["cases"], results)
        # Recompute from per-case.
        recomputed = sum(1 for r in results if r.answer_correct) / len(results)
        assert agg["answer_accuracy"] == pytest.approx(recomputed)

    def test_evidence_recall_recomputable_over_answerable(self):
        dataset = m4.load_dataset()
        docs = m4.load_documents()
        pipeline = m4.build_baseline_pipeline(docs)
        results = m4.run_baseline(dataset["cases"], pipeline)
        agg = m4.aggregate_results(dataset["cases"], results)
        answerable_recalls = [
            r.evidence_recall for r, c in zip(results, dataset["cases"])
            if c["answerable"] and r.evidence_recall is not None
        ]
        recomputed = sum(answerable_recalls) / len(answerable_recalls)
        assert agg["mean_evidence_recall"] == pytest.approx(recomputed)

    def test_comparability_notes_present(self, tmp_path):
        dataset = m4.load_dataset()
        docs = m4.load_documents()
        pipeline = m4.build_baseline_pipeline(docs)
        results = m4.run_baseline(dataset["cases"], pipeline)
        output = tmp_path / "r.json"
        m4.serialize_results(dataset, results, None, live=False, model=None,
                             output_path=output)
        payload = json.loads(output.read_text(encoding="utf-8"))
        assert "comparability_notes" in payload
        notes = payload["comparability_notes"]
        assert "COMPARABLE" in notes["retrieval_evidence_metrics"]
        assert "NOT directly comparable" in notes["answer_quality_metrics"]
        assert "keyword_group_match" in notes["scoring_caveat"]

    def test_aggregate_has_separated_metric_sections(self):
        dataset = m4.load_dataset()
        docs = m4.load_documents()
        pipeline = m4.build_baseline_pipeline(docs)
        results = m4.run_baseline(dataset["cases"], pipeline)
        agg = m4.aggregate_results(dataset["cases"], results)
        assert "retrieval_evidence_metrics" in agg
        assert "answer_quality_metrics" in agg
        assert "efficiency_metrics" in agg
        # Backward-compatible flat keys still present.
        assert "mean_evidence_recall" in agg
        assert "answer_accuracy" in agg


# ---------------------------------------------------------------------------
# M4.2: Failed / missing samples not counted as success
# ---------------------------------------------------------------------------


class TestFailedSamplesNotSuccess:
    """M4.2: errors, missing evidence, and unexecuted cases must not inflate scores."""

    def test_error_case_not_counted_as_correct(self):
        cases = [{"id": "e1", "answerable": True, "hops": 1}]
        results = [m4.CaseResult(
            id="e1", question="q", answer=None, answer_correct=False,
            status="error", sources=[], accessed_pages=[], required_pages=["p1"],
            evidence_recall=0.0, evidence_recall_numerator=0,
            evidence_recall_denominator=1, source_precision=0.0,
            steps=0, pages_read=0, answerable=True, error="boom",
        )]
        agg = m4.aggregate_results(cases, results)
        assert agg["correct_count"] == 0
        assert agg["answer_accuracy"] == 0.0
        assert agg["failure_rate"] == 1.0

    def test_budget_exhausted_not_counted_as_correct(self):
        cases = [{"id": "b1", "answerable": True, "hops": 2}]
        results = [m4.CaseResult(
            id="b1", question="q", answer=None, answer_correct=False,
            status="budget_exhausted", sources=[], accessed_pages=["p1"],
            required_pages=["p1", "p2"], evidence_recall=0.5,
            evidence_recall_numerator=1, evidence_recall_denominator=2,
            source_precision=0.0, steps=8, pages_read=1, answerable=True,
        )]
        agg = m4.aggregate_results(cases, results)
        assert agg["correct_count"] == 0
        assert agg["budget_exhausted_rate"] == 1.0
        assert agg["multi_hop_success_rate"] == 0.0

    def test_unanswerable_refusal_failure_not_counted_in_answerable(self):
        """An unanswerable case that fails to refuse must not affect answerable accuracy."""
        cases = [
            {"id": "a1", "answerable": True, "hops": 1},
            {"id": "u1", "answerable": False, "hops": 0},
        ]
        results = [
            m4.CaseResult(id="a1", question="q", answer="a", answer_correct=True,
                          status="answered", sources=[], accessed_pages=["p1"],
                          required_pages=["p1"], evidence_recall=1.0,
                          evidence_recall_numerator=1, evidence_recall_denominator=1,
                          source_precision=0.0, steps=1, pages_read=1, answerable=True),
            m4.CaseResult(id="u1", question="q", answer="I know the answer",
                          answer_correct=False, status="answered", sources=[],
                          accessed_pages=[], required_pages=[], evidence_recall=None,
                          evidence_recall_numerator=0, evidence_recall_denominator=0,
                          source_precision=0.0, steps=1, pages_read=0, answerable=False),
        ]
        agg = m4.aggregate_results(cases, results)
        # answerable_accuracy should be 1.0 (only a1, which is correct)
        assert agg["answerable_accuracy"] == 1.0
        # overall accuracy is 1/2
        assert agg["answer_accuracy"] == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# M4.3 Phase 2: Evaluation reliability (API key, overwrite, checkpoint, repro)
# ---------------------------------------------------------------------------


class TestApiKeyConfig:
    """M4.3 P2-Task1: API key help text, pre-flight check, and key redaction."""

    def test_help_text_mentions_ZMWIKI_API_KEY(self):
        """The --api-key help must reference ZMWIKI_API_KEY, not OPENAI_API_KEY."""
        import inspect
        source = inspect.getsource(m4.main)
        assert "ZMWIKI_API_KEY" in source
        assert "OPENAI_API_KEY" not in source

    def test_api_key_not_in_serialized_output(self, tmp_path):
        """Serialized results must never contain the API key or Authorization header."""
        dataset = m4.load_dataset()
        docs = m4.load_documents()
        pipeline = m4.build_baseline_pipeline(docs)
        results = m4.run_baseline(dataset["cases"], pipeline)
        output = tmp_path / "r.json"
        m4.serialize_results(dataset, results, None, live=False, model=None,
                             output_path=output, link_source="manual",
                             base_url="https://api.example.com/v1")
        raw = output.read_text(encoding="utf-8")
        # No key-like patterns. We use a fake key value to ensure it's never written.
        assert "sk-secret-test-key-12345" not in raw
        assert "Authorization" not in raw
        assert "api_key" not in raw or '"api_key_recorded": false' in raw

    def test_base_url_recorded_in_live_config(self, tmp_path):
        """When live=True, base_url must be recorded (without key)."""
        dataset = m4.load_dataset()
        docs = m4.load_documents()
        pipeline = m4.build_baseline_pipeline(docs)
        results = m4.run_baseline(dataset["cases"], pipeline)
        output = tmp_path / "r.json"
        m4.serialize_results(dataset, results, None, live=True, model="test-model",
                             output_path=output, link_source="manual",
                             base_url="https://api.example.com/v1")
        payload = json.loads(output.read_text(encoding="utf-8"))
        assert payload["config"]["experiment"]["base_url"] == "https://api.example.com/v1"
        assert payload["config"]["experiment"]["api_key_recorded"] is False

    def test_base_url_none_in_offline_mode(self, tmp_path):
        """When live=False, base_url should be None."""
        dataset = m4.load_dataset()
        docs = m4.load_documents()
        pipeline = m4.build_baseline_pipeline(docs)
        results = m4.run_baseline(dataset["cases"], pipeline)
        output = tmp_path / "r.json"
        m4.serialize_results(dataset, results, None, live=False, model=None,
                             output_path=output)
        payload = json.loads(output.read_text(encoding="utf-8"))
        assert payload["config"]["experiment"]["base_url"] is None


class TestOutputOverwriteProtection:
    """M4.3 P2-Task2: Existing result files must not be silently overwritten."""

    def test_new_file_writes_successfully(self, tmp_path):
        dataset = m4.load_dataset()
        docs = m4.load_documents()
        pipeline = m4.build_baseline_pipeline(docs)
        results = m4.run_baseline(dataset["cases"], pipeline)
        output = tmp_path / "new_result.json"
        m4.serialize_results(dataset, results, None, live=False, model=None,
                             output_path=output)
        assert output.exists()
        payload = json.loads(output.read_text(encoding="utf-8"))
        assert payload["results"]["baseline"]["aggregate"]["total_cases"] == 26

    def test_existing_file_raises_file_exists_error(self, tmp_path):
        dataset = m4.load_dataset()
        docs = m4.load_documents()
        pipeline = m4.build_baseline_pipeline(docs)
        results = m4.run_baseline(dataset["cases"], pipeline)
        output = tmp_path / "existing.json"
        # Write initial content.
        output.write_text('{"original": true}', encoding="utf-8")
        original_content = output.read_text(encoding="utf-8")
        with pytest.raises(FileExistsError, match="already exists"):
            m4.serialize_results(dataset, results, None, live=False, model=None,
                                 output_path=output)
        # Original content must be unchanged.
        assert output.read_text(encoding="utf-8") == original_content


class TestCheckpointPersistence:
    """M4.3 P2-Task3: Per-case incremental checkpoint via JSONL."""

    def _make_case_result(self, case_id: str) -> m4.CaseResult:
        return m4.CaseResult(
            id=case_id, question=f"q-{case_id}", answer="a",
            answer_correct=True, status="answered", sources=[],
            accessed_pages=["p1"], required_pages=["p1"],
            evidence_recall=1.0, evidence_recall_numerator=1,
            evidence_recall_denominator=1, source_precision=0.0,
            steps=1, pages_read=1, answerable=True,
        )

    def test_append_checkpoint_writes_json_line(self, tmp_path):
        cp = tmp_path / "eval.checkpoint.jsonl"
        cr = self._make_case_result("c1")
        m4._append_checkpoint(cp, cr)
        assert cp.exists()
        lines = cp.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 1
        record = json.loads(lines[0])
        assert record["id"] == "c1"
        assert record["answer_correct"] is True

    def test_multiple_cases_appended_incrementally(self, tmp_path):
        cp = tmp_path / "eval.checkpoint.jsonl"
        for cid in ["c1", "c2", "c3"]:
            m4._append_checkpoint(cp, self._make_case_result(cid))
        lines = cp.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 3
        ids = [json.loads(line)["id"] for line in lines]
        assert ids == ["c1", "c2", "c3"]

    def test_checkpoint_separate_from_final_result(self, tmp_path):
        """Checkpoint file must have a distinct name from the final JSON."""
        output = tmp_path / "myrun.json"
        checkpoint = output.with_name(output.stem + ".checkpoint.jsonl")
        assert checkpoint.name == "myrun.checkpoint.jsonl"
        assert checkpoint != output

    def test_run_experiment_accepts_checkpoint_path(self):
        """run_experiment signature must include checkpoint_path parameter."""
        import inspect
        sig = inspect.signature(m4.run_experiment)
        assert "checkpoint_path" in sig.parameters
        assert sig.parameters["checkpoint_path"].default is None


class TestReproducibilityMetadata:
    """M4.3 P2-Task4: Working tree status, source file hashes, and config traceability."""

    def test_working_tree_status_present(self, tmp_path):
        dataset = m4.load_dataset()
        docs = m4.load_documents()
        pipeline = m4.build_baseline_pipeline(docs)
        results = m4.run_baseline(dataset["cases"], pipeline)
        output = tmp_path / "r.json"
        m4.serialize_results(dataset, results, None, live=False, model=None,
                             output_path=output)
        payload = json.loads(output.read_text(encoding="utf-8"))
        assert "working_tree_status" in payload
        wts = payload["working_tree_status"]
        assert "dirty" in wts
        assert "file_count" in wts

    def test_source_file_hashes_present(self, tmp_path):
        dataset = m4.load_dataset()
        docs = m4.load_documents()
        pipeline = m4.build_baseline_pipeline(docs)
        results = m4.run_baseline(dataset["cases"], pipeline)
        output = tmp_path / "r.json"
        m4.serialize_results(dataset, results, None, live=False, model=None,
                             output_path=output)
        payload = json.loads(output.read_text(encoding="utf-8"))
        assert "source_file_sha256_16" in payload
        hashes = payload["source_file_sha256_16"]
        # Key files must be tracked.
        for key_file in [
            "benchmark/run_multihop_evaluation.py",
            "src/zmwiki/wiki/wiki_agent.py",
            "src/zmwiki/llm_generators.py",
        ]:
            assert key_file in hashes
            assert hashes[key_file] != "missing"
            assert len(hashes[key_file]) == 16

    def test_git_sha_still_present(self, tmp_path):
        dataset = m4.load_dataset()
        docs = m4.load_documents()
        pipeline = m4.build_baseline_pipeline(docs)
        results = m4.run_baseline(dataset["cases"], pipeline)
        output = tmp_path / "r.json"
        m4.serialize_results(dataset, results, None, live=False, model=None,
                             output_path=output)
        payload = json.loads(output.read_text(encoding="utf-8"))
        assert "git_sha" in payload
        assert payload["git_sha"]  # non-empty or "unknown"


# ---------------------------------------------------------------------------
# M4.3 P2 Supplement: Checkpoint isolation and anonymous live auth
# ---------------------------------------------------------------------------


class TestCheckpointIsolation:
    """M4.3 P2-Supp-Task1: Stale checkpoint must not be appended to across runs."""

    def test_stale_checkpoint_blocks_live_start(self, tmp_path, monkeypatch):
        """Output absent but checkpoint present + --live → refuse before any work."""
        output = tmp_path / "result.json"
        checkpoint = tmp_path / "result.checkpoint.jsonl"
        checkpoint.write_text('{"id": "old-case", "stale": true}\n', encoding="utf-8")
        old_content = checkpoint.read_text(encoding="utf-8")

        monkeypatch.setattr(sys, "argv", [
            "run_multihop_evaluation.py",
            "--live", "--base-url", "http://fake.invalid", "--model", "fake-model",
            "--output", str(output),
        ])
        with pytest.raises(SystemExit):
            m4.main()
        # Old checkpoint content must be unchanged.
        assert checkpoint.read_text(encoding="utf-8") == old_content
        # Output file must not have been created.
        assert not output.exists()

    def test_stale_checkpoint_ignored_in_offline_mode(self, tmp_path, monkeypatch):
        """Checkpoint protection only applies to --live; offline runs proceed."""
        output = tmp_path / "result.json"
        checkpoint = tmp_path / "result.checkpoint.jsonl"
        checkpoint.write_text('{"id": "old-case"}\n', encoding="utf-8")

        monkeypatch.setattr(sys, "argv", [
            "run_multihop_evaluation.py",
            "--output", str(output),
        ])
        # Offline mode should complete successfully (no SystemExit).
        m4.main()
        assert output.exists()

    def test_existing_output_blocks_before_data_loading(self, tmp_path, monkeypatch):
        """Existing output file must be rejected early, before dataset loading."""
        output = tmp_path / "result.json"
        output.write_text('{"original": true}', encoding="utf-8")
        original = output.read_text(encoding="utf-8")

        monkeypatch.setattr(sys, "argv", [
            "run_multihop_evaluation.py",
            "--output", str(output),
        ])
        with pytest.raises(SystemExit):
            m4.main()
        assert output.read_text(encoding="utf-8") == original


class TestAnonymousLiveAuth:
    """M4.3 P2-Supp-Task2: Anonymous live requests require explicit --allow-anonymous."""

    def test_no_key_without_allow_anonymous_blocks(self, tmp_path, monkeypatch):
        """--live without API key and without --allow-anonymous must exit before model calls."""
        output = tmp_path / "result.json"
        # Ensure no API key in environment for this test.
        monkeypatch.delenv("ZMWIKI_API_KEY", raising=False)

        monkeypatch.setattr(sys, "argv", [
            "run_multihop_evaluation.py",
            "--live", "--base-url", "http://fake.invalid", "--model", "fake-model",
            "--output", str(output),
        ])
        with pytest.raises(SystemExit) as exc_info:
            m4.main()
        # Exit code 2 from argparse.error.
        assert exc_info.value.code == 2
        # Output must not be created (we exited before any work).
        assert not output.exists()

    def test_no_key_with_allow_anonymous_proceeds(self, tmp_path, monkeypatch, capsys):
        """--live without key but with --allow-anonymous proceeds past pre-flight."""
        output = tmp_path / "result.json"
        monkeypatch.delenv("ZMWIKI_API_KEY", raising=False)
        # Mock run_experiment to avoid any real HTTP calls.
        monkeypatch.setattr(m4, "run_experiment", lambda *a, **kw: [])

        monkeypatch.setattr(sys, "argv", [
            "run_multihop_evaluation.py",
            "--live", "--base-url", "http://fake.invalid", "--model", "fake-model",
            "--allow-anonymous",
            "--output", str(output),
        ])
        # Should complete without SystemExit.
        m4.main()
        assert output.exists()
        # Verify the anonymous warning was printed.
        captured = capsys.readouterr()
        assert "anonymous" in captured.out.lower()
        assert "allow-anonymous" in captured.out.lower()

    def test_api_key_configured_no_allow_anonymous_needed(self, tmp_path, monkeypatch):
        """With an API key, --allow-anonymous is not required and no extra prompt."""
        output = tmp_path / "result.json"
        monkeypatch.delenv("ZMWIKI_API_KEY", raising=False)
        monkeypatch.setattr(m4, "run_experiment", lambda *a, **kw: [])

        monkeypatch.setattr(sys, "argv", [
            "run_multihop_evaluation.py",
            "--live", "--base-url", "http://fake.invalid", "--model", "fake-model",
            "--api-key", "sk-test-key-do-not-use",
            "--output", str(output),
        ])
        m4.main()
        assert output.exists()

    def test_api_key_value_not_printed(self, tmp_path, monkeypatch, capsys):
        """The API key value must never appear in stdout or stderr."""
        output = tmp_path / "result.json"
        secret = "sk-super-secret-key-12345"
        monkeypatch.delenv("ZMWIKI_API_KEY", raising=False)
        monkeypatch.setattr(m4, "run_experiment", lambda *a, **kw: [])

        monkeypatch.setattr(sys, "argv", [
            "run_multihop_evaluation.py",
            "--live", "--base-url", "http://fake.invalid", "--model", "fake-model",
            "--api-key", secret,
            "--output", str(output),
        ])
        m4.main()
        captured = capsys.readouterr()
        assert secret not in captured.out
        assert secret not in captured.err
        # Also verify not in the output JSON.
        raw = output.read_text(encoding="utf-8")
        assert secret not in raw

    def test_help_text_mentions_allow_anonymous(self):
        """--allow-anonymous must be documented in the parser help."""
        import inspect
        source = inspect.getsource(m4.main)
        assert "--allow-anonymous" in source


# ---------------------------------------------------------------------------
# M4.3 P3: --limit parameter for controlled smoke tests
# ---------------------------------------------------------------------------


class TestLimitParameter:
    """M4.3 P3-Task2: --limit restricts evaluation to first N cases."""

    def test_limit_three_runs_only_three_cases(self, tmp_path, monkeypatch):
        """--limit 3 must produce exactly 3 baseline results, not 26."""
        output = tmp_path / "limit3.json"
        monkeypatch.setattr(sys, "argv", [
            "run_multihop_evaluation.py",
            "--limit", "3",
            "--output", str(output),
        ])
        m4.main()
        payload = json.loads(output.read_text(encoding="utf-8"))
        per_case = payload["results"]["baseline"]["per_case"]
        assert len(per_case) == 3
        # First 3 case IDs from the dataset.
        assert per_case[0]["id"] == "mh01_manager_title"
        assert per_case[1]["id"] == "mh02_project_budget"
        assert per_case[2]["id"] == "mh03_orion_lead_chain"

    def test_limit_recorded_in_config(self, tmp_path, monkeypatch):
        """The applied limit must be recorded in the output config."""
        output = tmp_path / "limit_cfg.json"
        monkeypatch.setattr(sys, "argv", [
            "run_multihop_evaluation.py",
            "--limit", "2",
            "--output", str(output),
        ])
        m4.main()
        payload = json.loads(output.read_text(encoding="utf-8"))
        assert payload["config"]["limit"] == 2

    def test_no_limit_runs_all_cases(self, tmp_path, monkeypatch):
        """Without --limit, all 26 cases are evaluated."""
        output = tmp_path / "nolimit.json"
        monkeypatch.setattr(sys, "argv", [
            "run_multihop_evaluation.py",
            "--output", str(output),
        ])
        m4.main()
        payload = json.loads(output.read_text(encoding="utf-8"))
        assert len(payload["results"]["baseline"]["per_case"]) == 26
        assert payload["config"]["limit"] is None

    def test_limit_zero_rejected(self, tmp_path, monkeypatch):
        """--limit 0 must be rejected with parser.error."""
        output = tmp_path / "bad.json"
        monkeypatch.setattr(sys, "argv", [
            "run_multihop_evaluation.py",
            "--limit", "0",
            "--output", str(output),
        ])
        with pytest.raises(SystemExit) as exc_info:
            m4.main()
        assert exc_info.value.code == 2
        assert not output.exists()

    def test_limit_exceeds_dataset_rejected(self, tmp_path, monkeypatch):
        """--limit greater than dataset size must be rejected."""
        output = tmp_path / "bad2.json"
        monkeypatch.setattr(sys, "argv", [
            "run_multihop_evaluation.py",
            "--limit", "99",
            "--output", str(output),
        ])
        with pytest.raises(SystemExit) as exc_info:
            m4.main()
        assert exc_info.value.code == 2
        assert not output.exists()

    def test_limit_with_live_mocks_experiment(self, tmp_path, monkeypatch):
        """--limit works with --live; experiment also limited to N cases."""
        output = tmp_path / "limit_live.json"
        monkeypatch.delenv("ZMWIKI_API_KEY", raising=False)
        # Mock run_experiment to avoid real HTTP calls; capture the cases passed.
        received_cases = []
        def fake_run_experiment(cases, *a, **kw):
            received_cases.extend(cases)
            return []
        monkeypatch.setattr(m4, "run_experiment", fake_run_experiment)

        monkeypatch.setattr(sys, "argv", [
            "run_multihop_evaluation.py",
            "--live", "--base-url", "http://fake.invalid", "--model", "fake",
            "--allow-anonymous",
            "--limit", "3",
            "--output", str(output),
        ])
        m4.main()
        # Both baseline and experiment received only 3 cases.
        assert len(received_cases) == 3
        payload = json.loads(output.read_text(encoding="utf-8"))
        assert len(payload["results"]["baseline"]["per_case"]) == 3


# ---------------------------------------------------------------------------
# Regression guards
# ---------------------------------------------------------------------------


class TestRegressionGuards:
    def test_existing_rag_pipeline_unchanged(self):
        """The existing RAG pipeline constructor still works with required args."""
        from zmwiki import (
            DeterministicRAGPipeline, FixedSizeChunker, DeterministicEmbedder,
            InMemoryVectorStore, InMemoryLexicalIndex, HybridRetriever,
            DenseRetriever, LexicalRetriever, WeightedScoreFusion,
            DeterministicContextBuilder, DeterministicGenerator, DeterministicReranker,
        )
        vs = InMemoryVectorStore()
        li = InMemoryLexicalIndex()
        emb = DeterministicEmbedder()
        p = DeterministicRAGPipeline(
            chunker=FixedSizeChunker(chunk_size=120, overlap=20),
            embedder=emb,
            vector_store=vs,
            lexical_index=li,
            retriever=HybridRetriever(
                DenseRetriever(emb, vs), LexicalRetriever(li), WeightedScoreFusion(),
            ),
            reranker=DeterministicReranker(),
            context_builder=DeterministicContextBuilder(),
            generator=DeterministicGenerator(),
        )
        assert p is not None

    def test_wiki_agent_not_in_top_level_exports(self):
        import zmwiki
        assert not hasattr(zmwiki, "WikiAgent")

    def test_dataset_file_not_modified_existing(self):
        """The existing retrieval_small.json should still be valid and unchanged."""
        path = m4.ROOT / "benchmark" / "datasets" / "retrieval_small.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        assert len(payload["cases"]) == 8


# ---------------------------------------------------------------------------
# M4.4: Per-step agent diagnostics
# ---------------------------------------------------------------------------


class TestStepDiagnostics:
    """M4.4: AgentResult.steps must survive into per-case results and JSON."""

    CASE = {
        "id": "mh_diag", "question": "q", "answerable": True,
        "required_page_ids": ["a"], "hops": 1,
    }

    @staticmethod
    def _agent_result():
        from zmwiki.wiki.wiki_agent import AgentResult, AgentStep
        return AgentResult(
            answer="a", sources=(), status="answered",
            steps=(
                AgentStep(step=0, action="search", params={},
                          result_summary="found 3 results"),
                AgentStep(step=1, action="protocol_error", params={},
                          result_summary="invalid model output",
                          error="missing 'action' field",
                          raw_output='{"action": "search"}{"action": "read"}'),
                AgentStep(step=2, action="read", params={},
                          result_summary="read failed: paths not found",
                          error="no page 'x'"),
            ),
        )

    def _run_with_fake_agent(self, monkeypatch, answer_fn):
        class FakeAgent:
            def __init__(self, *args, **kwargs):
                pass

            def answer(self, question):
                return answer_fn(question)

        monkeypatch.setattr(m4, "WikiAgent", FakeAgent)
        return m4.run_experiment(
            [self.CASE], WikiIndex(), base_url="http://fake.invalid",
            model="fake-model", api_key=None,
        )

    def test_multiple_step_errors_preserved(self, monkeypatch):
        results = self._run_with_fake_agent(
            monkeypatch, lambda q: self._agent_result()
        )
        assert len(results) == 1
        details = results[0].step_details
        assert len(details) == 3
        assert [d["step"] for d in details] == [0, 1, 2]
        assert [d["action"] for d in details] == ["search", "protocol_error", "read"]
        assert [d["error"] for d in details] == [
            None, "missing 'action' field", "no page 'x'",
        ]
        assert details[2]["result_summary"] == "read failed: paths not found"
        assert details[0]["raw_output"] is None
        assert details[1]["raw_output"] == '{"action": "search"}{"action": "read"}'

    def test_to_dict_json_contains_step_details(self, monkeypatch):
        results = self._run_with_fake_agent(
            monkeypatch, lambda q: self._agent_result()
        )
        payload = json.loads(json.dumps(results[0].to_dict()))
        assert "step_details" in payload
        assert payload["step_details"][1]["error"] == "missing 'action' field"
        assert payload["step_details"][1]["raw_output"] == '{"action": "search"}{"action": "read"}'
        assert payload["step_details"][2]["action"] == "read"

    def test_exception_yields_empty_step_details(self, monkeypatch):
        def boom(question):
            raise RuntimeError("simulated agent failure")

        results = self._run_with_fake_agent(monkeypatch, boom)
        assert results[0].status == "error"
        assert results[0].step_details == []
        assert results[0].to_dict()["step_details"] == []

    def test_legacy_construction_without_step_details(self):
        result = m4.CaseResult(
            id="old", question="q", answer="a", answer_correct=True,
            status="answered", sources=[], accessed_pages=[], required_pages=[],
            evidence_recall=None, source_precision=0.0, steps=1, pages_read=0,
        )
        assert result.step_details == []
        assert result.to_dict()["step_details"] == []
        assert result.search_hit_page_ids == []
        assert result.read_page_ids == []
        assert result.access_recall is None
        assert result.read_evidence_recall is None


class TestReadEvidenceAudit:
    """Search hits, actual reads, and legacy access must stay distinguishable."""

    CASE = {
        "id": "read_audit", "question": "q", "answerable": True,
        "required_page_ids": ["hit", "read"], "hops": 2,
        "answer_keyword_groups": [["ok"]],
    }

    @staticmethod
    def _agent_result():
        from zmwiki.wiki.wiki_agent import AgentResult, AgentStatus, AgentStep

        return AgentResult(
            answer="ok", sources=("wiki/read",), status=AgentStatus.ANSWERED,
            read_page_ids=("read",), pages_read=1,
            steps=(
                AgentStep(
                    step=0, action="search",
                    params={"query": "needle", "top_k": 5},
                    result_summary="search returned 2 candidates",
                    search_hit_page_ids=("hit", "other"),
                ),
                AgentStep(
                    step=1, action="read",
                    params={"paths": ["wiki/read"], "offset": 12},
                    result_summary="read 1 page at offset 12",
                    read_page_ids=("read",),
                ),
                AgentStep(
                    step=2, action="answer",
                    params={"text": "ok", "sources": ["read"]},
                    result_summary="answer produced (1 sources)",
                ),
            ),
        )

    def _run(self, monkeypatch, index):
        agent_result = self._agent_result()

        class FakeAgent:
            def __init__(self, index, **kwargs):
                self.index = index

            def answer(self, question):
                # Simulate the monitor-visible accesses of a search and read.
                self.index.get("hit")
                self.index.get("other")
                self.index.get_by_path("wiki/read")
                return agent_result

        monkeypatch.setattr(m4, "WikiAgent", FakeAgent)
        return m4.run_experiment(
            [self.CASE], index, base_url="http://fake.invalid",
            model="fake-model", api_key="sk-secret-not-in-logs",
        )

    def test_access_and_read_recall_use_different_sets(self, monkeypatch):
        index = WikiIndex()
        for page_id in ("hit", "read", "other"):
            path = f"wiki/{page_id}"
            index.upsert(WikiPage(
                id=page_id, path=path, title=page_id.title(),
                content=f"{page_id} content",
            ))

        results = self._run(monkeypatch, index)
        assert len(results) == 1
        result = results[0]

        assert result.accessed_pages == ["hit", "other", "read"]
        assert result.search_hit_page_ids == ["hit", "other"]
        assert result.read_page_ids == ["read"]
        assert result.evidence_recall == 1.0
        assert result.access_recall == 1.0
        assert result.read_evidence_recall == 0.5

        agg = m4.aggregate_results([self.CASE], results)
        metrics = agg["retrieval_evidence_metrics"]
        assert metrics["mean_evidence_recall_answerable"]["value"] == 1.0
        assert metrics["mean_access_recall_answerable"]["value"] == 1.0
        assert metrics["mean_access_recall_answerable"]["cases_with_access_data"] == 1
        assert metrics["mean_read_evidence_recall_answerable"]["value"] == 0.5
        assert metrics["mean_read_evidence_recall_answerable"]["cases_with_read_data"] == 1
        assert agg["mean_evidence_recall"] == 1.0
        assert agg["mean_access_recall"] == 1.0
        assert agg["mean_read_evidence_recall"] == 0.5

    def test_step_params_and_page_ids_are_serialized(self, monkeypatch):
        index = WikiIndex()
        for page_id in ("hit", "read", "other"):
            index.upsert(WikiPage(
                id=page_id, path=f"wiki/{page_id}", title=page_id.title(),
                content=f"{page_id} content",
            ))

        result = self._run(monkeypatch, index)[0]
        payload = json.loads(json.dumps(result.to_dict()))
        details = payload["step_details"]

        assert details[0]["params"] == {"query": "needle", "top_k": 5}
        assert details[0]["search_hit_page_ids"] == ["hit", "other"]
        assert details[0]["read_page_ids"] == []
        assert details[1]["params"] == {
            "paths": ["wiki/read"], "offset": 12,
        }
        assert details[1]["search_hit_page_ids"] == []
        assert details[1]["read_page_ids"] == ["read"]
        assert details[2]["params"]["sources"] == ["read"]
        assert payload["search_hit_page_ids"] == ["hit", "other"]
        assert payload["read_page_ids"] == ["read"]

        raw = json.dumps(payload)
        assert "sk-secret-not-in-logs" not in raw
