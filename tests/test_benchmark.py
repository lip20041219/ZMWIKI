"""Benchmark runner tests: EvaluationCaseSet -> RetrievalEvaluationRunner.

Covers the Phase 9D BenchmarkRunner: the ABC contract, constructor dependency
injection (a RetrievalEvaluationRunner is required, never built internally),
input validation (cases type + top_k / candidate_k int rules + candidate_k >=
top_k), exactly-once delegation with the cases tuple converted to a list in
order, identity passthrough of the delegated EvaluationResult (never rebuilt,
never recomputed), empty-CaseSet delegation (never short-circuited), input
immutability, determinism across calls and instances, proof the benchmark
runner touches only the injected evaluation runner, and a real E2E through the
full deterministic retrieval pipeline.
"""

import pytest

from zmwiki import (
    BenchmarkRunner,
    Chunk,
    DeterministicBenchmarkRunner,
    DeterministicContextBuilder,
    DeterministicEmbedder,
    DeterministicRAGPipeline,
    DeterministicReranker,
    DenseRetriever,
    Document,
    EvaluationCase,
    EvaluationCaseSet,
    EvaluationHarness,
    EvaluationResult,
    FixedSizeChunker,
    HybridRetriever,
    InMemoryLexicalIndex,
    InMemoryVectorStore,
    LexicalRetriever,
    RAGPipeline,
    RetrievalEvaluationRunner,
    RetrievalEvaluator,
    SearchResult,
    WeightedScoreFusion,
)


def _case(query, *ids):
    return EvaluationCase(query=query, relevant_chunk_ids=ids)


def _sentinel():
    # Distinctive nonzero fields: if the benchmark ever recomputed or rebuilt
    # the result, these would change or the object identity would break.
    return EvaluationResult(
        hit_rate=0.123,
        recall=0.456,
        precision=0.789,
        total_cases=7,
        evaluated_cases=7,
        mrr=0.25,
        map=0.5,
        ndcg=0.75,
    )


def _sr(cid, score=1.0):
    return SearchResult(
        chunk=Chunk(id=cid, document_id="doc", text=cid, position=0), score=score
    )


class FakePipeline(RAGPipeline):
    """RAGPipeline stub returning scripted results and recording retrieve calls."""

    def __init__(self, script=None):
        self.script = {q: list(rs) for q, rs in (script or {}).items()}
        self.default = []
        self.retrieve_calls = []

    def ingest(self, document):
        raise AssertionError("must never ingest through the benchmark path")

    def retrieve(self, query, *, top_k=5, candidate_k=20):
        self.retrieve_calls.append(query)
        return list(self.script.get(query, self.default))

    def query(self, query, *, top_k=5, candidate_k=20, max_chars=2000):
        raise AssertionError("must never call query")


class FakeEvaluationRunner(RetrievalEvaluationRunner):
    """RetrievalEvaluationRunner that records run() and may short-circuit.

    When ``result`` is given it is returned (identity) without touching the
    pipeline -- proving the benchmark passes results through untouched. When
    omitted, ``run`` delegates to the real parent implementation over a
    scripted pipeline, exercising the genuine runner + harness + evaluator
    chain deterministically.
    """

    def __init__(self, script=None, result=None):
        self._pipe = FakePipeline(script=script)
        super().__init__(pipeline=self._pipe, harness=EvaluationHarness(RetrievalEvaluator()))
        self.sentinel = result
        self.run_count = 0
        self.last_cases = None
        self.last_top_k = None
        self.last_candidate_k = None

    def run(self, cases, *, top_k=5, candidate_k=20):
        self.run_count += 1
        self.last_cases = cases
        self.last_top_k = top_k
        self.last_candidate_k = candidate_k
        if self.sentinel is not None:
            return self.sentinel
        return super().run(cases, top_k=top_k, candidate_k=candidate_k)


def _benchmark(fake):
    return DeterministicBenchmarkRunner(evaluation_runner=fake)


# --- A. ABC ---


def test_benchmark_runner_is_abstract():
    with pytest.raises(TypeError):
        BenchmarkRunner()  # type: ignore[abstract]


def test_deterministic_benchmark_runner_is_instantiable():
    bench = _benchmark(FakeEvaluationRunner(result=_sentinel()))
    assert isinstance(bench, BenchmarkRunner)
    assert isinstance(bench, DeterministicBenchmarkRunner)


# --- B. Dependency injection ---


@pytest.mark.parametrize("bad", [None, "not a runner", 5, object()])
def test_constructor_requires_retrieval_evaluation_runner(bad):
    with pytest.raises(TypeError):
        DeterministicBenchmarkRunner(evaluation_runner=bad)  # type: ignore[arg-type]


def test_injected_runner_is_kept_not_constructed():
    fake = FakeEvaluationRunner(result=_sentinel())
    bench = _benchmark(fake)
    assert bench.evaluation_runner is fake  # nothing built internally


# --- C. Input validation (all fail before delegation) ---


@pytest.mark.parametrize(
    "bad",
    [None, [], (), "q", _case("q", "A")],  # a bare EvaluationCase is not a CaseSet
)
def test_cases_must_be_evaluation_case_set(bad):
    fake = FakeEvaluationRunner(result=_sentinel())
    with pytest.raises(TypeError):
        _benchmark(fake).run(bad)  # type: ignore[arg-type]
    assert fake.run_count == 0


@pytest.mark.parametrize("bad", [True, 1.5, "5", None])
def test_top_k_type_rejected(bad):
    fake = FakeEvaluationRunner(result=_sentinel())
    with pytest.raises(TypeError):
        _benchmark(fake).run(EvaluationCaseSet(cases=()), top_k=bad)  # type: ignore[arg-type]
    assert fake.run_count == 0


@pytest.mark.parametrize("bad", [0, -3])
def test_top_k_value_rejected(bad):
    fake = FakeEvaluationRunner(result=_sentinel())
    with pytest.raises(ValueError):
        _benchmark(fake).run(EvaluationCaseSet(cases=()), top_k=bad)
    assert fake.run_count == 0


@pytest.mark.parametrize("bad", [True, 1.5, "20", None])
def test_candidate_k_type_rejected(bad):
    fake = FakeEvaluationRunner(result=_sentinel())
    with pytest.raises(TypeError):
        _benchmark(fake).run(EvaluationCaseSet(cases=()), candidate_k=bad)  # type: ignore[arg-type]
    assert fake.run_count == 0


@pytest.mark.parametrize("bad", [0, -2])
def test_candidate_k_value_rejected(bad):
    fake = FakeEvaluationRunner(result=_sentinel())
    with pytest.raises(ValueError):
        _benchmark(fake).run(EvaluationCaseSet(cases=()), candidate_k=bad)
    assert fake.run_count == 0


def test_candidate_k_below_top_k_rejected():
    fake = FakeEvaluationRunner(result=_sentinel())
    with pytest.raises(ValueError):
        _benchmark(fake).run(EvaluationCaseSet(cases=()), top_k=30)  # default cand 20
    assert fake.run_count == 0
    with pytest.raises(ValueError):
        _benchmark(fake).run(EvaluationCaseSet(cases=()), top_k=10, candidate_k=5)
    assert fake.run_count == 0


# --- D. Delegation ---


def test_delegates_exactly_once_with_cases_list_in_order():
    case_set = EvaluationCaseSet(
        cases=(_case("z", "A"), _case("a", "B"), _case("m", "C"))
    )
    fake = FakeEvaluationRunner(result=_sentinel())
    _benchmark(fake).run(case_set, top_k=3, candidate_k=9)

    assert fake.run_count == 1  # exactly once
    assert isinstance(fake.last_cases, list)  # tuple -> the runner's list type
    assert fake.last_cases == list(case_set.cases)
    assert [c.query for c in fake.last_cases] == ["z", "a", "m"]  # order kept
    assert fake.last_top_k == 3 and fake.last_candidate_k == 9  # forwarded verbatim


def test_delegation_default_cutoffs():
    fake = FakeEvaluationRunner(result=_sentinel())
    _benchmark(fake).run(EvaluationCaseSet(cases=(_case("q", "A"),)))
    assert fake.run_count == 1
    assert (fake.last_top_k, fake.last_candidate_k) == (5, 20)


def test_delegated_cases_is_a_fresh_list_not_the_tuple():
    case_set = EvaluationCaseSet(cases=(_case("q", "A"),))
    fake = FakeEvaluationRunner(result=_sentinel())
    _benchmark(fake).run(case_set)
    assert fake.last_cases is not case_set.cases
    assert fake.last_cases == list(case_set.cases)


# --- E. Result passthrough ---


def test_result_identity_passthrough():
    sentinel = _sentinel()
    fake = FakeEvaluationRunner(result=sentinel)
    out = _benchmark(fake).run(EvaluationCaseSet(cases=(_case("q", "A"),)))
    assert out is sentinel  # the exact object the runner returned


# --- F. No metric computation / no rebuild ---


def test_benchmark_never_recomputes_or_rebuilds_metrics():
    sentinel = _sentinel()
    fake = FakeEvaluationRunner(result=sentinel)
    out = _benchmark(fake).run(EvaluationCaseSet(cases=(_case("q", "A"),)))
    assert out is sentinel
    assert (out.hit_rate, out.recall, out.precision) == (0.123, 0.456, 0.789)
    assert (out.mrr, out.map, out.ndcg) == (0.25, 0.5, 0.75)
    assert (out.total_cases, out.evaluated_cases) == (7, 7)


# --- G. Empty CaseSet ---


def test_empty_case_set_is_delegated_not_short_circuited():
    sentinel = _sentinel()
    fake = FakeEvaluationRunner(result=sentinel)
    out = _benchmark(fake).run(EvaluationCaseSet(cases=()))
    assert fake.run_count == 1  # never an early `if not cases: return ...`
    assert fake.last_cases == []  # list(()) reached the runner
    assert out is sentinel


def test_empty_case_set_through_real_chain_keeps_evaluator_contract():
    fake = FakeEvaluationRunner()  # real parent run() over an empty script
    out = _benchmark(fake).run(EvaluationCaseSet(cases=()))
    assert fake.run_count == 1 and fake.last_cases == []
    assert out == RetrievalEvaluator().evaluate([], [])  # empty-run contract


# --- H. Immutability ---


def test_run_never_mutates_the_case_set():
    cases = (_case("q1", "A"), _case("q2", "B", "C"))
    case_set = EvaluationCaseSet(cases=cases)
    snapshot_ids = [id(c) for c in case_set.cases]
    snapshot = [(c.query, c.relevant_chunk_ids) for c in case_set.cases]

    fake = FakeEvaluationRunner(result=_sentinel())
    _benchmark(fake).run(case_set, top_k=3, candidate_k=9)

    assert case_set.cases == cases
    assert [c.query for c in case_set.cases] == ["q1", "q2"]
    assert [(c.query, c.relevant_chunk_ids) for c in case_set.cases] == snapshot
    assert [id(c) for c in case_set.cases] == snapshot_ids  # identities unchanged


# --- I. Determinism ---


def test_repeated_calls_are_equal():
    script = {"q1": [_sr("A")], "q2": [_sr("X")]}
    cases = EvaluationCaseSet(cases=(_case("q1", "A"), _case("q2", "A", "B")))
    fake = FakeEvaluationRunner(script=script)  # real chain -> real metrics

    first = _benchmark(fake).run(cases, top_k=2, candidate_k=4)
    second = _benchmark(fake).run(cases, top_k=2, candidate_k=4)
    assert first == second
    assert first.to_dict() == second.to_dict()


def test_cross_instance_calls_are_equal():
    script = {"q1": [_sr("A")], "q2": []}
    cases = EvaluationCaseSet(cases=(_case("q1", "A"), _case("q2", "A")))
    out1 = _benchmark(FakeEvaluationRunner(script=script)).run(cases)
    out2 = _benchmark(FakeEvaluationRunner(script=script)).run(cases)
    assert out1 == out2  # independent instances over identical inputs agree


# --- benchmark touches only the injected evaluation runner ---


def test_benchmark_touches_only_the_injected_evaluation_runner():
    fake = FakeEvaluationRunner(result=_sentinel())
    _benchmark(fake).run(EvaluationCaseSet(cases=(_case("q", "A"),)))
    # The pipeline was never driven, and the fake short-circuited the harness/
    # evaluator too: the benchmark itself called only evaluation_runner.run.
    assert fake._pipe.retrieve_calls == []
    assert fake.run_count == 1


# --- J. Real E2E ---


def _real_wiring():
    chunker = FixedSizeChunker(chunk_size=12)
    embedder = DeterministicEmbedder(dimension=8)
    vstore, lindex = InMemoryVectorStore(), InMemoryLexicalIndex()
    hybrid = HybridRetriever(
        DenseRetriever(embedder, vstore),
        LexicalRetriever(lindex),
        WeightedScoreFusion(),
    )
    pipeline = DeterministicRAGPipeline(
        chunker=chunker,
        embedder=embedder,
        vector_store=vstore,
        lexical_index=lindex,
        retriever=hybrid,
        reranker=DeterministicReranker(),
        context_builder=DeterministicContextBuilder(),
    )
    pipeline.ingest(Document(id="a", content="needle apple banana"))
    pipeline.ingest(Document(id="b", content="orange pear grape"))
    harness = EvaluationHarness(RetrievalEvaluator())
    evaluation_runner = RetrievalEvaluationRunner(pipeline=pipeline, harness=harness)
    return evaluation_runner


def test_real_e2e_benchmark_runner_full_retrieval_chain():
    evaluation_runner = _real_wiring()
    bench = DeterministicBenchmarkRunner(evaluation_runner=evaluation_runner)

    case_set = EvaluationCaseSet(
        cases=(
            _case("needle", "a::0"),       # present -> a hit
            _case("not-present", "ghost::0"),  # id not in the index -> a miss
        )
    )
    result = bench.run(case_set, top_k=10, candidate_k=10)

    assert isinstance(result, EvaluationResult)
    assert (result.total_cases, result.evaluated_cases) == (2, 2)
    assert result.hit_rate == 0.5  # needle hit, ghost id miss
    assert result.recall == 0.5  # macro average of 1.0 and 0.0
    # "a::0" is the unique lexical match for "needle", so it ranks first:
    # MRR / MAP / NDCG average 0.5 over the hit + the empty miss.
    assert result.mrr == 0.5
    assert result.map == 0.5
    assert result.ndcg == 0.5

    # Deterministic re-run over the same immutable input.
    assert bench.run(case_set, top_k=10, candidate_k=10) == result
    assert bench.run(case_set, top_k=10, candidate_k=10).to_dict() == result.to_dict()


def test_public_import():
    from zmwiki import __all__

    assert "BenchmarkRunner" in __all__
    assert "DeterministicBenchmarkRunner" in __all__
    assert issubclass(DeterministicBenchmarkRunner, BenchmarkRunner)
