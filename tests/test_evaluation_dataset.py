"""EvaluationCaseSet tests: ordered immutable case collection contract.

Covers the Phase 9B EvaluationCaseSet: construction rules (tuple-only, empty
legal), element validation (EvaluationCase only, delegated -- CaseSet never
re-validates query / relevant ids), ordering (input order preserved, never
sorted/deduplicated), duplicate semantics (an ordered sequence, not a math
set), immutability (frozen, tuple-backed, matching Context.chunks), minimal
sequence behavior (len / iter / index), JSON-friendly deterministic
serialization (to_dict), and integration proof that ``case_set.cases`` feeds
the existing RetrievalEvaluationRunner unchanged -- plus a real-pipeline E2E
proving CaseSet is only an input contract (it computes no metric, performs no
retrieval).
"""

import json

import pytest

from zmwiki import (
    Chunk,
    DeterministicContextBuilder,
    DeterministicEmbedder,
    DeterministicRAGPipeline,
    DeterministicReranker,
    DenseRetriever,
    Document,
    EvaluationCase,
    EvaluationCaseSet,
    EvaluationHarness,
    EvaluationPrediction,
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


# --- A. Construction ---


def test_empty_case_set_is_valid():
    case_set = EvaluationCaseSet(cases=())
    assert len(case_set) == 0
    assert case_set.cases == ()


def test_single_case():
    case = _case("q", "A")
    case_set = EvaluationCaseSet(cases=(case,))
    assert case_set.cases == (case,)
    assert len(case_set) == 1


def test_multiple_cases():
    cases = (_case("q1", "A"), _case("q2", "A", "B"), _case("q3", "C"))
    case_set = EvaluationCaseSet(cases=cases)
    assert case_set.cases == cases
    assert len(case_set) == 3


def test_cases_must_be_a_tuple():
    # A mutable list is refused -- the same immutable-tuple contract as
    # Context.chunks. Callers convert via tuple(cases) if they hold a list.
    with pytest.raises(TypeError):
        EvaluationCaseSet(cases=[_case("q", "A")])  # type: ignore[arg-type]


@pytest.mark.parametrize("bad", ["x", 5, {"A"}, {"k": "v"}])
def test_wrong_cases_container_rejected(bad):
    with pytest.raises(TypeError):
        EvaluationCaseSet(cases=bad)  # type: ignore[arg-type]


# --- B. Element validation ---


def test_non_evaluation_case_element_rejected():
    with pytest.raises(TypeError):
        EvaluationCaseSet(cases=("not a case",))  # type: ignore[list-item]


def test_mixed_element_types_rejected():
    with pytest.raises(TypeError):
        EvaluationCaseSet(  # type: ignore[list-item]
            cases=(_case("q", "A"), "not a case")
        )


def test_all_valid_evaluation_cases_accepted():
    cases = (_case("q1", "A"), _case("q2", "B", "C"))
    case_set = EvaluationCaseSet(cases=cases)
    assert case_set.cases == cases


# --- C. Ordering ---


def test_input_order_preserved():
    z, a, m = _case("z", "A"), _case("a", "B"), _case("m", "C")
    case_set = EvaluationCaseSet(cases=(z, a, m))
    assert case_set.cases == (z, a, m)  # z a m -- never reordered to a m z


def test_query_order_preserved():
    case_set = EvaluationCaseSet(
        cases=(_case("z", "A"), _case("a", "B"), _case("m", "C"))
    )
    assert [c.query for c in case_set] == ["z", "a", "m"]


def test_relevant_ids_unchanged_and_canonical():
    # EvaluationCase owns canonicalization ({"B","A"} -> ("A","B")); CaseSet
    # never re-sorts or rewrites what the case already holds.
    case = EvaluationCase(query="q", relevant_chunk_ids=["B", "A"])
    case_set = EvaluationCaseSet(cases=(case,))
    assert case_set.cases[0].relevant_chunk_ids == ("A", "B")
    assert case_set.to_dict()["cases"][0]["relevant_chunk_ids"] == ["A", "B"]


# --- D. Duplicate semantics ---


def test_duplicate_cases_allowed():
    case = _case("needle", "A")
    case_set = EvaluationCaseSet(cases=(case, case))
    assert len(case_set) == 2
    assert case_set.cases == (case, case)


def test_duplicate_object_references_allowed():
    case = _case("needle", "A")
    case_set = EvaluationCaseSet(cases=(case, case))
    assert case_set.cases[0] is case and case_set.cases[1] is case


def test_no_deduplication():
    one = _case("needle", "A")
    two = _case("needle", "A")
    case_set = EvaluationCaseSet(cases=(one, two))
    # Structurally equal but distinct objects both survive.
    assert len(case_set) == 2
    assert case_set.cases[0] == case_set.cases[1]


def test_same_query_different_relevant_ids_preserved():
    case_set = EvaluationCaseSet(
        cases=(_case("q", "A"), _case("q", "A", "B"), _case("q", "C"))
    )
    assert [c.relevant_chunk_ids for c in case_set] == [("A",), ("A", "B"), ("C",)]


# --- E. Immutability ---


def test_cases_is_a_tuple():
    case_set = EvaluationCaseSet(cases=(_case("q", "A"),))
    assert isinstance(case_set.cases, tuple)


def test_cannot_assign_cases():
    case_set = EvaluationCaseSet(cases=(_case("q", "A"),))
    with pytest.raises(AttributeError):
        case_set.cases = ()  # type: ignore[misc]  # frozen dataclass


def test_cannot_append_or_mutate():
    case_set = EvaluationCaseSet(cases=(_case("q", "A"),))
    with pytest.raises(AttributeError):
        case_set.append(_case("q2", "B"))  # type: ignore[attr-defined]
    # The tuple field itself offers no mutators, so nothing aliases in.
    with pytest.raises(AttributeError):
        case_set.cases.append(_case("q2", "B"))  # type: ignore[attr-defined]


def test_caller_container_cannot_mutate_internal_state():
    # A list is rejected outright; a caller-held tuple is immutable, so the set
    # state can never change after construction. Re-constructing from the same
    # tuple yields an equal set.
    cases = (_case("q", "A"), _case("q2", "B"))
    first = EvaluationCaseSet(cases=cases)
    second = EvaluationCaseSet(cases=cases)
    assert first == second
    assert first.cases == cases


def test_contained_evaluation_case_identity_preserved():
    case = _case("q", "A")
    case_set = EvaluationCaseSet(cases=(case,))
    assert case_set.cases[0] is case  # no copies, no rebuilding
    assert case_set[0] is case


# --- F. Sequence behavior ---


def test_len():
    assert len(EvaluationCaseSet(cases=())) == 0
    assert len(EvaluationCaseSet(cases=(_case("q", "A"), _case("q2", "B")))) == 2


def test_iteration_yields_cases_in_order():
    cases = (_case("z", "A"), _case("a", "B"), _case("m", "C"))
    assert list(EvaluationCaseSet(cases=cases)) == list(cases)


def test_indexing_returns_original_case():
    case_set = EvaluationCaseSet(cases=(_case("q", "A"), _case("q2", "B")))
    assert case_set[0].query == "q"
    assert case_set[-1].query == "q2"


# --- G. Serialization ---


def test_to_dict_structure():
    case_set = EvaluationCaseSet(
        cases=(_case("needle", "A", "B"), _case("missing", "C"))
    )
    assert case_set.to_dict() == {
        "cases": [
            {"query": "needle", "relevant_chunk_ids": ["A", "B"]},
            {"query": "missing", "relevant_chunk_ids": ["C"]},
        ]
    }


def test_to_dict_is_json_dumpable():
    case_set = EvaluationCaseSet(cases=(_case("needle", "A"),))
    text = json.dumps(case_set.to_dict())
    assert json.loads(text) == case_set.to_dict()


def test_fixed_key_structure():
    case_set = EvaluationCaseSet(cases=(_case("q", "A"), _case("q2", "B")))
    assert list(case_set.to_dict().keys()) == ["cases"]
    for entry in case_set.to_dict()["cases"]:
        assert list(entry.keys()) == ["query", "relevant_chunk_ids"]


def test_to_dict_deterministic_across_calls():
    case_set = EvaluationCaseSet(cases=(_case("needle", "A"), _case("q", "B", "A")))
    assert case_set.to_dict() == case_set.to_dict()
    assert json.dumps(case_set.to_dict()) == json.dumps(case_set.to_dict())


def test_cross_instance_identical_output():
    cases = (_case("needle", "A"), _case("missing", "does-not-exist"))
    assert EvaluationCaseSet(cases=cases).to_dict() == EvaluationCaseSet(
        cases=cases
    ).to_dict()


def test_no_repr_or_object_address_leakage():
    case_set = EvaluationCaseSet(cases=(_case("q", "A"),))
    text = json.dumps(case_set.to_dict())
    assert "EvaluationCase" not in text
    assert "at 0x" not in text and "0x" not in text
    assert text == '{"cases": [{"query": "q", "relevant_chunk_ids": ["A"]}]}'


def test_no_timestamp_path_or_hash_fields():
    case_set = EvaluationCaseSet(cases=(_case("q", "A"),))
    payload = json.dumps(case_set.to_dict())
    for forbidden in ("time", "path", "hash", "source", "name", "version"):
        assert forbidden not in payload


# --- H. Evaluation integration ---


class FakePipeline(RAGPipeline):
    """RAGPipeline stub recording retrieve calls and returning scripted results."""

    def __init__(self, per_query=None, default=None):
        self.per_query = {q: list(r) for q, r in (per_query or {}).items()}
        self.default = list(default or [])
        self.calls = []

    def ingest(self, document):
        raise AssertionError("runner must never ingest")

    def retrieve(self, query, *, top_k=5, candidate_k=20):
        self.calls.append((query, top_k, candidate_k))
        return list(self.per_query.get(query, self.default))

    def query(self, query, *, top_k=5, candidate_k=20, max_chars=2000):
        raise AssertionError("runner must never call query")


def _sr(cid, score=0.0):
    return SearchResult(
        chunk=Chunk(id=cid, document_id="doc", text=cid, position=0), score=score
    )


def _harness():
    return EvaluationHarness(RetrievalEvaluator())


def test_case_set_cases_feed_retrieval_evaluation_runner():
    case_set = EvaluationCaseSet(cases=(_case("needle", "A"), _case("missing", "X")))
    pipe = FakePipeline(per_query={"needle": [_sr("A")], "missing": []})
    result = RetrievalEvaluationRunner(pipeline=pipe, harness=_harness()).run(
        list(case_set.cases)  # tuple -> list: the runner's existing input type
    )
    assert isinstance(result, EvaluationResult)
    assert (result.total_cases, result.evaluated_cases) == (2, 2)


def test_case_order_reaches_runner_unchanged():
    case_set = EvaluationCaseSet(
        cases=(_case("z", "A"), _case("a", "B"), _case("m", "C"))
    )
    pipe = FakePipeline(default=[_sr("X")])
    RetrievalEvaluationRunner(pipeline=pipe, harness=_harness()).run(
        list(case_set.cases)
    )
    assert [q for q, _, _ in pipe.calls] == ["z", "a", "m"]  # input order kept


def test_predictions_aligned_by_position_to_cases():
    # Crossed results prove alignment is by position: q1's prediction comes
    # from q1's own retrieval (a miss), never re-matched to q1's ground truth.
    cases = (_case("q1", "A"), _case("q2", "B"))
    pipe = FakePipeline(per_query={"q1": [_sr("B")], "q2": [_sr("A")]})
    result = RetrievalEvaluationRunner(pipeline=pipe, harness=_harness()).run(
        list(EvaluationCaseSet(cases=cases).cases)
    )
    assert result.hit_rate == 0.0  # neither query retrieved its own relevant id
    assert result.recall == 0.0


def test_metrics_come_from_evaluator_not_case_set():
    case_set = EvaluationCaseSet(cases=(_case("q1", "A"), _case("q2", "B")))
    pipe = FakePipeline(per_query={"q1": [_sr("A")], "q2": []})
    result = RetrievalEvaluationRunner(pipeline=pipe, harness=_harness()).run(
        list(case_set.cases)
    )
    assert result.hit_rate == 0.5 and result.recall == 0.5  # real macro average
    assert case_set.to_dict()  # CaseSet merely described input; it scored nothing


# --- Step 16: minimal real E2E through a real pipeline ---


def _needle_pipeline():
    doc = Document(id="doc-a", content="needle " + "wool " * 12)
    chunker = FixedSizeChunker(chunk_size=12)
    embedder = DeterministicEmbedder(dimension=8)
    vstore, lindex = InMemoryVectorStore(), InMemoryLexicalIndex()
    hybrid = HybridRetriever(
        DenseRetriever(embedder, vstore),
        LexicalRetriever(lindex),
        WeightedScoreFusion(),
    )
    pipe = DeterministicRAGPipeline(
        chunker=chunker,
        embedder=embedder,
        vector_store=vstore,
        lexical_index=lindex,
        retriever=hybrid,
        reranker=DeterministicReranker(),
        context_builder=DeterministicContextBuilder(),
    )
    pipe.ingest(doc)
    # chunk "doc-a::0" is the first window and the only one holding "needle".
    chunks = chunker.chunk(doc)
    assert chunks[0].id == "doc-a::0" and "needle" in chunks[0].text
    return pipe


def test_real_pipeline_case_set_is_pure_input_contract():
    pipe = _needle_pipeline()
    case_set = EvaluationCaseSet(
        cases=(
            _case("needle", "doc-a::0"),          # present -> a hit
            _case("missing", "does-not-exist"),   # absent  -> a miss
        )
    )
    # CaseSet.cases flows into the runner; metrics are the harness+evaluator's.
    result = RetrievalEvaluationRunner(pipeline=pipe, harness=_harness()).run(
        list(case_set.cases), top_k=10, candidate_k=10
    )
    assert isinstance(result, EvaluationResult)
    assert (result.total_cases, result.evaluated_cases) == (2, 2)
    assert result.hit_rate == 0.5  # needle hit, missing miss
    assert result.recall == 0.5  # macro average of 1.0 and 0.0
    # Deterministic re-run over the same immutable input.
    again = RetrievalEvaluationRunner(pipeline=pipe, harness=_harness()).run(
        list(case_set.cases), top_k=10, candidate_k=10
    )
    assert again == result
    assert result.to_dict() == again.to_dict()


def test_public_import():
    from zmwiki import __all__

    assert "EvaluationCaseSet" in __all__
    case_set = EvaluationCaseSet(cases=(_case("q", "A"),))
    assert isinstance(case_set, EvaluationCaseSet)
