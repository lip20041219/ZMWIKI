"""Evaluation dataset loader tests: local JSON -> EvaluationCaseSet.

Covers the Phase 9C EvaluationDatasetLoader: the ABC contract, minimal JSON
loading, strict top-level / case schema validation (ValueError for structure,
while EvaluationCase business constraints propagate untouched), strict file
errors, deterministic re-loads across calls and instances, immutability of the
returned CaseSet, JSON round-tripping through ``to_dict()``, the public import,
and a real E2E proving JSON -> loader -> CaseSet -> RetrievalEvaluationRunner ->
EvaluationHarness -> RetrievalEvaluator -> EvaluationResult.
"""

import json

import pytest

from zmwiki import (
    DeterministicContextBuilder,
    DeterministicEmbedder,
    DeterministicRAGPipeline,
    DeterministicReranker,
    DenseRetriever,
    Document,
    EvaluationCase,
    EvaluationCaseSet,
    EvaluationDatasetLoader,
    EvaluationHarness,
    EvaluationResult,
    FixedSizeChunker,
    HybridRetriever,
    InMemoryLexicalIndex,
    InMemoryVectorStore,
    JsonEvaluationDatasetLoader,
    LexicalRetriever,
    RetrievalEvaluationRunner,
    RetrievalEvaluator,
    WeightedScoreFusion,
)


def _loader():
    return JsonEvaluationDatasetLoader()


def _write(tmp_path, payload, name="cases.json"):
    p = tmp_path / name
    p.write_text(json.dumps(payload), encoding="utf-8")
    return p


# --- A. ABC ---


def test_evaluation_dataset_loader_is_abstract():
    with pytest.raises(TypeError):
        EvaluationDatasetLoader()  # type: ignore[abstract]


def test_json_loader_is_instantiable():
    assert isinstance(_loader(), EvaluationDatasetLoader)
    assert isinstance(_loader(), JsonEvaluationDatasetLoader)


# --- B. Basic loading ---


def test_load_minimal_legal_json(tmp_path):
    path = _write(tmp_path, {"cases": [{"query": "needle", "relevant_chunk_ids": ["A"]}]})
    case_set = _loader().load(path)
    assert isinstance(case_set, EvaluationCaseSet)
    assert len(case_set) == 1
    assert case_set.cases[0] == EvaluationCase(query="needle", relevant_chunk_ids=["A"])


def test_load_multiple_cases(tmp_path):
    path = _write(
        tmp_path,
        {
            "cases": [
                {"query": "q1", "relevant_chunk_ids": ["A"]},
                {"query": "q2", "relevant_chunk_ids": ["B", "C"]},
                {"query": "q3", "relevant_chunk_ids": ["D"]},
            ]
        },
    )
    case_set = _loader().load(path)
    assert len(case_set) == 3
    assert [c.query for c in case_set] == ["q1", "q2", "q3"]


def test_load_empty_cases(tmp_path):
    case_set = _loader().load(_write(tmp_path, {"cases": []}))
    assert case_set == EvaluationCaseSet(cases=())
    assert len(case_set) == 0 and case_set.cases == ()


def test_load_multiple_relevant_ids(tmp_path):
    path = _write(tmp_path, {"cases": [{"query": "q", "relevant_chunk_ids": ["A", "B", "C"]}]})
    case_set = _loader().load(path)
    assert case_set.cases[0].relevant_chunk_ids == ("A", "B", "C")


# --- C. Ordering ---


def test_cases_order_strictly_preserved(tmp_path):
    path = _write(
        tmp_path,
        {
            "cases": [
                {"query": "z", "relevant_chunk_ids": ["A"]},
                {"query": "a", "relevant_chunk_ids": ["B"]},
                {"query": "m", "relevant_chunk_ids": ["C"]},
            ]
        },
    )
    case_set = _loader().load(path)
    assert [c.query for c in case_set] == ["z", "a", "m"]  # never sorted


def test_no_sorting_by_any_field(tmp_path):
    # Queries and ids both out of order; the loader must not reorder cases and
    # must not apply a second sort on top of EvaluationCase canonicalization.
    path = _write(
        tmp_path,
        {"cases": [{"query": "z", "relevant_chunk_ids": ["z::1"]},
                   {"query": "a", "relevant_chunk_ids": ["z::0"]}]},
    )
    case_set = _loader().load(path)
    assert [c.query for c in case_set] == ["z", "a"]
    # EvaluationCase canonicalizes each case's own ids once (its contract);
    # loader adds no ordering of its own.
    assert case_set.cases[1].relevant_chunk_ids == ("z::0",)


def test_duplicate_cases_preserved(tmp_path):
    path = _write(
        tmp_path,
        {"cases": [
            {"query": "needle", "relevant_chunk_ids": ["A"]},
            {"query": "needle", "relevant_chunk_ids": ["A"]},
        ]},
    )
    case_set = _loader().load(path)
    assert len(case_set) == 2
    assert case_set.cases[0] == case_set.cases[1]


# --- D. EvaluationCase delegation (schema-legal, business-invalid) ---


@pytest.mark.parametrize("bad_query", ["", "   \n\t"])
def test_empty_or_whitespace_query_rejected(tmp_path, bad_query):
    path = _write(tmp_path, {"cases": [{"query": bad_query, "relevant_chunk_ids": ["A"]}]})
    with pytest.raises(TypeError):  # EvaluationCase's own _nonempty_str rule
        _loader().load(path)


def test_empty_relevant_ids_rejected(tmp_path):
    path = _write(tmp_path, {"cases": [{"query": "q", "relevant_chunk_ids": []}]})
    with pytest.raises(ValueError):  # EvaluationCase requires a non-empty set
        _loader().load(path)


def test_duplicate_relevant_ids_rejected(tmp_path):
    path = _write(tmp_path, {"cases": [{"query": "q", "relevant_chunk_ids": ["A", "A"]}]})
    with pytest.raises(ValueError):  # EvaluationCase rejects duplicates
        _loader().load(path)


@pytest.mark.parametrize("bad_id", ["", "   "])
def test_blank_relevant_id_rejected(tmp_path, bad_id):
    path = _write(tmp_path, {"cases": [{"query": "q", "relevant_chunk_ids": ["A", bad_id]}]})
    with pytest.raises(ValueError):  # EvaluationCase rejects blank ids
        _loader().load(path)


# --- E. Top-level schema (-> ValueError) ---


@pytest.mark.parametrize(
    "payload",
    [
        ["not", "an", "object"],
        "just a string",
        42,
        None,
    ],
)
def test_top_level_not_an_object(tmp_path, payload):
    with pytest.raises(ValueError):
        _loader().load(_write(tmp_path, payload))


def test_top_level_missing_cases(tmp_path):
    with pytest.raises(ValueError):
        _loader().load(_write(tmp_path, {"other": 1}))


def test_cases_not_a_list(tmp_path):
    with pytest.raises(ValueError):
        _loader().load(_write(tmp_path, {"cases": {"query": "q", "relevant_chunk_ids": []}}))


def test_unknown_top_level_field_rejected(tmp_path):
    with pytest.raises(ValueError):
        _loader().load(
            _write(tmp_path, {"cases": [], "foo": "bar"})
        )


# --- F. Case schema (-> ValueError) ---


def test_case_not_an_object(tmp_path):
    with pytest.raises(ValueError):
        _loader().load(_write(tmp_path, {"cases": ["q"]}))


def test_case_missing_query(tmp_path):
    with pytest.raises(ValueError):
        _loader().load(_write(tmp_path, {"cases": [{"relevant_chunk_ids": ["A"]}]}))


def test_case_missing_relevant_chunk_ids(tmp_path):
    with pytest.raises(ValueError):
        _loader().load(_write(tmp_path, {"cases": [{"query": "q"}]}))


@pytest.mark.parametrize("bad_query", [5, 1.5, None, True, ["q"], {"q": 1}])
def test_query_not_a_string(tmp_path, bad_query):
    with pytest.raises(ValueError):
        _loader().load(
            _write(tmp_path, {"cases": [{"query": bad_query, "relevant_chunk_ids": ["A"]}]})
        )


@pytest.mark.parametrize("bad_relevant", ["A", {"A": 1}, 5, None])
def test_relevant_chunk_ids_not_a_list(tmp_path, bad_relevant):
    with pytest.raises(ValueError):
        _loader().load(
            _write(tmp_path, {"cases": [{"query": "q", "relevant_chunk_ids": bad_relevant}]})
        )


@pytest.mark.parametrize("bad_id", [5, None, True, ["A"]])
def test_relevant_id_not_a_string(tmp_path, bad_id):
    with pytest.raises(ValueError):
        _loader().load(
            _write(tmp_path, {"cases": [{"query": "q", "relevant_chunk_ids": ["A", bad_id]}]})
        )


def test_unknown_case_field_rejected(tmp_path):
    with pytest.raises(ValueError):
        _loader().load(
            _write(
                tmp_path,
                {"cases": [{"query": "q", "relevant_chunk_ids": ["A"], "score": 1.0}]},
            )
        )


# --- G. File errors ---


def test_missing_file_raises_file_not_found(tmp_path):
    with pytest.raises(FileNotFoundError):
        _loader().load(tmp_path / "nope.json")


def test_directory_path_raises_is_a_directory(tmp_path):
    with pytest.raises(IsADirectoryError):
        _loader().load(tmp_path)


@pytest.mark.parametrize("bad", [None, 5, 3.5, True, [], {}, b"\x00bytes"])
def test_invalid_path_type_rejected(tmp_path, bad):
    with pytest.raises(TypeError):
        _loader().load(bad)  # type: ignore[arg-type]


def test_invalid_utf8_raises_unicode_decode_error(tmp_path):
    p = tmp_path / "bad.txt"
    p.write_bytes(b"\xff\xfe\x00{\"cases\"\x00\xff")
    with pytest.raises(UnicodeDecodeError):
        _loader().load(p)


def test_malformed_json_raises_decode_error(tmp_path):
    p = _write(tmp_path, {"cases": []})
    p.write_text('{"cases": [', encoding="utf-8")
    with pytest.raises(json.JSONDecodeError):
        _loader().load(p)


# --- H. Determinism ---


def test_same_loader_repeated_load_is_equal(tmp_path):
    path = _write(
        tmp_path,
        {"cases": [{"query": "q1", "relevant_chunk_ids": ["A"]},
                   {"query": "q2", "relevant_chunk_ids": ["B", "C"]}]},
    )
    loader = _loader()
    assert loader.load(path) == loader.load(path)
    assert loader.load(path).to_dict() == loader.load(path).to_dict()


def test_different_loader_instances_are_equal(tmp_path):
    path = _write(tmp_path, {"cases": [{"query": "q1", "relevant_chunk_ids": ["A"]}]})
    assert _loader().load(path) == _loader().load(path)


# --- I. Immutability ---


def test_returned_case_set_is_immutable(tmp_path):
    case_set = _loader().load(
        _write(tmp_path, {"cases": [{"query": "q", "relevant_chunk_ids": ["A"]}]})
    )
    with pytest.raises(AttributeError):
        case_set.cases = ()  # type: ignore[misc]  # frozen dataclass
    with pytest.raises(AttributeError):
        case_set.cases.append(1)  # type: ignore[attr-defined]  # tuple


def test_cases_is_a_tuple(tmp_path):
    case_set = _loader().load(
        _write(tmp_path, {"cases": [{"query": "q", "relevant_chunk_ids": ["A"]}]})
    )
    assert isinstance(case_set.cases, tuple)


def test_reload_yields_fresh_equal_case_objects(tmp_path):
    # No caching / shared identity: each load builds fresh EvaluationCase
    # values that are equal (deterministic) but distinct objects.
    path = _write(tmp_path, {"cases": [{"query": "q", "relevant_chunk_ids": ["A"]}]})
    first = _loader().load(path)
    second = _loader().load(path)
    assert first.cases[0] == second.cases[0]
    assert first.cases[0] is not second.cases[0]
    assert first.cases[0].query == "q"
    assert first.cases[0].relevant_chunk_ids == ("A",)


# --- J. JSON round trip ---


def test_load_to_dict_round_trips_input(tmp_path):
    payload = {
        "cases": [
            {"query": "needle", "relevant_chunk_ids": ["doc-a::0"]},
            {"query": "missing", "relevant_chunk_ids": ["doc-b::0", "doc-c::1"]},
        ]
    }
    path = _write(tmp_path, payload)
    case_set = _loader().load(path)
    assert case_set.to_dict() == payload  # ids already canonical -> lossless


def test_to_dict_is_json_serializable(tmp_path):
    path = _write(tmp_path, {"cases": [{"query": "q", "relevant_chunk_ids": ["A"]}]})
    case_set = _loader().load(path)
    text = json.dumps(case_set.to_dict())
    assert json.loads(text) == case_set.to_dict()


# --- K. Public API ---


def test_public_import_loader_types():
    from zmwiki import __all__

    assert "EvaluationDatasetLoader" in __all__
    assert "JsonEvaluationDatasetLoader" in __all__
    assert issubclass(JsonEvaluationDatasetLoader, EvaluationDatasetLoader)


# --- L. Real E2E ---


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
    chunks = chunker.chunk(doc)
    assert chunks[0].id == "doc-a::0" and "needle" in chunks[0].text
    return pipe


def test_real_e2e_json_file_through_full_evaluation_chain(tmp_path):
    path = _write(
        tmp_path,
        {
            "cases": [
                {"query": "needle", "relevant_chunk_ids": ["doc-a::0"]},   # hit
                {"query": "missing", "relevant_chunk_ids": ["does-not-exist"]},  # miss
            ]
        },
    )
    case_set = JsonEvaluationDatasetLoader().load(path)
    assert isinstance(case_set, EvaluationCaseSet)
    assert len(case_set) == 2

    harness = EvaluationHarness(RetrievalEvaluator())
    pipe = _needle_pipeline()
    runner = RetrievalEvaluationRunner(pipeline=pipe, harness=harness)
    result = runner.run(list(case_set.cases), top_k=10, candidate_k=10)

    assert isinstance(result, EvaluationResult)
    assert (result.total_cases, result.evaluated_cases) == (2, 2)
    assert result.hit_rate == 0.5  # needle hit, missing miss
    assert result.recall == 0.5  # macro average of 1.0 and 0.0
    # The needle chunk is the unique lexical match, so it ranks first:
    # MRR / MAP / NDCG average 0.5 over the hit + the empty miss.
    assert result.mrr == 0.5
    assert result.map == 0.5
    assert result.ndcg == 0.5

    # Deterministic re-run of the same loaded (immutable) input.
    again = runner.run(list(case_set.cases), top_k=10, candidate_k=10)
    assert again == result
    assert again.to_dict() == result.to_dict()
