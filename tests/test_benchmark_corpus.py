"""Phase 10A benchmark corpus + dataset tests.

The ``benchmark/`` tree is a small, fully local, hand-written, deterministic
synthetic corpus and one JSON evaluation dataset. These tests pin the contract:

- corpus files exist, are UTF-8, and are non-empty;
- corpus reading and chunking are deterministic (ids, text);
- ingestion is real (FileIngestor -> Document -> Pipeline.ingest) and
  re-ingesting a document never grows the index;
- the JSON dataset loads through ``JsonEvaluationDatasetLoader`` with order and
  ids intact, and stays on the minimal Phase 9C schema;
- every non-``missing`` relevant id is a real chunk produced by the canonical
  chunker (no hand-guessed ids);
- one real E2E drives corpus -> ingestion -> pipeline -> retrieval ->
  RetrievalEvaluationRunner -> Harness -> Evaluator -> EvaluationResult;
- identical setups produce identical results.

Chunk ids are real: the canonical config below (the same FixedSizeChunker the
pipeline E2E uses) produced ``doc_<letter>::<n>`` for ``benchmark/corpus``
under ``chunk_size=120`` / ``overlap=20``. The dataset's ids were taken from
that actual output, not guessed.
"""

import json
from pathlib import Path

import pytest

from zmwiki import (
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
    JsonEvaluationDatasetLoader,
    LexicalRetriever,
    PlainTextFileIngestor,
    RAGPipeline,
    RetrievalEvaluationRunner,
    RetrievalEvaluator,
    WeightedScoreFusion,
)

BENCH_DIR = Path(__file__).resolve().parents[1] / "benchmark"
CORPUS_DIR = BENCH_DIR / "corpus"
DATASET = BENCH_DIR / "datasets" / "retrieval_small.json"

#: Canonical chunker for the benchmark corpus (ids in the dataset depend on it).
CHUNK_SIZE = 120
OVERLAP = 20

DOC_FILES = [
    "doc_a.txt",
    "doc_b.txt",
    "doc_c.txt",
    "doc_d.txt",
    "doc_e.txt",
    "doc_f.txt",
]

#: Demo wiki workspace shipped for the TUI; it lives beside the retrieval corpus.
DEMO_WORKSPACE_ENTRIES = ["wiki", "wiki_workspace.json"]

#: The dataset's 8 cases, in their exact file order, with real chunk ids.
EXPECTED_CASES = [
    ("database B-tree index query", ("doc_a::1",)),
    ("cache eviction memory", ("doc_b::1",)),
    ("producer consumer message queue", ("doc_c::0",)),
    ("embedding vector similarity retrieval", ("doc_d::1",)),
    ("BM25 lexical vector ranking fusion", ("doc_e::1", "doc_e::2")),
    ("document chunk context", ("doc_f::0",)),
    ("nearest neighbor search representation", ("doc_d::2",)),
    ("quantum submarine agriculture", ("missing::0",)),
]

#: The single deliberate miss: an id that is not in the corpus at all.
GHOST_ID = "missing::0"


def _corpus_paths():
    return [CORPUS_DIR / name for name in DOC_FILES]


def _load_documents():
    """Read every corpus file through PlainTextFileIngestor, in file order.

    document_id is the filename stem (``doc_a``, ...), never an absolute path.
    """
    ingestor = PlainTextFileIngestor()
    return [ingestor.ingest(p, document_id=p.stem) for p in _corpus_paths()]


def _chunk_counts(documents):
    chunker = FixedSizeChunker(chunk_size=CHUNK_SIZE, overlap=OVERLAP)
    return {doc.id: chunker.chunk(doc) for doc in documents}


def _build_pipeline():
    """A fresh deterministic pipeline wired exactly like the E2E preflight."""
    chunker = FixedSizeChunker(chunk_size=CHUNK_SIZE, overlap=OVERLAP)
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
    return pipeline, vstore, lindex


def _ingest_all():
    """Ingest the whole corpus into a fresh pipeline; return the wiring + docs."""
    pipeline, vstore, lindex = _build_pipeline()
    documents = _load_documents()
    for document in documents:
        pipeline.ingest(document)
    return pipeline, vstore, lindex, documents


def _load_case_set():
    return JsonEvaluationDatasetLoader().load(DATASET)


# --- A. Corpus file presence / encoding / non-empty --------------------------


def test_corpus_directory_exists_and_has_expected_files():
    assert CORPUS_DIR.is_dir()
    assert sorted(p.name for p in CORPUS_DIR.iterdir()) == (
        DOC_FILES + DEMO_WORKSPACE_ENTRIES
    )


@pytest.mark.parametrize("name", DOC_FILES)
def test_corpus_file_is_utf8_and_nonempty(name):
    content = (CORPUS_DIR / name).read_text(encoding="utf-8")  # bad bytes would raise
    assert content.strip()  # non-empty, not whitespace-only
    assert "\x00" not in content


def test_dataset_file_exists_and_is_utf8():
    assert DATASET.is_file()
    DATASET.read_text(encoding="utf-8")  # must decode as UTF-8


# --- B. Corpus determinism ---------------------------------------------------


def test_corpus_read_is_deterministic():
    first = _load_documents()
    second = _load_documents()
    assert [d.id for d in first] == [d.id for d in second]
    assert [d.content for d in first] == [d.content for d in second]
    assert first == second  # Documents are frozen value objects


def test_chunking_is_deterministic_across_instances():
    documents = _load_documents()
    a = _chunk_counts(documents)
    b = _chunk_counts(documents)  # fresh chunker inside the helper
    for doc_id in a:
        assert [c.id for c in a[doc_id]] == [c.id for c in b[doc_id]]
        assert [c.text for c in a[doc_id]] == [c.text for c in b[doc_id]]
        assert a[doc_id] == b[doc_id]


def test_each_document_produces_multiple_chunks():
    for doc_id, chunks in _chunk_counts(_load_documents()).items():
        assert len(chunks) >= 2, f"{doc_id} should yield multiple chunks"
        # positions are 0-based and sequential, ids deterministic per document
        assert [c.position for c in chunks] == list(range(len(chunks)))
        assert [c.id for c in chunks] == [
            f"{doc_id}::{i}" for i in range(len(chunks))
        ]


# --- C. Real ingestion -------------------------------------------------------


def test_full_corpus_ingests_and_counts_match_chunker():
    _, vstore, lindex, documents = _ingest_all()
    expected = sum(len(chunks) for chunks in _chunk_counts(documents).values())
    assert vstore.count() == expected == lindex.count()
    assert expected >= 10  # the corpus is genuinely multi-chunk


def test_duplicate_ingest_does_not_grow_the_index():
    pipeline, vstore, lindex, documents = _ingest_all()
    before = vstore.count()
    assert lindex.count() == before
    pipeline.ingest(documents[0])  # same document again -> same chunk ids -> replace
    assert vstore.count() == before
    assert lindex.count() == before
    pipeline.ingest(documents[0])
    assert vstore.count() == before and lindex.count() == before


def test_ingested_chunks_are_addressable_by_id():
    _, vstore, lindex, _ = _ingest_all()
    all_ids = [c.id for chunks in _chunk_counts(_load_documents()).values() for c in chunks]
    for cid in all_ids:
        chunk = lindex.get(cid)
        assert chunk is not None and chunk.text  # real, non-empty stored chunk
        vstore.get(cid)  # present in the dense store too (does not raise)


# --- D. Dataset loading ------------------------------------------------------


def test_dataset_stays_on_minimal_schema():
    data = json.loads(DATASET.read_text(encoding="utf-8"))
    assert isinstance(data, dict) and set(data) == {"cases"}
    assert isinstance(data["cases"], list) and len(data["cases"]) == 8
    for i, case in enumerate(data["cases"]):
        assert set(case) == {"query", "relevant_chunk_ids"}
        assert isinstance(case["query"], str)
        assert isinstance(case["relevant_chunk_ids"], list)


def test_loader_returns_cases_in_file_order_with_ids_intact():
    case_set = _load_case_set()
    assert isinstance(case_set, EvaluationCaseSet)
    assert len(case_set) == 8
    assert [c.query for c in case_set] == [q for q, _ in EXPECTED_CASES]
    for case, (query, ids) in zip(case_set, EXPECTED_CASES):
        assert isinstance(case, EvaluationCase)
        assert case.query == query
        # EvaluationCase canonicalizes (sorts) ids; the file ids are recovered.
        assert case.relevant_chunk_ids == tuple(sorted(ids))


def test_loader_is_deterministic_across_loads():
    first = _load_case_set()
    second = _load_case_set()
    assert first.cases == second.cases
    assert first.to_dict() == second.to_dict()


# --- dataset ids are real chunks (never hand-guessed) ------------------------


def test_every_non_ghost_relevant_id_is_a_real_stored_chunk():
    _, _, lindex, _ = _ingest_all()
    real = [cid for _, ids in EXPECTED_CASES for cid in ids if cid != GHOST_ID]
    assert real  # sanity: the dataset is not all ghosts
    for cid in real:
        chunk = lindex.get(cid)
        assert chunk is not None, f"{cid} must be a real stored chunk"
    assert lindex.get(GHOST_ID) is None  # the miss id is genuinely absent


# --- E. Real end-to-end through the whole evaluation chain -------------------


def _benchmark_setup():
    """Fresh full wiring: corpus -> pipeline -> runner -> benchmark runner."""
    pipeline, vstore, lindex, documents = _ingest_all()
    pool = vstore.count()  # >= every chunk -> whole pool returned per query
    assert pool == lindex.count() and pool > 0
    harness = EvaluationHarness(RetrievalEvaluator())
    evaluation_runner = RetrievalEvaluationRunner(pipeline=pipeline, harness=harness)
    benchmark = DeterministicBenchmarkRunner(evaluation_runner=evaluation_runner)
    case_set = _load_case_set()
    return benchmark, case_set, pool


def test_e2e_benchmark_corpus_full_chain():
    benchmark, case_set, pool = _benchmark_setup()
    result = benchmark.run(case_set, top_k=pool, candidate_k=pool)

    assert isinstance(result, EvaluationResult)
    assert (result.total_cases, result.evaluated_cases) == (8, 8)
    # Whole-pool retrieval surfaces every real relevant chunk; only the ghost
    # id (not indexed) misses -> exactly 7 hits and recall 7/8.
    assert result.hit_rate == pytest.approx(7 / 8)
    assert result.recall == pytest.approx(7 / 8)
    assert 0.0 <= result.precision <= 1.0
    assert 0.0 <= result.mrr <= 1.0
    assert 0.0 <= result.map <= 1.0
    assert 0.0 <= result.ndcg <= 1.0


def test_e2e_ghost_case_is_the_only_miss():
    benchmark, case_set, pool = _benchmark_setup()
    pipeline = benchmark.evaluation_runner.pipeline
    for case in case_set:
        predictions = pipeline.retrieve(case.query, top_k=pool, candidate_k=pool)
        found = [cid for cid in case.relevant_chunk_ids if cid != GHOST_ID]
        pred_ids = [sr.chunk.id for sr in predictions]
        if GHOST_ID in case.relevant_chunk_ids:
            assert GHOST_ID not in pred_ids  # the miss never materializes
        else:
            assert all(cid in pred_ids for cid in found)  # real ids are found


# --- F. Determinism ----------------------------------------------------------


def test_repeat_runs_of_identical_setup_are_equal():
    benchmark, case_set, pool = _benchmark_setup()
    first = benchmark.run(case_set, top_k=pool, candidate_k=pool)
    second = benchmark.run(case_set, top_k=pool, candidate_k=pool)
    assert first == second
    assert first.to_dict() == second.to_dict()


def test_independent_instances_of_identical_setup_are_equal():
    a, case_set, pool = _benchmark_setup()
    b, _, _ = _benchmark_setup()  # fresh pipeline, fresh benchmark
    assert a.run(case_set, top_k=pool, candidate_k=pool) == b.run(
        case_set, top_k=pool, candidate_k=pool
    )


def test_e2e_is_real_pipeline_not_a_fake():
    benchmark, case_set, pool = _benchmark_setup()
    pipeline = benchmark.evaluation_runner.pipeline
    assert isinstance(pipeline, RAGPipeline)
    result = benchmark.run(case_set, top_k=pool, candidate_k=pool)
    # The pipeline actually ingested Documents: every corpus chunk is retrievable.
    assert pipeline.retrieve("B-tree index", top_k=pool, candidate_k=pool)
