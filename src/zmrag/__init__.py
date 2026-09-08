"""ZMRAG — a model-agnostic Knowledge / Retrieval layer.

Public API (kept deliberately small): the core data contract. Internal
modules are not re-exported until they become stable public surface.
"""

from .benchmark import BenchmarkRunner, DeterministicBenchmarkRunner
from .chunkers import Chunker, FixedSizeChunker
from .context import Context, ContextBuilder, DeterministicContextBuilder
from .directory_ingestion import DirectoryIngestor, PlainTextDirectoryIngestor
from .embedders import DeterministicEmbedder, Embedding, EmbeddingProvider
from .evaluation import (
    EvaluationCase,
    EvaluationPrediction,
    EvaluationResult,
    Evaluator,
    RetrievalEvaluator,
)
from .evaluation_dataset import EvaluationCaseSet
from .evaluation_harness import EvaluationHarness
from .evaluation_loader import EvaluationDatasetLoader, JsonEvaluationDatasetLoader
from .evaluation_runner import RetrievalEvaluationRunner
from .file_ingestion import FileIngestor, FileSource, PlainTextFileIngestor
from .fusion import FusionStrategy, RRFFusion, WeightedScoreFusion
from .ingestion import PlainTextIngestor, TextIngestor
from .lexical import InMemoryLexicalIndex, LexicalIndex
from .pipeline import DeterministicRAGPipeline, RAGPipeline
from .rerankers import DeterministicReranker, Reranker
from .retrievers import DenseRetriever, HybridRetriever, LexicalRetriever, Retriever
from .schema import Chunk, Document, Metadata, Source
from .stores import InMemoryVectorStore, SearchResult, VectorStore
from .tokenizers import SimpleTokenizer, Tokenizer

__all__ = [
    "BenchmarkRunner",
    "Chunk",
    "Chunker",
    "Context",
    "ContextBuilder",
    "DenseRetriever",
    "DeterministicBenchmarkRunner",
    "DeterministicContextBuilder",
    "DeterministicEmbedder",
    "DeterministicRAGPipeline",
    "DeterministicReranker",
    "Document",
    "DirectoryIngestor",
    "Embedding",
    "EmbeddingProvider",
    "EvaluationCase",
    "EvaluationCaseSet",
    "EvaluationDatasetLoader",
    "EvaluationHarness",
    "EvaluationPrediction",
    "EvaluationResult",
    "Evaluator",
    "FileIngestor",
    "FileSource",
    "FixedSizeChunker",
    "FusionStrategy",
    "HybridRetriever",
    "InMemoryLexicalIndex",
    "InMemoryVectorStore",
    "JsonEvaluationDatasetLoader",
    "LexicalIndex",
    "LexicalRetriever",
    "Metadata",
    "PlainTextDirectoryIngestor",
    "PlainTextFileIngestor",
    "PlainTextIngestor",
    "RAGPipeline",
    "RRFFusion",
    "Reranker",
    "Retriever",
    "RetrievalEvaluationRunner",
    "RetrievalEvaluator",
    "SearchResult",
    "SimpleTokenizer",
    "Source",
    "TextIngestor",
    "Tokenizer",
    "VectorStore",
    "WeightedScoreFusion",
]

__version__ = "0.1.0"
