"""ZMRAG — a model-agnostic Knowledge / Retrieval layer.

Public API (kept deliberately small): the core data contract. Internal
modules are not re-exported until they become stable public surface.
"""

from .benchmark import BenchmarkRunner, DeterministicBenchmarkRunner
from .chunkers import Chunker, FixedSizeChunker
from .citations import AnswerResult, Citation, citations_from_context
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
from .generators import DeterministicGenerator, Generator, build_prompt
from .ingestion import PlainTextIngestor, TextIngestor
from .lexical import InMemoryLexicalIndex, LexicalIndex
from .pipeline import DeterministicRAGPipeline, RAGPipeline
from .rerankers import DeterministicReranker, Reranker
from .retrievers import DenseRetriever, HybridRetriever, LexicalRetriever, Retriever
from .rewriters import DeterministicQueryRewriter, QueryRewriter
from .schema import Chunk, Document, Metadata, Source
from .stores import InMemoryVectorStore, SearchResult, VectorStore
from .tokenizers import SimpleTokenizer, Tokenizer

__all__ = [
    "AnswerResult",
    "BenchmarkRunner",
    "Chunk",
    "Chunker",
    "Citation",
    "Context",
    "ContextBuilder",
    "DenseRetriever",
    "DeterministicBenchmarkRunner",
    "DeterministicContextBuilder",
    "DeterministicEmbedder",
    "DeterministicGenerator",
    "DeterministicQueryRewriter",
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
    "Generator",
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
    "QueryRewriter",
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
    "build_prompt",
    "citations_from_context",
]

__version__ = "0.1.0"
