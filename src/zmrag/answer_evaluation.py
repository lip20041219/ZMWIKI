"""Answer evaluation: score an answer against a reference, its context, its sources.

The other half of ZMRAG's evaluation story. Retrieval Evaluation (``evaluation.py``)
answers "did we retrieve the right context?"; this module answers "did the
generated answer cover the expected answer, and did it stay grounded in the
context it was given?". They are independent: this layer never retrieves,
reranks, packs context, generates, or reads a retrievable store -- it scores one
already-produced answer and nothing else.

**What ships here is three deterministic lexical baselines, not a judge.**
Nothing in this module understands meaning:

- **lexical reference coverage** -- the share of the reference answer's tokens
  that appear in the answer. A paraphrase with different words scores low even
  when it is correct; that is the known, documented ceiling of a token-overlap
  baseline, not a bug to paper over with a synonym list.
- **lexical groundedness** -- the share of the answer's sentences whose tokens
  are mostly found in the context. It cannot detect negation or contradiction:
  a sentence built from context words scores as grounded even when it asserts
  the opposite of what the context says.
- **source-level citation coverage** -- the same sentence measure, computed
  against the *cited source texts* rather than the whole context. It is
  source-level, not claim-level: it never attributes a particular sentence to a
  particular source, because a ``Citation`` carries no such information.

No embeddings, no model, no network, no third-party library, no LLM judge. The
names above are deliberate and are used verbatim in the README and in
``docs/architecture.md``: calling any of this "semantic correctness" or
"faithfulness" would overstate what a token overlap can see.

Token identity comes from the existing :class:`~zmrag.tokenizers.SimpleTokenizer`
(plus NFC normalization), so case, whitespace, and punctuation differences never
count as differences, and CJK text matches at character granularity.
"""

from __future__ import annotations

import unicodedata
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from .citations import Citation
from .context import Context
from .schema import _nonempty_str
from .tokenizers import SimpleTokenizer, Tokenizer

__all__ = [
    "DEFAULT_GROUNDEDNESS_THRESHOLD",
    "AnswerEvaluation",
    "AnswerEvaluator",
    "aggregate_evaluations",
]

#: A sentence counts as grounded when at least this share of its tokens appear
#: in the support text. Fixed here, never tuned against a dataset: it is a
#: stated baseline convention, and every evidence row records the value used.
DEFAULT_GROUNDEDNESS_THRESHOLD = 0.5

#: Characters that end a sentence. Deliberately naive: decimals and
#: abbreviations split too, which can only make the sentence count larger (and
#: each sentence shorter). Recorded as a limitation rather than patched with a
#: language-specific rule set this project does not have.
_SENTENCE_TERMINATORS = frozenset(".!?;。！？；\n")


@dataclass(frozen=True)
class AnswerEvaluation:
    """One answer's scores. Immutable, JSON-ready, no object references.

    ``query_id`` is the dataset's own identifier for the case -- the join key
    back to retrieval results. The three rates are floats in [0, 1] and are
    never rounded, so a report can round them however it likes. The counters
    exist so a score can be audited without re-running anything: a coverage of
    0.5 over 2 reference tokens is a different situation from 0.5 over 40.

    Empty-input conventions (all explicit, no NaN can be produced):

    - empty answer (or whitespace only) -> ``has_answer`` False, every rate 0.0
      and zero sentences, because there is nothing to ground or to match;
    - empty context -> ``groundedness`` 0.0, since no sentence can be supported;
    - no sources -> ``citation_coverage`` 0.0 and ``has_sources`` False.
    """

    query_id: str
    reference_coverage: float
    groundedness: float
    citation_coverage: float
    has_answer: bool
    has_sources: bool
    source_count: int
    answer_sentences: int
    grounded_sentences: int
    cited_sentences: int
    reference_tokens: int
    matched_reference_tokens: int

    def __post_init__(self) -> None:
        _nonempty_str(self.query_id, "query_id")
        for name, value in (
            ("reference_coverage", self.reference_coverage),
            ("groundedness", self.groundedness),
            ("citation_coverage", self.citation_coverage),
        ):
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"{name} must be a float, got {type(value).__name__}")
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1], got {value!r}")
        for name, value in (
            ("has_answer", self.has_answer),
            ("has_sources", self.has_sources),
        ):
            if not isinstance(value, bool):
                raise TypeError(f"{name} must be a bool, got {type(value).__name__}")
        for name, value in (
            ("source_count", self.source_count),
            ("answer_sentences", self.answer_sentences),
            ("grounded_sentences", self.grounded_sentences),
            ("cited_sentences", self.cited_sentences),
            ("reference_tokens", self.reference_tokens),
            ("matched_reference_tokens", self.matched_reference_tokens),
        ):
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an int, got {type(value).__name__}")
            if value < 0:
                raise ValueError(f"{name} must be >= 0, got {value}")
        if self.grounded_sentences > self.answer_sentences:
            raise ValueError("grounded_sentences cannot exceed answer_sentences")
        if self.cited_sentences > self.answer_sentences:
            raise ValueError("cited_sentences cannot exceed answer_sentences")
        if self.matched_reference_tokens > self.reference_tokens:
            raise ValueError("matched_reference_tokens cannot exceed reference_tokens")
        if self.has_sources != (self.source_count > 0):
            raise ValueError("has_sources must agree with source_count")

    def to_dict(self) -> dict[str, object]:
        """Stable, JSON-friendly dict with a fixed key order."""
        return {
            "query_id": self.query_id,
            "reference_coverage": self.reference_coverage,
            "groundedness": self.groundedness,
            "citation_coverage": self.citation_coverage,
            "has_answer": self.has_answer,
            "has_sources": self.has_sources,
            "source_count": self.source_count,
            "answer_sentences": self.answer_sentences,
            "grounded_sentences": self.grounded_sentences,
            "cited_sentences": self.cited_sentences,
            "reference_tokens": self.reference_tokens,
            "matched_reference_tokens": self.matched_reference_tokens,
        }

    def summary(self) -> str:
        """Fixed-format, human-readable one-liner block (no timestamp, no paths)."""
        return (
            "## Answer Evaluation\n\n"
            f"Query id: {self.query_id}\n"
            f"Reference coverage: {self.reference_coverage:.6f}\n"
            f"Groundedness: {self.groundedness:.6f}\n"
            f"Citation coverage: {self.citation_coverage:.6f}\n"
            f"Answer present: {self.has_answer}\n"
            f"Sources: {self.source_count}\n"
        )


class AnswerEvaluator:
    """Scores one answer on three deterministic lexical baselines.

    Pure and stateless: the same inputs always produce an equal
    :class:`AnswerEvaluation`, in any process, with no model and no network.

    Definitions, in full:

    - ``reference_coverage`` = ``matched / |unique reference tokens|``, where a
      reference token is matched when the answer contains it. Unique tokens are
      counted once: a word repeated in the reference is one fact, not two.
    - ``groundedness`` = ``grounded sentences / answer sentences``. A sentence is
      grounded when the share of its unique tokens found in the context is at
      least ``groundedness_threshold``. Sentences with no tokens at all cannot
      be grounded.
    - ``citation_coverage`` = the same sentence measure against the concatenated
      ``Citation.text`` of ``sources`` instead of the context. Source-level: it
      says the cited material supports the answer's wording, never that a
      specific sentence came from a specific source.
    """

    def __init__(
        self,
        *,
        tokenizer: Tokenizer | None = None,
        groundedness_threshold: float = DEFAULT_GROUNDEDNESS_THRESHOLD,
    ) -> None:
        self.tokenizer = tokenizer if tokenizer is not None else SimpleTokenizer()
        if not isinstance(self.tokenizer, Tokenizer):
            raise TypeError(
                f"tokenizer must be a Tokenizer, got {type(self.tokenizer).__name__}"
            )
        if isinstance(groundedness_threshold, bool) or not isinstance(
            groundedness_threshold, (int, float)
        ):
            raise TypeError(
                "groundedness_threshold must be a number, got "
                f"{type(groundedness_threshold).__name__}"
            )
        if not 0.0 < groundedness_threshold <= 1.0:
            raise ValueError(
                "groundedness_threshold must be in (0, 1], got "
                f"{groundedness_threshold!r}"
            )
        self.groundedness_threshold = float(groundedness_threshold)

    def evaluate(
        self,
        *,
        query_id: str,
        query: str,
        answer: str,
        context: Context,
        reference_answer: str,
        sources: tuple[Citation, ...] = (),
    ) -> AnswerEvaluation:
        """Score ``answer`` against ``reference_answer``, ``context`` and ``sources``.

        ``query`` is validated but is not an input to any of the three baselines
        -- none of them looks at the question. It stays in the signature because
        the caller always has it, it keeps the evaluator's shape aligned with a
        future query-aware metric, and validating it catches a mismatched call
        early. An empty or whitespace-only ``reference_answer`` is rejected: a
        reference is what coverage is measured against, and a missing one is a
        dataset defect, not a zero score.
        """
        _nonempty_str(query_id, "query_id")
        _nonempty_str(query, "query")
        _nonempty_str(reference_answer, "reference_answer")
        if not isinstance(answer, str):
            raise TypeError(f"answer must be a str, got {type(answer).__name__}")
        if not isinstance(context, Context):
            raise TypeError(f"context must be a Context, got {type(context).__name__}")
        if not isinstance(sources, tuple):
            raise TypeError(
                f"sources must be a tuple of Citation, got {type(sources).__name__}"
            )
        for source in sources:
            if not isinstance(source, Citation):
                raise TypeError(
                    f"each source must be a Citation, got {type(source).__name__}"
                )

        sentences = _split_sentences(answer)
        answer_tokens = self._tokens(answer)
        reference_tokens = _unique(self._tokens(reference_answer))

        matched = sum(1 for token in reference_tokens if token in set(answer_tokens))
        reference_coverage = matched / len(reference_tokens)

        grounded_sentences = self._grounded_count(sentences, self._tokens(context.text))
        cited_sentences = self._grounded_count(
            sentences, self._tokens(" ".join(source.text for source in sources))
        )

        return AnswerEvaluation(
            query_id=query_id,
            reference_coverage=reference_coverage,
            groundedness=_ratio(grounded_sentences, len(sentences)),
            citation_coverage=_ratio(cited_sentences, len(sentences)),
            has_answer=bool(answer.strip()),
            has_sources=bool(sources),
            source_count=len(sources),
            answer_sentences=len(sentences),
            grounded_sentences=grounded_sentences,
            cited_sentences=cited_sentences,
            reference_tokens=len(reference_tokens),
            matched_reference_tokens=matched,
        )

    def _tokens(self, text: str) -> list[str]:
        """NFC-normalize, then tokenize with the injected tokenizer."""
        return self.tokenizer.tokenize(unicodedata.normalize("NFC", text))

    def _grounded_count(self, sentences: Sequence[str], support: Sequence[str]) -> int:
        """How many sentences are covered by ``support`` at the threshold."""
        support_tokens = set(support)
        grounded = 0
        for sentence in sentences:
            tokens = _unique(self._tokens(sentence))
            if not tokens:
                continue  # a sentence with no tokens has nothing to support
            covered = sum(1 for token in tokens if token in support_tokens) / len(tokens)
            if covered >= self.groundedness_threshold:
                grounded += 1
        return grounded


def aggregate_evaluations(evaluations: Iterable[AnswerEvaluation]) -> dict[str, float]:
    """Mean/rate aggregates over many evaluations, in a fixed key order.

    Empty input yields zeros rather than an error, matching the retrieval
    evaluator's empty-run contract (zero queries -> zero aggregates). Rates are
    fractions of cases; ``mean_source_count`` is an average count, not a rate.
    """
    results = list(evaluations)
    total = len(results)
    if total == 0:
        return {
            "mean_reference_coverage": 0.0,
            "mean_groundedness": 0.0,
            "mean_citation_coverage": 0.0,
            "answer_rate": 0.0,
            "source_presence_rate": 0.0,
            "mean_source_count": 0.0,
        }
    count = float(total)
    return {
        "mean_reference_coverage": sum(r.reference_coverage for r in results) / count,
        "mean_groundedness": sum(r.groundedness for r in results) / count,
        "mean_citation_coverage": sum(r.citation_coverage for r in results) / count,
        "answer_rate": sum(1 for r in results if r.has_answer) / count,
        "source_presence_rate": sum(1 for r in results if r.has_sources) / count,
        "mean_source_count": sum(r.source_count for r in results) / count,
    }


def _split_sentences(text: str) -> list[str]:
    """Split on sentence terminators; empty pieces are dropped."""
    sentences: list[str] = []
    current: list[str] = []
    for char in text:
        if char in _SENTENCE_TERMINATORS:
            piece = "".join(current).strip()
            if piece:
                sentences.append(piece)
            current = []
        else:
            current.append(char)
    tail = "".join(current).strip()
    if tail:
        sentences.append(tail)
    return sentences


def _unique(tokens: Sequence[str]) -> list[str]:
    """First-occurrence order preserved; duplicates dropped."""
    seen: set[str] = set()
    out: list[str] = []
    for token in tokens:
        if token not in seen:
            seen.add(token)
            out.append(token)
    return out


def _ratio(numerator: int, denominator: int) -> float:
    """0.0 when there is nothing to divide by -- never NaN."""
    if denominator == 0:
        return 0.0
    return numerator / denominator
