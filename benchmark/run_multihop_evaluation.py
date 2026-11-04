"""ZMWiki multi-hop QA benchmark (deterministic baseline vs WikiAgent).

Runs two systems over the same source corpus and the same multi-hop questions:

  Baseline:   DeterministicRAGPipeline (single-pass hybrid retrieval +
              extractive generation). Fully offline, no LLM.
  Experiment: WikiAgent (multi-step search/read/follow_links + LLM answer).
              Requires --live with a real model endpoint; offline mode skips
              the experiment and only validates infrastructure.

Usage (offline, baseline only + infrastructure validation):
    python benchmark/run_multihop_evaluation.py --output path/to/result.json

Usage (live, both systems):
    python benchmark/run_multihop_evaluation.py --live \
        --base-url https://api.example.com/v1 --model gpt-4o \
        --api-key KEY --output path/to/result.json

This script never overwrites committed benchmark results.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from zmwiki import (
    DeterministicContextBuilder,
    DeterministicEmbedder,
    DeterministicGenerator,
    DeterministicRAGPipeline,
    DeterministicReranker,
    FixedSizeChunker,
    HybridRetriever,
    InMemoryLexicalIndex,
    InMemoryVectorStore,
    LexicalRetriever,
    DenseRetriever,
    WeightedScoreFusion,
)
from zmwiki.schema import Document
from zmwiki.wiki.wiki_agent import AgentStatus, WikiAgent
from zmwiki.wiki.wiki_index import WikiIndex
from zmwiki.wiki.wiki_link_resolver import LinkResolutionReport, resolve_candidate_links
from zmwiki.wiki.wiki_schema import WikiPage

# ---------------------------------------------------------------------------
# Paths & constants
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]
CORPUS_DIR = ROOT / "benchmark" / "corpus" / "wiki"
DATASET_PATH = ROOT / "benchmark" / "datasets" / "multihop_wiki.json"
DEFAULT_OUTPUT = ROOT / "benchmark" / "results" / "multihop_wiki_comparison.json"

EXPERIMENT_ID = "multihop_wiki_comparison"

# Baseline pipeline config — mirrors run_answer_evaluation.py.
CHUNK_SIZE = 120
OVERLAP = 20
TOP_K = 5
CANDIDATE_K = 50
MAX_CHARS = 2000

# Agent defaults.
AGENT_MAX_STEPS = 8
AGENT_SEARCH_TOP_K = 5

#: page_id -> (filename, links_to) for building the linked WikiIndex.
#: Links are manually verified from source document content, NOT from WikiCompiler.
_PAGE_LINKS: dict[str, tuple[str, list[str]]] = {
    "alice": ("alice_smith.txt", ["bob", "project_phoenix"]),
    "bob": ("bob_jones.txt", ["alice", "carol", "building_a"]),
    "carol": ("carol_chen.txt", ["bob", "project_phoenix", "project_orion"]),
    "project_phoenix": ("project_phoenix.txt", ["alice", "carol"]),
    "project_orion": ("project_orion.txt", ["david", "carol"]),
    "david": ("david_kim.txt", ["project_orion", "eve"]),
    "eve": ("eve_martinez.txt", ["bob", "david", "building_b"]),
    "building_a": ("building_a.txt", ["bob"]),
    "building_b": ("building_b.txt", ["eve"]),
}

#: page_id -> candidate link strings simulating WikiCompiler output.
#: These are stored in metadata["candidate_links"] and resolved by
#: resolve_candidate_links(). Distinct from _PAGE_LINKS (manual, direct page_ids).
#: Candidates use page_ids (the most reliable identifier a compiler can emit);
#: the resolver also supports path/title/alias matching, but title matching
#: requires exact title equality and derived titles are full sentences.
_CANDIDATE_LINKS: dict[str, list[str]] = {
    "alice": ["bob", "project_phoenix"],
    "bob": ["alice", "carol", "building_a"],
    "carol": ["bob", "project_phoenix", "project_orion"],
    "project_phoenix": ["alice", "carol"],
    "project_orion": ["david", "carol"],
    "david": ["project_orion", "eve"],
    "eve": ["bob", "david", "building_b"],
    "building_a": ["bob"],
    "building_b": ["eve"],
}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_dataset() -> dict:
    """Load and validate the multi-hop dataset."""
    payload = json.loads(DATASET_PATH.read_text(encoding="utf-8"))
    cases = payload.get("cases")
    if not isinstance(cases, list) or not cases:
        raise RuntimeError("multihop_wiki.json has no cases list")
    seen = set()
    required = ("id", "question", "reference_answer", "answerable",
                "required_page_ids", "hops", "question_type")
    for case in cases:
        missing = [f for f in required if f not in case]
        if missing:
            raise RuntimeError(f"case {case.get('id')!r} missing {missing}")
        if case["id"] in seen:
            raise RuntimeError(f"duplicate case id: {case['id']!r}")
        seen.add(case["id"])
        if not isinstance(case["required_page_ids"], list):
            raise RuntimeError(f"case {case['id']}: required_page_ids must be a list")
        alternatives = case.get("alternative_evidence_sets", [])
        if not isinstance(alternatives, list) or any(
            not isinstance(path, list) for path in alternatives
        ):
            raise RuntimeError(
                f"case {case['id']}: alternative_evidence_sets must be a list of lists"
            )
        for path in alternatives:
            bad = set(path) - set(_PAGE_LINKS)
            if bad:
                raise RuntimeError(
                    f"case {case['id']}: alternative evidence has unknown pages {bad}"
                )
    return payload


def load_documents() -> list[Document]:
    """Load source documents from the wiki corpus. Document IDs = page IDs."""
    documents = []
    for page_id, (filename, _) in _PAGE_LINKS.items():
        path = CORPUS_DIR / filename
        if not path.exists():
            raise RuntimeError(f"missing corpus file: {path}")
        text = path.read_text(encoding="utf-8").strip()
        documents.append(Document(id=page_id, content=text))
    return documents


def build_wiki_pages(documents: list[Document], *, use_candidate_links: bool = False) -> list[WikiPage]:
    """Build WikiPages.

    Parameters
    ----------
    use_candidate_links : bool
        If True, pages carry metadata["candidate_links"] (title strings from
        _CANDIDATE_LINKS) but have empty links_to/links_from — intended for
        subsequent resolve_candidate_links(). If False (default), pages carry
        manually verified bidirectional links from _PAGE_LINKS.
    """
    doc_by_id = {d.id: d for d in documents}
    pages = []
    for page_id, (_, manual_links_to) in _PAGE_LINKS.items():
        doc = doc_by_id[page_id]
        title = _derive_title(doc.content, page_id)
        if use_candidate_links:
            links_to: tuple[str, ...] = ()
            links_from: tuple[str, ...] = ()
            metadata = {"candidate_links": list(_CANDIDATE_LINKS.get(page_id, []))}
        else:
            links_to = tuple(manual_links_to)
            links_from = tuple(
                other_id for other_id, (_, other_links) in _PAGE_LINKS.items()
                if page_id in other_links and other_id != page_id
            )
            metadata = {}
        pages.append(WikiPage(
            id=page_id,
            path=f"wiki/{page_id}",
            title=title,
            content=doc.content,
            links_to=links_to,
            links_from=links_from,
            sources=(page_id,),
            metadata=metadata,
        ))
    return pages


def _derive_title(content: str, page_id: str) -> str:
    """Extract a title from the first sentence; fall back to page_id."""
    first_sentence = re.split(r"[.!?]", content, maxsplit=1)[0].strip()
    if len(first_sentence) > 80:
        first_sentence = first_sentence[:80].rsplit(" ", 1)[0]
    return first_sentence if first_sentence else page_id.replace("_", " ").title()


def build_wiki_index(pages: list[WikiPage]) -> WikiIndex:
    idx = WikiIndex()
    for page in pages:
        idx.upsert(page)
    return idx


class MonitoringWikiIndex(WikiIndex):
    """A WikiIndex subclass that records every page access (get / get_by_path).

    ``accessed_page_ids`` is intentionally broad: it includes pages touched
    while searching as well as pages opened by read/follow_links. It is kept
    for legacy access_recall. For actual read evidence, use
    ``AgentResult.read_page_ids`` (the source for read_evidence_recall).
    Inherits from WikiIndex so it passes WikiAgent's isinstance check while
    copying the inner index's fully-built state.
    """

    def __init__(self, inner: WikiIndex) -> None:
        super().__init__(tokenizer=inner._tokenizer)
        # Copy the fully-built state from the inner index.
        self._pages = dict(inner._pages)
        self._path_index = dict(inner._path_index)
        self._metadata_index = inner._metadata_index
        self._content_index = inner._content_index
        self.accessed_page_ids: set[str] = set()

    def get(self, page_id: str):
        result = super().get(page_id)
        if result is not None:
            self.accessed_page_ids.add(page_id)
        return result

    def get_by_path(self, path: str):
        result = super().get_by_path(path)
        if result is not None:
            self.accessed_page_ids.add(result.id)
        return result


# ---------------------------------------------------------------------------
# Link resolution diagnostic
# ---------------------------------------------------------------------------


def run_link_resolution(pages: list[WikiPage]) -> dict:
    """Run resolve_candidate_links() and return a metrics dict from its report.

    Uses the REAL resolver from wiki_link_resolver.py — does not reimplement
    matching logic. The input pages must carry metadata["candidate_links"].
    """
    resolved_pages, report = resolve_candidate_links(pages)
    # Denominator for resolution rate: total candidates minus self-links
    # (self-links are intentionally skipped, not failures).
    resolvable_denominator = report.candidates_total - report.self_link_skipped
    resolution_rate = (
        report.resolved_count / resolvable_denominator if resolvable_denominator else 0.0
    )
    return {
        "uses_real_resolver": True,
        "pages_processed": report.pages_processed,
        "candidates_total": report.candidates_total,
        "resolved_count": report.resolved_count,
        "unresolved_count": report.unresolved_count,
        "ambiguous_count": report.ambiguous_count,
        "self_link_skipped": report.self_link_skipped,
        "resolution_rate": resolution_rate,
        "resolution_rate_denominator": resolvable_denominator,
        "resolution_rate_note": "resolved_count / (candidates_total - self_link_skipped). "
                                 "Empty/non-string candidates are not counted in candidates_total.",
        "broken_existing_links": report.broken_existing_links,
        "inconsistent_existing_links": report.inconsistent_existing_links,
        "resolved_links": [
            {"source": rl.source_page_id, "candidate": rl.candidate,
             "target": rl.target_page_id, "match_method": rl.match_method}
            for rl in report.resolved_links
        ],
        "unresolved_candidates": [
            {"page": p, "candidate": c} for p, c in report.unresolved_candidates
        ],
        "ambiguous_candidates": [
            {"page": p, "candidate": c, "matched_ids": list(ids)}
            for p, c, ids in report.ambiguous_candidates
        ],
        "resolved_pages_count": len(resolved_pages),
    }


# ---------------------------------------------------------------------------
# Baseline pipeline
# ---------------------------------------------------------------------------


def build_baseline_pipeline(documents: list[Document]) -> DeterministicRAGPipeline:
    chunker = FixedSizeChunker(chunk_size=CHUNK_SIZE, overlap=OVERLAP)
    embedder = DeterministicEmbedder()
    # Same instances must be shared between the pipeline and the retriever,
    # otherwise ingestion writes to one store while retrieval reads from another.
    vector_store = InMemoryVectorStore()
    lexical_index = InMemoryLexicalIndex()
    pipeline = DeterministicRAGPipeline(
        chunker=chunker,
        embedder=embedder,
        vector_store=vector_store,
        lexical_index=lexical_index,
        retriever=HybridRetriever(
            DenseRetriever(embedder, vector_store),
            LexicalRetriever(lexical_index),
            WeightedScoreFusion(),
        ),
        reranker=DeterministicReranker(),
        context_builder=DeterministicContextBuilder(),
        generator=DeterministicGenerator(),
    )
    for doc in documents:
        pipeline.ingest(doc)
    return pipeline


# ---------------------------------------------------------------------------
# Answer scoring (deterministic, no LLM judge)
# ---------------------------------------------------------------------------

_NORMALIZE_RE = re.compile(r"[^\w\s]", re.UNICODE)


def _normalize(text: str) -> str:
    return _NORMALIZE_RE.sub("", text.lower())


def _matches_any(text: str, keywords: list[str]) -> bool:
    """True if any non-empty normalized keyword is a substring of ``text``."""
    return any((kw := _normalize(k)) and kw in text for k in keywords)


def _split_sentences(text: str) -> list[str]:
    """Split an answer into sentence-like units for structured checks."""
    return [s for s in re.split(r"[.!?;\n]+", text) if s.strip()]


def _alias_spans(text: str, aliases: list[str]) -> list[tuple[int, int, str]]:
    """Return every (start, end, normalized_alias) span for the aliases."""
    spans: list[tuple[int, int, str]] = []
    for alias in aliases:
        norm_alias = _normalize(alias)
        if not norm_alias:
            continue
        # Allow a trailing possessive "s" left behind by _normalize()
        # (e.g. "Bob Jones's" -> "bob joness").
        for match in re.finditer(rf"\b{re.escape(norm_alias)}s?\b", text):
            spans.append((match.start(), match.end(), norm_alias))
    return sorted(spans)


def _find_alias(text: str, aliases: list[str]) -> tuple[int, int, str] | None:
    spans = _alias_spans(text, aliases)
    return spans[0] if spans else None


_NEGATION_RE = re.compile(
    r"\b(?:not|never|no longer|isnt|wasnt|arent|werent|doesnt|didnt|hasnt|havent|cannot|cant|wont|without)\b"
)


def _negated_between(text: str, start: int, end: int) -> bool:
    return bool(_NEGATION_RE.search(text[max(0, start):min(len(text), end)]))


def _binding_window(
    sentence: str,
    subject_end: int,
    subject_aliases: list[str],
    entity_groups: list[list[str]],
) -> str:
    """Return the forward clause after a subject, bounded by other entities."""
    subject_norm = {_normalize(a) for a in subject_aliases if _normalize(a)}
    end = len(sentence)
    for group in entity_groups:
        group_norm = {_normalize(a) for a in group if _normalize(a)}
        if group_norm & subject_norm:
            continue
        for alias in group_norm:
            for match in re.finditer(rf"\b{re.escape(alias)}s?\b", sentence):
                if match.start() >= subject_end:
                    end = min(end, match.start())
    return sentence[subject_end:end]


def _sentence_has_binding(
    sentence: str,
    subject_aliases: list[str],
    value_aliases: list[str],
    entity_groups: list[list[str]],
) -> bool:
    for _, subject_end, _ in _alias_spans(sentence, subject_aliases):
        window = _binding_window(
            sentence, subject_end, subject_aliases, entity_groups
        )
        for _, value_end, _ in _alias_spans(window, value_aliases):
            if not _negated_between(window, 0, value_end):
                return True
    return False


_PRONOUN_RE = re.compile(r"\b(?:her|his|their|she|he|they)\b")


def _sentence_has_pronoun_binding(
    sentences: list[str],
    index: int,
    subject_aliases: list[str],
    value_aliases: list[str],
    entity_groups: list[list[str]],
) -> bool:
    """Resolve a clear pronoun subject to the immediately preceding entity.

    The previous sentence must name the binding's subject and no other
    candidate entity. The pronoun sentence must not name another candidate
    before the pronoun. Ambiguous antecedents are rejected, never guessed.
    """
    sentence = sentences[index]
    pronoun = _PRONOUN_RE.search(sentence)
    if pronoun is None:
        return False
    value_found = False
    for _, value_end, _ in _alias_spans(sentence, value_aliases):
        if not _negated_between(sentence, 0, value_end):
            value_found = True
            break
    if not value_found:
        return False
    if index == 0:
        return False
    previous = sentences[index - 1]
    if _find_alias(previous, subject_aliases) is None:
        return False
    subject_norm = {_normalize(a) for a in subject_aliases if _normalize(a)}
    for group in entity_groups:
        group_norm = {_normalize(a) for a in group if _normalize(a)}
        if group_norm & subject_norm:
            continue
        # The previous sentence must not offer another possible antecedent.
        if any(_find_alias(previous, [alias]) for alias in group_norm):
            return False
        # The pronoun sentence must not name another candidate before it.
        for alias in group_norm:
            for match in re.finditer(rf"\b{re.escape(alias)}s?\b", sentence):
                if match.start() < pronoun.start():
                    return False
    return True


def _check_answer_bindings(answer: str, case: dict) -> bool:
    bindings = case.get("answer_bindings", [])
    if not bindings:
        return True
    sentences = [_normalize(s) for s in _split_sentences(answer)]
    if not sentences:
        return False
    entity_groups = case.get("binding_entities", [])
    for binding in bindings:
        subject_aliases = binding.get("subject_aliases", [])
        require_groups = binding.get("require_groups", [])
        forbid_groups = binding.get("forbid_groups", [])
        if not subject_aliases or (not require_groups and not forbid_groups):
            return False
        for group in require_groups:
            direct = any(
                _sentence_has_binding(s, subject_aliases, group, entity_groups)
                for s in sentences
            )
            pronoun = any(
                _sentence_has_pronoun_binding(
                    sentences, index, subject_aliases, group, entity_groups
                )
                for index in range(len(sentences))
            )
            if not (direct or pronoun):
                return False
        forbid_entities = (
            [] if binding.get("forbid_scope") == "sentence" else entity_groups
        )
        for group in forbid_groups:
            direct = any(
                _sentence_has_binding(s, subject_aliases, group, forbid_entities)
                for s in sentences
            )
            pronoun = any(
                _sentence_has_pronoun_binding(
                    sentences, index, subject_aliases, group, forbid_entities
                )
                for index in range(len(sentences))
            )
            if direct or pronoun:
                return False
    return True


def _sentence_has_shared_entities(
    sentence: str, entity_groups: list[list[str]], value_aliases: list[str]
) -> bool:
    has_all_names = all(_find_alias(sentence, group) for group in entity_groups)
    has_pronoun = bool(
        re.search(r"\b(?:both|they|shared|share|common)\b", sentence)
    )
    if not has_all_names and not has_pronoun:
        return False
    value = _find_alias(sentence, value_aliases)
    if value is None:
        return False
    return not _negated_between(sentence, 0, value[1])


def _check_shared_entities(answer: str, case: dict) -> bool:
    specs = case.get("shared_entities", [])
    if not specs:
        return True
    sentences = [_normalize(s) for s in _split_sentences(answer)]
    for spec in specs:
        entities = spec.get("entities", [])
        values = spec["require"]
        all_bound = all(
            any(_sentence_has_binding(s, group, values, []) for s in sentences)
            for group in entities
        )
        pronoun_bound = any(
            _sentence_has_shared_entities(s, entities, values) for s in sentences
        )
        if not (all_bound or pronoun_bound):
            return False
    return True


def _ordered(text: str, order_groups: list[list[str]]) -> bool:
    positions = []
    for group in order_groups:
        span = _find_alias(text, group)
        if span is None:
            return False
        positions.append(span[0])
    return all(a < b for a, b in zip(positions, positions[1:]))


def _has_ordered_entities(
    answer: str, sentences: list[str], order_groups: list[list[str]]
) -> bool:
    return _ordered(answer, order_groups) or any(
        _ordered(s, order_groups) for s in sentences
    )


def _entity_between(
    sentence: str,
    start: int,
    end: int,
    from_aliases: list[str],
    to_aliases: list[str],
    entity_groups: list[list[str]],
) -> bool:
    exclude = {_normalize(a) for a in from_aliases + to_aliases if _normalize(a)}
    lo, hi = sorted((start, end))
    for group in entity_groups:
        group_norm = {_normalize(a) for a in group if _normalize(a)}
        if group_norm & exclude:
            continue
        for alias in group_norm:
            for match in re.finditer(rf"\b{re.escape(alias)}s?\b", sentence):
                if lo < match.start() < hi:
                    return True
    return False


_INVERSE_RELATION_RE = re.compile(r"\b(?:to whom|by)\b")


def _inverse_relation_phrase(sentence: str, to_end: int, from_start: int) -> bool:
    """True when a to-before-from mention uses an explicit relative/passive clause."""
    if to_end >= from_start:
        return False
    return bool(_INVERSE_RELATION_RE.search(sentence[to_end:from_start]))


def _marker_near_pair(
    sentence: str,
    markers: list[str],
    from_start: int,
    from_end: int,
    to_start: int,
    to_end: int,
    entity_groups: list[list[str]] | None,
    *,
    inside_only: bool = False,
    relative_clause_guard: bool = False,
) -> bool:
    """Require the relation marker in the clause around this entity pair.

    The clause is bounded by the nearest other entity mentions, so a direct
    marker used for a different pair in the same sentence cannot satisfy this
    pair. ``inside_only`` is used for forbidden-direct checks: only a marker
    between the two mentions can accuse this exact pair.
    """
    lo, hi = min(from_start, to_start), max(from_end, to_end)
    if inside_only:
        if relative_clause_guard:
            previous: tuple[int, int] | None = None
            for group in entity_groups or []:
                for alias in {_normalize(a) for a in group if _normalize(a)}:
                    for match in re.finditer(rf"\b{re.escape(alias)}s?\b", sentence):
                        if match.end() <= lo and (
                            previous is None or match.end() > previous[1]
                        ):
                            previous = (match.start(), match.end())
            if previous is not None:
                previous_is_to = previous[0] < to_end and previous[1] > to_start
                if (
                    not previous_is_to
                    and re.search(r"\bto whom\b", sentence[previous[1]:lo])
                ):
                    return False
        return _positive_marker(sentence[lo:hi], markers)
    prev_end, next_start = 0, len(sentence)
    for group in entity_groups or []:
        for alias in {_normalize(a) for a in group if _normalize(a)}:
            for match in re.finditer(rf"\b{re.escape(alias)}s?\b", sentence):
                if match.end() <= lo:
                    prev_end = max(prev_end, match.end())
                elif match.start() >= hi:
                    next_start = min(next_start, match.start())
    return _positive_marker(sentence[prev_end:next_start], markers)


def _has_relation_pair(
    sentences: list[str],
    from_aliases: list[str],
    to_aliases: list[str],
    markers: list[str],
    *,
    direction: str = "from_before_to",
    entity_groups: list[list[str]] | None = None,
    adjacent: bool = False,
    inverse_requires_phrase: bool = False,
    marker_inside_pair: bool = False,
    relative_clause_guard: bool = False,
) -> bool:
    for sentence in sentences:
        for from_start, from_end, _ in _alias_spans(sentence, from_aliases):
            for to_start, to_end, _ in _alias_spans(sentence, to_aliases):
                if direction == "from_before_to" and from_start >= to_start:
                    continue
                if direction == "to_before_from" and to_start >= from_start:
                    continue
                if (
                    direction == "either"
                    and inverse_requires_phrase
                    and from_start > to_start
                    and not _inverse_relation_phrase(sentence, to_end, from_start)
                ):
                    continue
                if adjacent and entity_groups and _entity_between(
                    sentence, from_start, to_start, from_aliases, to_aliases,
                    entity_groups,
                ):
                    continue
                if not _marker_near_pair(
                    sentence, markers, from_start, from_end, to_start, to_end,
                    entity_groups, inside_only=marker_inside_pair,
                    relative_clause_guard=relative_clause_guard,
                ):
                    continue
                start = min(from_start, to_start)
                end = max(from_end, to_end)
                if _negated_between(sentence, start, end):
                    continue
                return True
    return False


def _check_relation_chain(answer: str, case: dict) -> bool:
    chain = case.get("relation_chain")
    if not chain:
        return True
    sentences = [_normalize(s) for s in _split_sentences(answer)]
    if not sentences:
        return False
    norm_answer = _normalize(answer)
    order = chain.get("order")
    if order and not _has_ordered_entities(norm_answer, sentences, order):
        return False
    entity_groups = chain.get("entities", [])
    for pair in chain.get("direct_pairs", []):
        if not _has_relation_pair(
            sentences,
            pair["from"],
            pair["to"],
            pair.get("markers", ["direct", "directly"]),
            direction="either",
            entity_groups=entity_groups,
            adjacent=True,
            inverse_requires_phrase=True,
        ):
            return False
    for pair in chain.get("indirect_pairs", []):
        if not _has_relation_pair(
            sentences,
            pair["from"],
            pair["to"],
            pair.get("markers", ["indirect", "indirectly", "through", "two levels"]),
            direction="either",
            entity_groups=entity_groups,
        ):
            return False
    for pair in chain.get("forbidden_direct_pairs", []):
        if _has_relation_pair(
            sentences,
            pair["from"],
            pair["to"],
            pair.get("markers", ["direct", "directly"]),
            direction="either",
            entity_groups=entity_groups,
            adjacent=True,
            marker_inside_pair=True,
            relative_clause_guard=True,
        ):
            return False
    for pair in chain.get("leadership_pairs", []):
        if not _has_relation_pair(
            sentences,
            pair["from"],
            pair["to"],
            pair.get("markers", ["leads", "led", "leader"]),
            direction="either",
            entity_groups=entity_groups,
        ):
            return False
    return True


def _matches_any_pattern(
    text: str, patterns: list[str], *, ignore_case: bool = True
) -> bool:
    flags = re.IGNORECASE if ignore_case else 0
    return any(re.search(pattern, text, flags) for pattern in patterns)


def _positive_marker(sentence: str, markers: list[str]) -> bool:
    """True when a non-negated comparison marker is present in the sentence."""
    for marker in markers:
        for match in re.finditer(rf"\b{re.escape(marker)}\b", sentence):
            before = sentence[max(0, match.start() - 16):match.start()]
            if not _NEGATION_RE.search(before):
                return True
    return False


def _check_comparison(answer: str, case: dict) -> bool:
    """Check an explicitly directed comparison (greater/less than).

    A greater-marker must connect the left entity before the right entity,
    or a less-marker must connect the right entity before the left entity.
    Reversed comparisons and bare number lists are rejected.
    """
    specs = case.get("comparison_checks", [])
    if not specs:
        return True
    sentences = [_normalize(s) for s in _split_sentences(answer)]
    for spec in specs:
        left_aliases = [
            alias for group in spec.get("left_entities", []) for alias in group
        ]
        right_aliases = [
            alias for group in spec.get("right_entities", []) for alias in group
        ]
        left_markers = spec.get("left_greater_markers", [])
        right_markers = spec.get("right_less_markers", [])
        if not left_aliases or not right_aliases:
            return False
        matched = False
        for sentence in sentences:
            left = _find_alias(sentence, left_aliases)
            right = _find_alias(sentence, right_aliases)
            if left is None or right is None:
                continue
            if left[0] < right[0] and _positive_marker(sentence, left_markers):
                matched = True
                break
            if right[0] < left[0] and _positive_marker(sentence, right_markers):
                matched = True
                break
        if not matched:
            return False
    return True


_POSITIVE_NEGATION_RE = re.compile(
    r"\b(?:not|never|no|none|nobody|without|isnt|wasnt|arent|werent|"
    r"doesnt|didnt|hasnt|havent|cannot|cant|wont)\b"
)


def _matches_positive_pattern(text: str, patterns: list[str]) -> bool:
    """Regex guard that ignores matches surrounded by negation markers."""
    for pattern in patterns:
        for match in re.finditer(pattern, text, re.IGNORECASE):
            window = _normalize(
                text[max(0, match.start() - 20):min(len(text), match.end() + 20)]
            )
            if _POSITIVE_NEGATION_RE.search(window):
                continue
            return True
    return False


_NUMBER_WORDS = {
    0: "zero",
    1: "one",
    2: "two",
    3: "three",
    4: "four",
    5: "five",
    6: "six",
    7: "seven",
    8: "eight",
    9: "nine",
    10: "ten",
}


def _ratio_matches(text: str, numerator: int, denominator: int) -> bool:
    """True for an unambiguous n:d ratio expression, never a bare digit."""
    n = str(numerator)
    d = str(denominator)
    n_word = _NUMBER_WORDS.get(numerator)
    d_word = _NUMBER_WORDS.get(denominator)
    # Boundaries reject 15:1, 5:10, 5:1,000, 0.5:1, 5:1:0, etc.
    left = r"(?<![\w.:/])"
    right = r"(?![\w:/]|\.\d|,\d)"
    patterns = [
        rf"{left}{n}\s*:\s*{d}{right}",
        rf"{left}{n}\s*/\s*{d}{right}",
        rf"{left}{n}\s+to\s+{d}{right}",
        rf"\bratio\s+(?:is|=|of)\s*{n}(?!\w|\.\d)",
    ]
    if n_word and d_word:
        patterns.extend(
            [
                rf"\b{re.escape(n_word)}\s+to\s+{re.escape(d_word)}\b",
                rf"\b{re.escape(n_word)}\s*:\s*{re.escape(d_word)}\b",
                rf"\b{re.escape(n_word)}\s*/\s*{re.escape(d_word)}\b",
                rf"\bratio\s+(?:is|=|of)\s*{re.escape(n_word)}\b",
            ]
        )
    for pattern in patterns:
        for match in re.finditer(pattern, text, re.IGNORECASE):
            before = _normalize(text[max(0, match.start() - 16):match.start()])
            if _NEGATION_RE.search(before):
                continue
            return True
    return False


def _check_ratio(answer: str, case: dict) -> bool:
    specs = case.get("ratio_checks", [])
    if not specs:
        return True
    stripped = answer.strip()
    norm = _normalize(stripped)
    for spec in specs:
        numerator = spec.get("numerator")
        denominator = spec.get("denominator")
        if not isinstance(numerator, int) or not isinstance(denominator, int):
            continue
        if denominator == 0:
            continue
        if spec.get("allow_bare"):
            bare = {str(numerator)}
            word = _NUMBER_WORDS.get(numerator)
            if word:
                bare.add(word)
            if norm in bare:
                return True
        if _ratio_matches(stripped, numerator, denominator):
            return True
    return False


def score_answer(answer: str | None, case: dict) -> bool:
    """Deterministic answer scoring with optional structured checks.

    Existing keyword_group_match behavior is preserved. Cases may additionally
    declare:
      - answer_exact_aliases: whole-answer aliases for short numeric answers.
      - answer_bindings + binding_entities: entity-to-attribute binding checks.
      - shared_entities: require a project/attribute to be stated for all named
        entities (or with "both"/"they").
      - relation_chain: ordered chain plus direct/indirect relation checks.
      - ratio_checks: bounded n:d ratio recognition (never a bare numeric
        substring such as "51").
      - comparison_checks: directed greater/less comparison with entity order.
      - fabrication_patterns / fabrication_patterns_cs: regex guards for
        unanswerable cases (still heuristic; manual review may be required).
      - fabrication_patterns_positive: regex guards that ignore negated
        matches, for refusing answers that mention a true fact as context.
    """
    if answer is None or not answer.strip():
        return False
    norm = _normalize(answer)
    if not case["answerable"]:
        if not _matches_any(norm, case.get("refusal_keywords", [])):
            return False
        if _matches_any(norm, case.get("fabrication_keywords", [])):
            return False
        if _matches_any_pattern(answer, case.get("fabrication_patterns", [])):
            return False
        if _matches_any_pattern(
            answer, case.get("fabrication_patterns_cs", []), ignore_case=False
        ):
            return False
        if _matches_positive_pattern(
            answer, case.get("fabrication_patterns_positive", [])
        ):
            return False
        return True

    exact_aliases = case.get("answer_exact_aliases", [])
    if exact_aliases:
        raw = answer.strip().lower()
        exact_raw = {alias.strip().lower() for alias in exact_aliases}
        exact_norm = {_normalize(alias) for alias in exact_aliases}
        if raw in exact_raw or norm in exact_norm:
            return (
                _check_answer_bindings(answer, case)
                and _check_shared_entities(answer, case)
                and _check_relation_chain(answer, case)
                and _check_ratio(answer, case)
                and _check_comparison(answer, case)
            )

    groups = case.get("answer_keyword_groups", [])
    structured_checks = (
        "answer_bindings",
        "shared_entities",
        "relation_chain",
        "ratio_checks",
        "comparison_checks",
    )
    if not groups and not any(key in case for key in structured_checks):
        return False
    for group in groups:
        if not _matches_any(norm, group):
            return False
    if not _check_answer_bindings(answer, case):
        return False
    if not _check_shared_entities(answer, case):
        return False
    if not _check_relation_chain(answer, case):
        return False
    if not _check_ratio(answer, case):
        return False
    if not _check_comparison(answer, case):
        return False
    return True


# ---------------------------------------------------------------------------
# Per-case result & metrics
# ---------------------------------------------------------------------------


@dataclass
class CaseResult:
    id: str
    question: str
    answer: str | None
    #: Formal correctness: lexical match AND status == 'answered'.
    #: The legacy lexical value is retained in lexical_match.
    answer_correct: bool
    status: str
    sources: list[str]
    accessed_pages: list[str]
    required_pages: list[str]
    evidence_recall: float | None  # None for unanswerable cases (not applicable)
    source_precision: float
    steps: int
    pages_read: int
    answerable: bool = True
    scoring_rule: str = "keyword_group_match"  # lexical heuristic, not semantic
    link_source: str = "manual"  # "manual" | "auto" | "none"
    evidence_recall_numerator: int = 0
    evidence_recall_denominator: int = 0
    error: str | None = None
    #: Per-step agent diagnostics (step/action/params/result_summary/errors).
    #: Empty for baseline runs and exceptions; populated from AgentResult.steps.
    step_details: list[dict] = field(default_factory=list)
    #: Pages returned by successful Agent search actions (union over all steps).
    search_hit_page_ids: list[str] = field(default_factory=list)
    #: Pages actually read by the Agent (AgentResult.read_page_ids). For the
    #: baseline, this is the retrieved document set because its single-pass
    #: pipeline consumes that context.
    read_page_ids: list[str] = field(default_factory=list)
    #: |accessed_pages ∩ required| / |required|. This keeps the legacy
    #: evidence_recall semantics under an explicit name; accessed_pages also
    #: includes search candidate lookups.
    access_recall: float | None = None
    access_recall_numerator: int = 0
    access_recall_denominator: int = 0
    #: |read_page_ids ∩ required| / |required|. Unlike access_recall, this
    #: uses only pages actually read by the Agent.
    read_evidence_recall: float | None = None
    read_evidence_recall_numerator: int = 0
    read_evidence_recall_denominator: int = 0
    #: Citation-backed correctness: answer_correct AND at least one cited
    #: source. This checks citation presence only; it is NOT a per-fact
    #: entailment/support check and must not be described as a strict
    #: fact-level evidence-support rate.
    grounded_correct: bool = False
    #: Keyword-correct but not grounded (no source, or a non-answered terminal
    #: status). Exposes lexical false positives and no-evidence answers.
    unsupported_correct: bool = False
    #: Keyword/refusal match only, ignoring status and sources. This preserves
    #: the historical lexical score as a diagnostic; it is NOT formal
    #: correctness and must not be counted in formal accuracy.
    lexical_match: bool = False

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "question": self.question,
            "answer": self.answer,
            "answer_correct": self.answer_correct,
            "lexical_match": self.lexical_match,
            "grounded_correct": self.grounded_correct,
            "unsupported_correct": self.unsupported_correct,
            "answerable": self.answerable,
            "scoring_rule": self.scoring_rule,
            "link_source": self.link_source,
            "status": self.status,
            "sources": self.sources,
            "accessed_pages": self.accessed_pages,
            "required_pages": self.required_pages,
            "evidence_recall": self.evidence_recall,
            "evidence_recall_numerator": self.evidence_recall_numerator,
            "evidence_recall_denominator": self.evidence_recall_denominator,
            "source_precision": self.source_precision,
            "steps": self.steps,
            "pages_read": self.pages_read,
            "error": self.error,
            "step_details": self.step_details,
            "search_hit_page_ids": self.search_hit_page_ids,
            "read_page_ids": self.read_page_ids,
            "access_recall": self.access_recall,
            "access_recall_numerator": self.access_recall_numerator,
            "access_recall_denominator": self.access_recall_denominator,
            "read_evidence_recall": self.read_evidence_recall,
            "read_evidence_recall_numerator": self.read_evidence_recall_numerator,
            "read_evidence_recall_denominator": self.read_evidence_recall_denominator,
        }


def _compute_recall(
    page_ids: set[str],
    required: list[str],
    alternative_sets: list[list[str]] | None = None,
) -> tuple[float | None, int, int]:
    """Return (recall, numerator, denominator) for the best evidence path.

    The primary path is ``required``. Each entry in ``alternative_sets`` is a
    complete legal evidence path; the path with the highest page coverage wins.
    A fully covered alternative path therefore yields recall 1.0 without
    weakening the primary requirement for cases that use the primary path.
    For unanswerable cases (required empty), recall is None and counts are zero.
    """
    if not required:
        return None, 0, 0
    paths = [list(required)] + [list(p) for p in (alternative_sets or []) if p]
    best_path = max(
        paths,
        key=lambda path: len(page_ids & set(path)) / len(path),
    )
    numerator = len(page_ids & set(best_path))
    denominator = len(best_path)
    return numerator / denominator, numerator, denominator


def _compute_evidence_metrics(
    accessed: set[str],
    required: list[str],
    sources: list[str],
    alternative_sets: list[list[str]] | None = None,
) -> tuple[float | None, float, int, int]:
    """Return (evidence_recall, source_precision, recall_numerator, recall_denominator).

    evidence_recall uses the best legal evidence path (primary required pages or
    an annotated alternative_evidence_sets path). source_precision treats every
    page in the primary and alternative paths as relevant evidence.
      - For unanswerable cases (required empty): recall is None (not applicable,
        NOT 1.0); unanswerable cases are excluded from mean_evidence_recall.
    recall_numerator / recall_denominator: counts for the selected best path.
    """
    recall, numerator, denominator = _compute_recall(
        accessed, required, alternative_sets
    )
    if not required:
        return recall, 0.0, numerator, denominator
    relevant = set(required)
    for path in alternative_sets or []:
        relevant.update(path)
    if not sources:
        precision = 0.0
    else:
        precision = len(set(sources) & relevant) / len(sources)
    return recall, precision, numerator, denominator


def aggregate_results(cases: list[dict], results: list[CaseResult]) -> dict:
    """Compute aggregate metrics from per-case results.

    Metrics are organized into two groups:
      - retrieval_evidence_metrics: comparable across Baseline and Experiment
        (both measure whether required pages were found, independent of generator).
      - answer_quality_metrics: NOT directly comparable across systems because
        Baseline uses DeterministicGenerator (offline extractive) while Experiment
        uses a real LLM. Reported as independent system results.

    Evidence recall is computed ONLY over answerable cases. Unanswerable cases
    carry evidence_recall=None and are excluded from the mean. access_recall
    and read_evidence_recall are likewise excluded for unanswerable cases;
    access_recall preserves the legacy accessed_pages definition, while
    read_evidence_recall uses AgentResult.read_page_ids only.
    """
    total = len(results)
    answerable = [r for r, c in zip(results, cases) if c["answerable"]]
    unanswerable = [r for r, c in zip(results, cases) if not c["answerable"]]

    def _safe_mean(values: list[float]) -> float:
        return sum(values) / len(values) if values else 0.0

    # --- Answer quality (system-dependent, not directly comparable) ---
    correct = [r for r in results if r.answer_correct]
    lexical_matches = [r for r in results if r.lexical_match]
    grounded = [r for r in results if r.grounded_correct]
    unsupported = [r for r in results if r.unsupported_correct]
    answerable_correct = [r for r in answerable if r.answer_correct]
    unanswerable_correct = [r for r in unanswerable if r.answer_correct]

    answer_quality = {
        "note": "NOT directly comparable across systems: Baseline uses DeterministicGenerator "
                "(offline extractive), Experiment uses a real LLM. Report as independent results.",
        "answer_accuracy": {
            "value": len(correct) / total if total else 0.0,
            "numerator": len(correct),
            "denominator": total,
            "scope": "all cases",
            "grounding_note": "Status-aware formal correctness: lexical match AND "
                              "status=='answered'. A lexical match in a non-answered "
                              "terminal status is NOT counted here; see lexical_match "
                              "for the legacy diagnostic.",
        },
        "lexical_match": {
            "count": len(lexical_matches),
            "rate": len(lexical_matches) / total if total else 0.0,
            "denominator": total,
            "definition": "score_answer(...) only, ignoring status and sources. "
                          "Diagnostic for historical lexical scoring; NOT formal correctness.",
        },
        "verified_answer_accuracy": {
            "value": len(grounded) / total if total else 0.0,
            "numerator": len(grounded),
            "denominator": total,
            "scope": "all cases",
            "definition": "answer_correct AND status=='answered' AND at least one "
                          "cited source. Citation presence only; NOT a per-fact "
                          "entailment or evidence-support rate.",
        },
        "unsupported_correct": {
            "count": len(unsupported),
            "rate": len(unsupported) / total if total else 0.0,
            "definition": "lexical_match but NOT grounded: no cited source, or a "
                          "non-answered terminal status (insufficient_evidence / budget / "
                          "protocol / model error)",
            "interpretation": "Candidate false positives from the lexical scorer or "
                              "answers produced without evidence.",
        },
        "answerable_accuracy": {
            "value": len(answerable_correct) / len(answerable) if answerable else 0.0,
            "numerator": len(answerable_correct),
            "denominator": len(answerable),
            "scope": "answerable cases only (formal, status-aware)",
        },
        "unanswerable_handling": {
            "value": len(unanswerable_correct) / len(unanswerable) if unanswerable else 0.0,
            "numerator": len(unanswerable_correct),
            "denominator": len(unanswerable),
            "scope": "unanswerable cases only",
            "caveat": f"{len(unanswerable)} unanswerable case(s); keyword-only refusal "
                      "scoring can accept a refusal followed by a fabricated answer. "
                      "Treat as a manual-audit metric.",
        },
        "scoring_rule": "keyword_group_match (lexical heuristic, NOT semantic correctness; "
                         "paraphrases may score low even when correct)",
    }

    # --- Retrieval & evidence (comparable across systems) ---
    answerable_recalls = [r.evidence_recall for r in answerable if r.evidence_recall is not None]
    total_recall_numerator = sum(r.evidence_recall_numerator for r in answerable)
    total_recall_denominator = sum(r.evidence_recall_denominator for r in answerable)
    # New explicit metrics. access_recall mirrors legacy evidence_recall.
    # Falls back to legacy evidence_recall for CaseResults constructed without
    # the new fields (e.g. third-party tests or old in-memory fixtures).
    access_recalls: list[float] = []
    access_numerator_sum = 0
    access_denominator_sum = 0
    read_recalls: list[float] = []
    read_numerator_sum = 0
    read_denominator_sum = 0
    for r in answerable:
        access_value = (
            r.access_recall if r.access_recall is not None else r.evidence_recall
        )
        if access_value is not None:
            access_recalls.append(access_value)
            access_numerator_sum += (
                r.access_recall_numerator
                if r.access_recall is not None
                else r.evidence_recall_numerator
            )
            access_denominator_sum += (
                r.access_recall_denominator
                if r.access_recall is not None
                else r.evidence_recall_denominator
            )
        if r.read_evidence_recall is not None:
            read_recalls.append(r.read_evidence_recall)
            read_numerator_sum += r.read_evidence_recall_numerator
            read_denominator_sum += r.read_evidence_recall_denominator
    evidence_supported = [r for r in answerable if r.evidence_recall == 1.0 and r.answer_correct]
    verified_supported = [
        r for r in answerable if r.grounded_correct and r.evidence_recall == 1.0
    ]
    multi_hop = [r for r, c in zip(results, cases) if c["hops"] >= 2]
    multi_hop_answerable = [r for r in multi_hop if r.answerable]
    multi_hop_success = [r for r in multi_hop_answerable
                         if r.answer_correct and r.evidence_recall == 1.0]
    verified_multi_hop_success = [
        r for r in multi_hop_answerable
        if r.grounded_correct and r.evidence_recall == 1.0
    ]

    retrieval_evidence = {
        "note": "Comparable across Baseline and Experiment: both measure whether required "
                "pages were found, independent of the answer generator.",
        "mean_evidence_recall_answerable": {
            "value": _safe_mean(answerable_recalls),
            "numerator_sum": total_recall_numerator,
            "denominator_sum": total_recall_denominator,
            "scope": "answerable cases only (unanswerable excluded; evidence_recall=None)",
            "definition": "LEGACY: |accessed_pages ∩ required| / |required|. "
                          "accessed_pages includes search candidate accesses."""
        },
        "mean_access_recall_answerable": {
            "value": _safe_mean(access_recalls),
            "numerator_sum": access_numerator_sum,
            "denominator_sum": access_denominator_sum,
            "cases_with_access_data": len(access_recalls),
            "scope": "answerable cases only (unanswerable excluded; access_recall=None)",
            "definition": "|accessed_pages ∩ required| / |required|. "
                          "accessed_pages records every get/get_by_path lookup, "
                          "including search candidates; it is not read-only.",
        },
        "mean_read_evidence_recall_answerable": {
            "value": _safe_mean(read_recalls),
            "numerator_sum": read_numerator_sum,
            "denominator_sum": read_denominator_sum,
            "cases_with_read_data": len(read_recalls),
            "scope": "answerable cases only (unanswerable excluded; "
                     "read_evidence_recall=None)",
            "definition": "|read_page_ids ∩ required| / |required|. "
                          "read_page_ids comes from AgentResult.read_page_ids "
                          "(actual read/follow_links results), not search candidates. "
                          "Baseline maps read_page_ids to its retrieved context set.",
        },
        "mean_source_precision": {
            "value": _safe_mean([r.source_precision for r in results]),
            "scope": "all cases",
        },
        "evidence_support_rate_answerable": {
            "value": len(evidence_supported) / len(answerable) if answerable else 0.0,
            "numerator": len(evidence_supported),
            "denominator": len(answerable),
            "definition": "answer_correct (formal: lexical match AND "
                          "status=='answered') AND evidence_recall==1.0, among "
                          "answerable cases. Does NOT require cited sources; use "
                          "verified_evidence_support_rate_answerable for grounded results.",
        },
        "verified_evidence_support_rate_answerable": {
            "value": len(verified_supported) / len(answerable) if answerable else 0.0,
            "numerator": len(verified_supported),
            "denominator": len(answerable),
            "definition": "grounded_correct AND evidence_recall==1.0, among answerable "
                          "cases. grounded_correct requires a successful answer status "
                          "and at least one cited source; it checks citation presence, "
                          "not per-fact entailment.",
        },
        "multi_hop_success_rate": {
            "value": len(multi_hop_success) / len(multi_hop_answerable) if multi_hop_answerable else 0.0,
            "numerator": len(multi_hop_success),
            "denominator": len(multi_hop_answerable),
            "definition": "answer_correct (formal: lexical match AND "
                          "status=='answered') AND evidence_recall==1.0, among "
                          "answerable multi-hop cases. See "
                          "verified_multi_hop_success_rate.",
        },
        "verified_multi_hop_success_rate": {
            "value": len(verified_multi_hop_success) / len(multi_hop_answerable) if multi_hop_answerable else 0.0,
            "numerator": len(verified_multi_hop_success),
            "denominator": len(multi_hop_answerable),
            "definition": "grounded_correct AND evidence_recall==1.0, among answerable "
                          "multi-hop cases.",
        },
        "unanswerable_cases": {
            "count": len(unanswerable),
            "evidence_recall_treatment": "excluded from mean (evidence_recall=None, not 1.0)",
        },
    }

    # --- Efficiency & reliability ---
    failures = [r for r in results if r.error is not None or r.status in (
        AgentStatus.MODEL_ERROR, AgentStatus.PROTOCOL_ERROR)]
    budget_exhausted = [r for r in results if r.status == AgentStatus.BUDGET_EXHAUSTED]

    status_counts: dict[str, int] = {}
    for r in results:
        status_counts[r.status] = status_counts.get(r.status, 0) + 1
    status_counts = dict(sorted(status_counts.items()))

    def _status_rate(status: str) -> float:
        return (
            sum(1 for r in results if r.status == status) / total if total else 0.0
        )

    efficiency = {
        "mean_steps": _safe_mean([r.steps for r in results]),
        "mean_pages_read": _safe_mean([r.pages_read for r in results]),
        "failure_rate": len(failures) / total if total else 0.0,
        "failure_rate_definition": "cases with a terminal error or status in "
                                   "(model_error, protocol_error); budget_exhausted also "
                                   "carries an error string, so it is included",
        "budget_exhausted_rate": len(budget_exhausted) / total if total else 0.0,
        "status_counts": status_counts,
        "status_rates": {
            "insufficient_evidence": _status_rate(AgentStatus.INSUFFICIENT_EVIDENCE),
            "budget_exhausted": _status_rate(AgentStatus.BUDGET_EXHAUSTED),
            "protocol_error": _status_rate(AgentStatus.PROTOCOL_ERROR),
            "model_error": _status_rate(AgentStatus.MODEL_ERROR),
        },
    }

    return {
        "total_cases": total,
        "answerable_cases": len(answerable),
        "unanswerable_cases": len(unanswerable),
        "multi_hop_cases": len(multi_hop),
        "retrieval_evidence_metrics": retrieval_evidence,
        "answer_quality_metrics": answer_quality,
        "efficiency_metrics": efficiency,
        # Backward-compatible flat keys (pre-M4.2 format).
        "answer_accuracy": answer_quality["answer_accuracy"]["value"],
        "lexical_match_rate": answer_quality["lexical_match"]["rate"],
        "lexical_match_count": answer_quality["lexical_match"]["count"],
        "verified_answer_accuracy": answer_quality["verified_answer_accuracy"]["value"],
        "unsupported_correct_count": answer_quality["unsupported_correct"]["count"],
        "answerable_accuracy": answer_quality["answerable_accuracy"]["value"],
        "unanswerable_handling": answer_quality["unanswerable_handling"]["value"],
        "mean_evidence_recall": retrieval_evidence["mean_evidence_recall_answerable"]["value"],
        "mean_access_recall": retrieval_evidence["mean_access_recall_answerable"]["value"],
        "mean_read_evidence_recall":
            retrieval_evidence["mean_read_evidence_recall_answerable"]["value"],
        "mean_source_precision": retrieval_evidence["mean_source_precision"]["value"],
        "multi_hop_success_rate": retrieval_evidence["multi_hop_success_rate"]["value"],
        "verified_multi_hop_success_rate":
            retrieval_evidence["verified_multi_hop_success_rate"]["value"],
        "mean_steps": efficiency["mean_steps"],
        "mean_pages_read": efficiency["mean_pages_read"],
        "failure_rate": efficiency["failure_rate"],
        "budget_exhausted_rate": efficiency["budget_exhausted_rate"],
        "status_counts": efficiency["status_counts"],
        "status_rates": efficiency["status_rates"],
        "correct_count": len(correct),
        "answerable_correct_count": len(answerable_correct),
    }


# ---------------------------------------------------------------------------
# Baseline runner
# ---------------------------------------------------------------------------


def run_baseline(cases: list[dict], pipeline: DeterministicRAGPipeline) -> list[CaseResult]:
    results = []
    for case in cases:
        query = case["question"]
        try:
            answer_result = pipeline.answer_with_sources(
                query, top_k=TOP_K, candidate_k=CANDIDATE_K, max_chars=MAX_CHARS
            )
            context = pipeline.query(
                query, top_k=TOP_K, candidate_k=CANDIDATE_K, max_chars=MAX_CHARS
            )
            # Retrieved document IDs (from chunk IDs like "alice::0").
            retrieved_doc_ids = set()
            for chunk in context.chunks:
                doc_id = chunk.id.split("::", 1)[0]
                retrieved_doc_ids.add(doc_id)
            # Cited source document IDs, deduplicated in citation order: the
            # same document cited twice is one source, not two.
            cited_doc_ids = list(dict.fromkeys(
                s.chunk_id.split("::", 1)[0] for s in answer_result.sources
            ))
            answer = answer_result.answer
            status = "answered" if answer.strip() else "no_answer"
            error = None
        except Exception as exc:
            answer = None
            status = "error"
            retrieved_doc_ids = set()
            cited_doc_ids = []
            error = str(exc)

        lexical_match = score_answer(answer, case)
        correct = lexical_match and status == "answered"
        # Citation presence only; no per-fact entailment check.
        grounded = correct and bool(cited_doc_ids)
        recall, precision, rec_num, rec_den = _compute_evidence_metrics(
            retrieved_doc_ids, case["required_page_ids"], cited_doc_ids,
            case.get("alternative_evidence_sets"),
        )
        results.append(CaseResult(
            id=case["id"],
            question=query,
            answer=answer,
            answer_correct=correct,
            lexical_match=lexical_match,
            grounded_correct=grounded,
            unsupported_correct=lexical_match and not grounded,
            status=status,
            sources=cited_doc_ids,
            accessed_pages=sorted(retrieved_doc_ids),
            required_pages=case["required_page_ids"],
            evidence_recall=recall,
            source_precision=precision,
            steps=1,  # baseline is single-pass
            pages_read=len(retrieved_doc_ids),
            answerable=case["answerable"],
            scoring_rule="keyword_group_match",
            link_source="none",  # baseline has no wiki links
            evidence_recall_numerator=rec_num,
            evidence_recall_denominator=rec_den,
            # Baseline has no Agent search/read steps. Its single-pass
            # pipeline retrieves and consumes this context, so both diagnostic
            # sets are the retrieved document set.
            search_hit_page_ids=sorted(retrieved_doc_ids),
            read_page_ids=sorted(retrieved_doc_ids),
            access_recall=recall,
            access_recall_numerator=rec_num,
            access_recall_denominator=rec_den,
            read_evidence_recall=recall,
            read_evidence_recall_numerator=rec_num,
            read_evidence_recall_denominator=rec_den,
            error=error,
        ))
    return results


# ---------------------------------------------------------------------------
# Experiment runner (requires live LLM)
# ---------------------------------------------------------------------------


def run_experiment(
    cases: list[dict],
    wiki_index: WikiIndex,
    *,
    base_url: str,
    model: str,
    api_key: str | None,
    max_steps: int = AGENT_MAX_STEPS,
    link_source: str = "manual",
    checkpoint_path: Path | None = None,
) -> list[CaseResult]:
    """Run WikiAgent over all cases. Requires a live model endpoint.

    link_source: "manual" (pages built with _PAGE_LINKS) or "auto"
    (pages resolved via resolve_candidate_links). Recorded per-case for
    auditability.

    checkpoint_path: if provided, each completed CaseResult is appended as a
    JSON line immediately after the case finishes. This protects against
    losing all results if the process is interrupted mid-run. The checkpoint
    file is separate from the final result JSON. Write failures are reported
    to stderr but do not stop the evaluation.
    """
    results = []
    for case in cases:
        monitor = MonitoringWikiIndex(wiki_index)
        agent = WikiAgent(
            monitor,
            base_url=base_url,
            model=model,
            api_key=api_key,
            max_steps=max_steps,
            search_top_k=AGENT_SEARCH_TOP_K,
        )
        search_hit_page_ids: list[str] = []
        read_page_ids: list[str] = []
        try:
            agent_result = agent.answer(case["question"])
            answer = agent_result.answer
            status = agent_result.status
            error = agent_result.error
            # Sources are page paths; resolve them through the index so nested
            # paths (e.g. "wikipedia/11675" -> id "wikipedia-11675") are not
            # truncated by string splitting. Duplicates keep first position.
            source_page_ids = _source_ids_from_paths(
                agent_result.sources, wiki_index
            )
            steps = len(agent_result.steps)
            pages_read = agent_result.pages_read
            search_hit_page_ids = sorted({
                page_id
                for s in agent_result.steps
                for page_id in getattr(s, "search_hit_page_ids", ())
            })
            read_page_ids = sorted(set(
                getattr(agent_result, "read_page_ids", ()) or ()
            ))
            step_details = [
                {
                    "step": s.step,
                    "action": s.action,
                    "params": getattr(s, "params", {}),
                    "error": s.error,
                    "result_summary": s.result_summary,
                    "raw_output": s.raw_output,
                    "read_page_ids": list(getattr(s, "read_page_ids", ())),
                    "search_hit_page_ids": list(
                        getattr(s, "search_hit_page_ids", ())
                    ),
                }
                for s in agent_result.steps
            ]
        except Exception as exc:
            answer = None
            status = "error"
            error = str(exc)
            source_page_ids = []
            steps = 0
            pages_read = 0
            step_details = []

        lexical_match = score_answer(answer, case)
        correct = lexical_match and status == AgentStatus.ANSWERED
        # Citation presence only; no per-fact entailment check.
        grounded = correct and bool(source_page_ids)
        recall, precision, rec_num, rec_den = _compute_evidence_metrics(
            monitor.accessed_page_ids, case["required_page_ids"], source_page_ids,
            case.get("alternative_evidence_sets"),
        )
        read_recall, read_rec_num, read_rec_den = _compute_recall(
            set(read_page_ids), case["required_page_ids"],
            case.get("alternative_evidence_sets"),
        )
        case_result = CaseResult(
            id=case["id"],
            question=case["question"],
            answer=answer,
            answer_correct=correct,
            lexical_match=lexical_match,
            grounded_correct=grounded,
            unsupported_correct=lexical_match and not grounded,
            status=status,
            sources=source_page_ids,
            accessed_pages=sorted(monitor.accessed_page_ids),
            required_pages=case["required_page_ids"],
            evidence_recall=recall,
            source_precision=precision,
            steps=steps,
            pages_read=pages_read,
            answerable=case["answerable"],
            scoring_rule="keyword_group_match",
            link_source=link_source,
            evidence_recall_numerator=rec_num,
            evidence_recall_denominator=rec_den,
            error=error,
            step_details=step_details,
            search_hit_page_ids=search_hit_page_ids,
            read_page_ids=read_page_ids,
            access_recall=recall,
            access_recall_numerator=rec_num,
            access_recall_denominator=rec_den,
            read_evidence_recall=read_recall,
            read_evidence_recall_numerator=read_rec_num,
            read_evidence_recall_denominator=read_rec_den,
        )
        results.append(case_result)

        if checkpoint_path is not None:
            _append_checkpoint(checkpoint_path, case_result)

    return results


def _source_ids_from_paths(paths: tuple[str, ...], index: WikiIndex) -> list[str]:
    """Map AgentResult source paths to page ids, preserving citation order.

    Uses the real index so ids with slashes survive. Falls back to the old
    last-segment heuristic only if a path is missing from the index.
    """
    ids: list[str] = []
    for path in paths:
        page = index.get_by_path(path)
        page_id = page.id if page is not None else path.rstrip("/").split("/")[-1]
        if page_id not in ids:
            ids.append(page_id)
    return ids


def _append_checkpoint(checkpoint_path: Path, case_result: CaseResult) -> None:
    """Append one CaseResult as a JSON line to the checkpoint file.

    Failures are printed to stderr but never raised — the in-memory results
    remain valid and the final JSON will still be written.
    """
    try:
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        with open(checkpoint_path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(case_result.to_dict(), ensure_ascii=False) + "\n")
    except OSError as exc:
        print(
            f"WARNING: failed to write checkpoint for case {case_result.id!r}: {exc}",
            file=sys.stderr,
        )


# ---------------------------------------------------------------------------
# Answer leakage check
# ---------------------------------------------------------------------------


def check_answer_leakage(dataset: dict, pages: list[WikiPage]) -> list[str]:
    """Verify that reference answers don't appear verbatim in page content.

    Returns a list of warnings (empty if no issues found). This is a
    heuristic check — it catches exact reference-answer strings in content,
    not paraphrased leaks.
    """
    warnings = []
    page_content = {p.id: _normalize(p.content) for p in pages}
    for case in dataset["cases"]:
        ref = _normalize(case["reference_answer"])
        # Check if the full reference answer (normalized) appears in any page.
        for pid, content in page_content.items():
            if ref and ref in content:
                warnings.append(
                    f"case {case['id']}: full reference answer appears in page {pid}"
                )
    return warnings


# ---------------------------------------------------------------------------
# Result serialization
# ---------------------------------------------------------------------------


def _git_sha() -> str:
    try:
        import subprocess
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(ROOT), capture_output=True, text=True, timeout=5,
        )
        return result.stdout.strip() if result.returncode == 0 else "unknown"
    except Exception:
        return "unknown"


def _dataset_hash() -> str:
    return hashlib.sha256(DATASET_PATH.read_bytes()).hexdigest()[:16]


def _working_tree_status() -> dict:
    """Detect whether the working tree has uncommitted or untracked files.

    Returns a dict with 'dirty' (bool) and 'file_count' (int). This is
    important because git_sha only records HEAD, while all ZMWiki code is
    currently untracked — HEAD alone would misleadingly suggest a clean tree.
    """
    try:
        import subprocess
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=str(ROOT), capture_output=True, text=True, timeout=5,
        )
        if result.returncode != 0:
            return {"dirty": None, "file_count": None, "note": "git status failed"}
        lines = [l for l in result.stdout.splitlines() if l.strip()]
        return {"dirty": len(lines) > 0, "file_count": len(lines)}
    except Exception:
        return {"dirty": None, "file_count": None, "note": "git status unavailable"}


#: Key source files whose content hash is recorded for reproducibility.
_SOURCE_FILES_FOR_HASH = [
    "benchmark/run_multihop_evaluation.py",
    "src/zmwiki/wiki/wiki_agent.py",
    "src/zmwiki/wiki/wiki_compiler.py",
    "src/zmwiki/wiki/wiki_index.py",
    "src/zmwiki/wiki/wiki_retriever.py",
    "src/zmwiki/wiki/wiki_schema.py",
    "src/zmwiki/wiki/wiki_link_resolver.py",
    "src/zmwiki/llm_generators.py",
]


def _source_file_hashes() -> dict[str, str]:
    """Compute SHA-256 (first 16 chars) of key source files for reproducibility."""
    hashes = {}
    for rel_path in _SOURCE_FILES_FOR_HASH:
        abs_path = ROOT / rel_path
        if abs_path.exists():
            hashes[rel_path] = hashlib.sha256(abs_path.read_bytes()).hexdigest()[:16]
        else:
            hashes[rel_path] = "missing"
    return hashes


def serialize_results(
    dataset: dict,
    baseline_results: list[CaseResult],
    experiment_results: list[CaseResult] | None,
    *,
    live: bool,
    model: str | None,
    output_path: Path,
    link_source: str = "manual",
    link_resolution: dict | None = None,
    base_url: str | None = None,
    limit: int | None = None,
) -> None:
    cases = dataset["cases"]
    payload = {
        "experiment_id": EXPERIMENT_ID,
        "dataset": "benchmark/datasets/multihop_wiki.json",
        "dataset_version": dataset.get("version", "unknown"),
        "dataset_sha256_16": _dataset_hash(),
        "git_sha": _git_sha(),
        "python_version": platform.python_version(),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "scoring_rule": dataset.get("scoring_rule", "unknown"),
        "link_construction": dataset.get("link_construction", "unknown"),
        "link_source": link_source,
        "comparability_notes": {
            "retrieval_evidence_metrics": "COMPARABLE across Baseline and Experiment. "
                "Both measure whether required pages were found (evidence_recall, source_precision, "
                "multi_hop_success), independent of the answer generator. evidence_recall is "
                "the legacy name for access_recall (accessed_pages also includes search "
                "candidate lookups). read_evidence_recall uses only pages actually read by "
                "the Agent; for the baseline, read_page_ids is its retrieved context set.",
            "answer_quality_metrics": "NOT directly comparable. Baseline uses DeterministicGenerator "
                "(offline extractive, copies text from retrieved chunks). Experiment uses a real LLM. "
                "Differences in answer_accuracy may come from retrieval, evidence organization, OR the "
                "generator itself. Report as independent system results, not a head-to-head comparison.",
            "scoring_caveat": "Answer scoring uses keyword_group_match (lexical heuristic). "
                "A semantically correct paraphrase may score 0; a verbatim copy may score 1. "
                "This is not an LLM-as-judge evaluation.",
            "unanswerable_caveat": f"{sum(1 for c in dataset['cases'] if not c['answerable'])} "
                "unanswerable case(s). Keyword-only refusal scoring can accept a refusal "
                "followed by a fabricated answer; unanswerable results require manual audit.",
            "metric_update_status_aware": "answer_correct and answer_accuracy now "
                "require status=='answered' (status-aware formal correctness). The "
                "previous keyword-only value is preserved per case in lexical_match and "
                "in aggregate lexical_match_rate/count. Older result files keep their "
                "original answer_correct values; compare them against lexical_match or "
                "re-score with the current code. unsupported_correct exposes lexical "
                "matches without a cited, successfully-answered basis.",
            "source_deduplication": "Cited source lists are deduplicated before "
                "source_precision is computed. Earlier result files may count the same "
                "document twice, so mean_source_precision is not directly comparable "
                "across that change.",
        },
        "config": {
            "baseline": {
                "pipeline": "DeterministicRAGPipeline",
                "retriever": "HybridRetriever(dense + bm25)",
                "chunk_size": CHUNK_SIZE,
                "overlap": OVERLAP,
                "top_k": TOP_K,
                "generator": "DeterministicGenerator (offline, no LLM)",
            },
            "experiment": {
                "agent": "WikiAgent",
                "live": live,
                "base_url": base_url if live else None,
                "model": model if live else "not_run (offline mode)",
                "max_steps": AGENT_MAX_STEPS if live else None,
                "search_top_k": AGENT_SEARCH_TOP_K if live else None,
                "link_source": link_source,
                "api_key_recorded": False,
            },
            "limit": limit,
        },
        "working_tree_status": _working_tree_status(),
        "source_file_sha256_16": _source_file_hashes(),
        "results": {
            "baseline": {
                "per_case": [r.to_dict() for r in baseline_results],
                "aggregate": aggregate_results(cases, baseline_results),
            },
        },
    }
    if link_resolution is not None:
        payload["link_resolution"] = link_resolution
    if experiment_results is not None:
        payload["results"]["experiment"] = {
            "per_case": [r.to_dict() for r in experiment_results],
            "aggregate": aggregate_results(cases, experiment_results),
        }
    else:
        payload["results"]["experiment"] = {
            "per_case": [],
            "aggregate": None,
            "note": "Experiment not run in offline mode. Use --live to run WikiAgent with a real model.",
        }

    # Refuse to overwrite existing result files. Users must choose a new path.
    if output_path.exists():
        raise FileExistsError(
            f"Output file already exists, refusing to overwrite: {output_path}\n"
            f"Use a different --output path to preserve existing results."
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="ZMWiki multi-hop QA benchmark (baseline vs WikiAgent)")
    parser.add_argument("--live", action="store_true", help="Run WikiAgent with a real LLM")
    parser.add_argument("--base-url", default=None, help="OpenAI-compatible API base URL")
    parser.add_argument("--model", default=None, help="Model identifier for live evaluation")
    parser.add_argument("--api-key", default=None,
                        help="API key for live evaluation (or set ZMWIKI_API_KEY env var). "
                             "If omitted, --allow-anonymous is required to send unauthenticated requests.")
    parser.add_argument("--allow-anonymous", action="store_true",
                        help="Allow --live evaluation to send requests without an API key "
                             "(no Authorization header). Ignored when an API key is configured.")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT), help="Output JSON path")
    parser.add_argument("--max-steps", type=int, default=AGENT_MAX_STEPS)
    parser.add_argument("--limit", type=int, default=None,
                        help="Only evaluate the first N cases from the dataset. "
                             "Useful for controlled smoke tests. Does not modify the dataset file.")
    parser.add_argument("--auto-links", action="store_true",
                        help="Use resolve_candidate_links() instead of manual _PAGE_LINKS "
                             "for the Experiment WikiIndex. Always runs link resolution diagnostic.")
    args = parser.parse_args()

    if args.live and (not args.base_url or not args.model):
        parser.error("--live requires --base-url and --model")

    # Derive output and checkpoint paths early so existence checks happen before
    # any data loading or model calls — no point doing work if the output is blocked.
    output_path = Path(args.output)
    checkpoint_path = output_path.with_name(output_path.stem + ".checkpoint.jsonl")

    # Early output file protection: refuse to overwrite existing results.
    if output_path.exists():
        parser.error(
            f"Output file already exists: {output_path}\n"
            f"Refusing to overwrite. Use a different --output path to preserve existing results."
        )

    # Early checkpoint protection: if running live and a checkpoint from a previous
    # run already exists, refuse to append to it. The formal output may not exist
    # (e.g. previous run was interrupted before final serialization), but the stale
    # checkpoint must not be polluted with new results.
    if args.live and checkpoint_path.exists():
        parser.error(
            f"Checkpoint file already exists from a previous run: {checkpoint_path}\n"
            f"Refusing to append to a stale checkpoint. Use a different --output path "
            f"(which changes the checkpoint path), or move/delete the old checkpoint file manually."
        )

    # Pre-flight API key check (no network requests, never prints the key value).
    if args.live:
        has_key = args.api_key is not None or bool(
            os.environ.get("ZMWIKI_API_KEY", "").strip()
        )
        if not has_key and not args.allow_anonymous:
            parser.error(
                "No API key provided (--api-key or ZMWIKI_API_KEY).\n"
                "Live evaluation requires an API key, or explicit --allow-anonymous "
                "to send requests without an Authorization header.\n"
                "Anonymous requests will fail against endpoints that require authentication."
            )
        if not has_key:
            print("WARNING: No API key provided. Sending anonymous requests "
                  "(no Authorization header) as permitted by --allow-anonymous.")
            print("  If your endpoint requires authentication, all cases will fail.\n")

    print(f"Loading dataset from {DATASET_PATH}...")
    dataset = load_dataset()
    cases = dataset["cases"]
    if args.limit is not None:
        if args.limit < 1:
            parser.error(f"--limit must be >= 1, got {args.limit}")
        if args.limit > len(cases):
            parser.error(f"--limit {args.limit} exceeds dataset size {len(cases)}")
        cases = cases[:args.limit]
        print(f"  Limited to first {len(cases)} cases (--limit {args.limit})")
    else:
        print(f"  {len(cases)} cases loaded")

    print("Loading source documents...")
    documents = load_documents()
    print(f"  {len(documents)} documents")

    link_source = "auto" if args.auto_links else "manual"
    link_resolution: dict | None = None

    if args.auto_links:
        print("Building WikiPages with candidate_links (simulating WikiCompiler output)...")
        raw_pages = build_wiki_pages(documents, use_candidate_links=True)
        print(f"  {len(raw_pages)} pages with candidate_links")
        print("Running resolve_candidate_links() (real resolver)...")
        resolved_pages, report = resolve_candidate_links(raw_pages)
        link_resolution = run_link_resolution(raw_pages)
        print(f"  resolved={report.resolved_count}, unresolved={report.unresolved_count}, "
              f"ambiguous={report.ambiguous_count}, self_skipped={report.self_link_skipped}")
        pages = resolved_pages
        wiki_index = build_wiki_index(pages)
    else:
        print("Building WikiIndex with manually verified links (_PAGE_LINKS)...")
        pages = build_wiki_pages(documents)
        wiki_index = build_wiki_index(pages)

    link_report = wiki_index.check_links()
    print(f"  {wiki_index.count()} pages, link check clean={link_report.is_clean}")
    if not link_report.is_clean:
        print(f"  WARNING: broken_links={link_report.broken_links}, "
              f"inconsistent_links={link_report.inconsistent_links}")

    print("Checking answer leakage...")
    leakage = check_answer_leakage(dataset, pages)
    if leakage:
        for w in leakage:
            print(f"  WARNING: {w}")
    else:
        print("  No exact reference-answer leakage found")

    print("\n=== Baseline (DeterministicRAGPipeline) ===")
    baseline_pipeline = build_baseline_pipeline(documents)
    baseline_results = run_baseline(cases, baseline_pipeline)
    baseline_agg = aggregate_results(cases, baseline_results)
    print(f"  Answer accuracy: {baseline_agg['answer_accuracy']:.3f} "
          f"({baseline_agg['correct_count']}/{baseline_agg['total_cases']})")
    print(f"  Verified answer accuracy: {baseline_agg['verified_answer_accuracy']:.3f} "
          f"(unsupported correct: {baseline_agg['unsupported_correct_count']})")
    print(f"  Mean evidence recall (answerable): {baseline_agg['mean_evidence_recall']:.3f}")
    print(f"  Multi-hop success: {baseline_agg['multi_hop_success_rate']:.3f}")

    experiment_results = None
    if args.live:
        print(f"\n=== Experiment (WikiAgent, model={args.model}, link_source={link_source}) ===")
        print(f"  Checkpoint: {checkpoint_path}")
        experiment_results = run_experiment(
            cases, wiki_index,
            base_url=args.base_url, model=args.model,
            api_key=args.api_key, max_steps=args.max_steps,
            link_source=link_source,
            checkpoint_path=checkpoint_path,
        )
        exp_agg = aggregate_results(cases, experiment_results)
        print(f"  Answer accuracy: {exp_agg['answer_accuracy']:.3f} "
              f"({exp_agg['correct_count']}/{exp_agg['total_cases']})")
        print(f"  Verified answer accuracy: {exp_agg['verified_answer_accuracy']:.3f} "
              f"(unsupported correct: {exp_agg['unsupported_correct_count']})")
        print(f"  Mean evidence recall (answerable): {exp_agg['mean_evidence_recall']:.3f}")
        print(f"  Mean steps: {exp_agg['mean_steps']:.2f}")
        print(f"  Failure rate: {exp_agg['failure_rate']:.3f}")
        print(f"  Status counts: {exp_agg['status_counts']}")
    else:
        print("\n=== Experiment (WikiAgent) ===")
        print("  Skipped in offline mode. Use --live to run with a real model.")

    serialize_results(
        dataset, baseline_results, experiment_results,
        live=args.live, model=args.model, output_path=output_path,
        link_source=link_source, link_resolution=link_resolution,
        base_url=args.base_url, limit=args.limit,
    )
    print(f"\nResults written to {output_path}")


if __name__ == "__main__":
    main()
