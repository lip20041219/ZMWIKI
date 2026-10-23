"""Shared text normalization for the evaluation layers.

One tokenization path for answer evaluation and claim attribution, so the two
can never drift into measuring different things. Everything here is stdlib plus
the existing :class:`~zmwiki.tokenizers.Tokenizer`: NFC normalization, sentence
splitting, unique-token handling, and token coverage.

Not a NLP toolkit, and deliberately not exported at the package root: these are
internal plumbing for the evaluation modules. No stemming, no stopword list, no
language model, no sentence-boundary model -- a naive terminator scan is the
whole sentence splitter, and its failure modes (decimals and abbreviations
split too) are documented rather than patched with rules this project does not
have.
"""

from __future__ import annotations

import unicodedata
from collections.abc import Sequence

from .tokenizers import Tokenizer

__all__ = [
    "SENTENCE_TERMINATORS",
    "normalize_text",
    "safe_ratio",
    "split_sentences",
    "token_coverage",
    "tokenize_text",
    "unique_tokens",
]

#: Characters that end a sentence: the ASCII and CJK terminators, plus the line
#: break (a newline ends a sentence too, but is dropped rather than kept).
SENTENCE_TERMINATORS = frozenset(".!?;。！？；\n")


def normalize_text(text: str) -> str:
    """NFC-normalize ``text`` so composed and decomposed forms tokenize alike."""
    return unicodedata.normalize("NFC", text)


def tokenize_text(text: str, tokenizer: Tokenizer) -> list[str]:
    """Normalize then tokenize -- the single entry point every metric uses."""
    return tokenizer.tokenize(normalize_text(text))


def unique_tokens(tokens: Sequence[str]) -> list[str]:
    """First-occurrence order preserved; duplicates dropped.

    A word repeated in a claim or a reference is one fact, not two, so every
    coverage measure counts tokens once.
    """
    seen: set[str] = set()
    out: list[str] = []
    for token in tokens:
        if token not in seen:
            seen.add(token)
            out.append(token)
    return out


def split_sentences(text: str) -> list[str]:
    """Split ``text`` into non-empty sentences, keeping the terminator.

    A sentence ends at any :data:`SENTENCE_TERMINATORS` character. Punctuation
    terminators are kept (a claim then reads exactly like the sentence it came
    from); a whitespace terminator is dropped instead. Empty pieces -- e.g. from
    ``"a. . b"`` -- are not sentences and are dropped, so an empty or
    whitespace-only answer yields no sentences at all.

    Naive on purpose: ``"3.5"`` and ``"e.g."`` split too. That can only make
    sentences shorter and more numerous, and it is a documented limitation of a
    baseline that has no sentence-boundary model.
    """
    sentences: list[str] = []
    current: list[str] = []
    for char in text:
        if char in SENTENCE_TERMINATORS:
            piece = "".join(current).strip()
            if piece:
                sentences.append(piece + ("" if char.isspace() else char))
            current = []
        else:
            current.append(char)
    tail = "".join(current).strip()
    if tail:
        sentences.append(tail)
    return sentences


def token_coverage(tokens: Sequence[str], support: Sequence[str]) -> float:
    """Share of ``tokens`` (unique) that appear in ``support``.

    ``0.0`` when there are no tokens to cover -- never a division by zero, never
    NaN. This is the one overlap formula behind lexical groundedness, sentence
    support, and claim support.
    """
    unique = unique_tokens(tokens)
    if not unique:
        return 0.0
    support_tokens = set(support)
    matched = sum(1 for token in unique if token in support_tokens)
    return matched / len(unique)


def safe_ratio(numerator: int, denominator: int) -> float:
    """``numerator / denominator``, or ``0.0`` when the denominator is zero."""
    if denominator == 0:
        return 0.0
    return numerator / denominator
