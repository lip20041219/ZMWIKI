"""Minimal lexical tokenization for BM25 retrieval.

Stdlib-only and deterministic. This is an infrastructure tokenizer, not an
NLP pipeline: no stemming, no POS, no dictionary/CRF/jieba. Chinese text is
handled at character granularity so it stays retrievable -- it is explicitly
*not* semantic word segmentation.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

__all__ = ["Tokenizer", "SimpleTokenizer"]


class Tokenizer(ABC):
    """Splits text into a deterministic token sequence."""

    @abstractmethod
    def tokenize(self, text: str) -> list[str]:
        """Return the tokens of ``text``. Same input -> same output."""


class SimpleTokenizer(Tokenizer):
    """Deterministic, dependency-free tokenizer.

    Strategy per code point:
      - a run of ASCII letters/digits -> one token, lowercased
        (``"Hello123"`` -> ``["hello123"]``);
      - any other Unicode alphanumeric code point (CJK ideographs, kana,
        hangul, accented Latin, ...) -> one single-character token;
      - everything else (whitespace, punctuation, symbols, emoji) is dropped.

    Consequence: a Latin word containing a non-ASCII letter splits, e.g.
    ``"café"`` -> ``["caf", "é"]``, and Chinese ``"苹果"`` -> ``["苹", "果"]``.
    Deterministic and identical across processes; no external services.
    """

    def tokenize(self, text: str) -> list[str]:
        if not isinstance(text, str):
            raise TypeError(f"text must be a str, got {type(text).__name__}")
        tokens: list[str] = []
        latin: list[str] = []
        for ch in text:
            if "a" <= ch <= "z" or "A" <= ch <= "Z" or "0" <= ch <= "9":
                latin.append(ch)
            else:
                if latin:
                    tokens.append("".join(latin).lower())
                    latin = []
                if ord(ch) > 127 and ch.isalnum():
                    tokens.append(ch.lower())
        if latin:
            tokens.append("".join(latin).lower())
        return tokens
