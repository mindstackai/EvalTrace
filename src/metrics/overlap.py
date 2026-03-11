"""Text overlap metrics for content-based matching.

Useful when exact ID matching isn't possible — e.g., comparing chunks
produced by different chunking strategies where IDs differ but content
overlaps.
"""
from __future__ import annotations

from typing import Sequence


def text_overlap_ratio(text_a: str, text_b: str) -> float:
    """Compute word-level overlap ratio between two texts.

    Returns the fraction of words in the shorter text that also appear
    in the longer text. Case-insensitive.

    Args:
        text_a: First text.
        text_b: Second text.

    Returns:
        Overlap ratio as a float in [0, 1]. Returns 0.0 if either text is empty.
    """
    words_a = set(text_a.lower().split())
    words_b = set(text_b.lower().split())
    if not words_a or not words_b:
        return 0.0
    shorter, longer = (words_a, words_b) if len(words_a) <= len(words_b) else (words_b, words_a)
    return len(shorter & longer) / len(shorter)


def is_content_match(
    candidate: str,
    references: Sequence[str],
    threshold: float = 0.5,
) -> bool:
    """Check if a candidate text overlaps significantly with any reference text.

    Args:
        candidate: Text to check.
        references: List of reference texts to compare against.
        threshold: Minimum overlap ratio to count as a match.

    Returns:
        True if the candidate matches at least one reference.
    """
    for ref in references:
        if text_overlap_ratio(candidate, ref) >= threshold:
            return True
    return False


def content_reciprocal_rank(
    retrieved_texts: Sequence[str],
    expected_texts: Sequence[str],
    threshold: float = 0.5,
) -> float:
    """Compute reciprocal rank using content overlap instead of exact ID matching.

    Args:
        retrieved_texts: Ordered list of retrieved text contents (best first).
        expected_texts: List of expected/reference text contents.
        threshold: Minimum overlap ratio to count as a match.

    Returns:
        Reciprocal rank as a float in [0, 1]. Returns 0.0 if no match found.
    """
    for rank, text in enumerate(retrieved_texts, start=1):
        if is_content_match(text, expected_texts, threshold):
            return 1.0 / rank
    return 0.0
