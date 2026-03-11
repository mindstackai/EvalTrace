"""Ranking and retrieval metrics for information retrieval systems.

These metrics are domain-agnostic — they work for search engines,
recommendation systems, RAG retrieval, ad ranking, etc.

All functions operate on ordered lists of item IDs and return floats in [0, 1].
"""
from __future__ import annotations

from typing import List, Sequence


def hit_rate(retrieved_ids: Sequence[str], relevant_ids: Sequence[str]) -> float:
    """Binary hit: did any relevant item appear in the retrieved set?

    Args:
        retrieved_ids: Ordered list of retrieved item IDs.
        relevant_ids: Ground-truth list of relevant item IDs.

    Returns:
        1.0 if at least one relevant item was retrieved, 0.0 otherwise.
    """
    if not relevant_ids:
        return 0.0
    relevant = set(relevant_ids)
    return 1.0 if any(rid in relevant for rid in retrieved_ids) else 0.0


def mrr(retrieved_ids: Sequence[str], relevant_ids: Sequence[str]) -> float:
    """Mean Reciprocal Rank: reciprocal of the rank of the first relevant item.

    For a single query this is 1/rank of the first hit, or 0.0 if no hit.
    To compute MRR across multiple queries, average the per-query values.

    Args:
        retrieved_ids: Ordered list of retrieved item IDs (best first).
        relevant_ids: Ground-truth list of relevant item IDs.

    Returns:
        Reciprocal rank as a float in [0, 1].
    """
    relevant = set(relevant_ids)
    for rank, item_id in enumerate(retrieved_ids, start=1):
        if item_id in relevant:
            return 1.0 / rank
    return 0.0


def recall_at_k(
    retrieved_ids: Sequence[str],
    relevant_ids: Sequence[str],
    k: int,
) -> float:
    """Recall@k: fraction of relevant items found in the top-k retrieved.

    Args:
        retrieved_ids: Ordered list of retrieved item IDs (best first).
        relevant_ids: Ground-truth list of relevant item IDs.
        k: Cut-off rank. Only the first k retrieved items are considered.

    Returns:
        Recall@k as a float in [0, 1]. Returns 0.0 when relevant_ids is empty.
    """
    if not relevant_ids:
        return 0.0
    top_k = set(retrieved_ids[:k])
    relevant = set(relevant_ids)
    return len(top_k & relevant) / len(relevant)


def precision_at_k(
    retrieved_ids: Sequence[str],
    relevant_ids: Sequence[str],
    k: int,
) -> float:
    """Precision@k: fraction of top-k retrieved items that are relevant.

    Args:
        retrieved_ids: Ordered list of retrieved item IDs (best first).
        relevant_ids: Ground-truth list of relevant item IDs.
        k: Cut-off rank. Only the first k retrieved items are considered.

    Returns:
        Precision@k as a float in [0, 1]. Returns 0.0 when k is 0.
    """
    if k == 0:
        return 0.0
    top_k = list(retrieved_ids[:k])
    relevant = set(relevant_ids)
    return sum(1 for rid in top_k if rid in relevant) / k


def average_precision(
    retrieved_ids: Sequence[str],
    relevant_ids: Sequence[str],
) -> float:
    """Average Precision: area under the precision-recall curve for one query.

    Computes precision at each rank where a relevant item appears,
    then averages. Used to compute MAP (Mean Average Precision) across queries.

    Args:
        retrieved_ids: Ordered list of retrieved item IDs (best first).
        relevant_ids: Ground-truth list of relevant item IDs.

    Returns:
        Average precision as a float in [0, 1]. Returns 0.0 when
        relevant_ids is empty.
    """
    if not relevant_ids:
        return 0.0
    relevant = set(relevant_ids)
    hits = 0
    sum_precisions = 0.0
    for rank, item_id in enumerate(retrieved_ids, start=1):
        if item_id in relevant:
            hits += 1
            sum_precisions += hits / rank
    return sum_precisions / len(relevant)
