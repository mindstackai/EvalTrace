"""General-purpose IR and ranking metrics.

These metrics are not RAG-specific — they apply to any information retrieval,
search, or recommendation system.

    from metrics.ranking import hit_rate, mrr, recall_at_k
    from metrics.overlap import text_overlap_ratio
"""

from .ranking import hit_rate, mrr, recall_at_k
from .overlap import text_overlap_ratio

__all__ = ["hit_rate", "mrr", "recall_at_k", "text_overlap_ratio"]
