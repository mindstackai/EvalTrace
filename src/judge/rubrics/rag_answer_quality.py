from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

from .base import RubricScore


@dataclass(frozen=True, slots=True)
class RagAnswerQualityRubric:
    name: str = "rag_answer_quality"
    instructions: str = (
        "You are grading a RAG system answer. Score each dimension from 0.0 to 1.0. "
        "Prefer evidence-based, grounded answers. Penalize hallucinations and unsupported claims."
    )
    dimensions: List[RubricScore] = (
        RubricScore("correctness", 0.0, 1.0, "Is the answer factually correct given the expected answer and retrieved context?"),
        RubricScore("faithfulness", 0.0, 1.0, "Does the answer rely only on retrieved context without hallucinations or unsupported claims?"),
        RubricScore("relevance", 0.0, 1.0, "Does the answer directly and completely address the user's question?"),
    )

    def as_dict(self) -> Dict:
        return {
            "name": self.name,
            "instructions": self.instructions,
            "dimensions": [
                {
                    "key": d.key,
                    "scale_min": d.scale_min,
                    "scale_max": d.scale_max,
                    "description": d.description,
                }
                for d in self.dimensions
            ],
        }
