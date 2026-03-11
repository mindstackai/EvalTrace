"""AI-as-judge evaluation: rubrics, prompt templates, extraction, and scoring."""
from .runner import JudgeClient, run_judge
from .extract import build_judge_request, JudgeRequest
from .scoring import JudgeResult
from .scorers import correctness_score, faithfulness_score, relevance_score

__all__ = [
    "JudgeClient",
    "run_judge",
    "build_judge_request",
    "JudgeRequest",
    "JudgeResult",
    "correctness_score",
    "faithfulness_score",
    "relevance_score",
]
