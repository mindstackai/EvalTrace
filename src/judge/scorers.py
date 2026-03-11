"""Standalone LLM-as-judge scoring functions for RAG evaluation.

Each function takes text inputs and a JudgeClient, returns a float in [0.0, 1.0].
These are lightweight single-dimension scorers — for multi-dimension evaluation
use the full rubric pipeline (build_judge_request → run_judge).
"""
from __future__ import annotations

import re
from typing import Sequence

from judge.runner import JudgeClient


def _parse_score(raw: str) -> float:
    """Extract a float score from LLM output, clamped to [0.0, 1.0]."""
    raw = raw.strip()
    try:
        return max(0.0, min(1.0, float(raw)))
    except ValueError:
        match = re.search(r"[0-9]+(?:\.[0-9]+)?", raw)
        score = float(match.group()) if match else 0.0
        return max(0.0, min(1.0, score))


def correctness_score(
    answer: str,
    expected_answer: str,
    client: JudgeClient,
) -> float:
    """Score how correct the generated answer is compared to the expected answer.

    Args:
        answer: The generated answer to evaluate.
        expected_answer: The ground-truth / reference answer.
        client: A JudgeClient that takes a prompt string and returns a string.

    Returns:
        Correctness score as a float in [0.0, 1.0].
    """
    prompt = (
        "You are an evaluation assistant. Compare the generated answer to the "
        "expected answer and score how correct and complete the generated answer is.\n\n"
        f"Expected answer:\n{expected_answer}\n\n"
        f"Generated answer:\n{answer}\n\n"
        "Instructions:\n"
        "- Score 1.0 if the generated answer captures all key facts from the expected answer.\n"
        "- Score 0.0 if the generated answer is completely wrong or irrelevant.\n"
        "- Use intermediate values for partial correctness.\n"
        "- Respond with ONLY a single decimal number between 0.0 and 1.0, nothing else.\n\n"
        "Score:"
    )
    return _parse_score(client.complete(prompt))


def faithfulness_score(
    answer: str,
    context_chunks: Sequence[str],
    client: JudgeClient,
) -> float:
    """Score whether the answer is faithfully grounded in the provided context.

    Args:
        answer: The generated answer to evaluate.
        context_chunks: List of retrieved text chunks used to produce the answer.
        client: A JudgeClient that takes a prompt string and returns a string.

    Returns:
        Faithfulness score as a float in [0.0, 1.0].
    """
    context_text = "\n\n".join(
        f"[Chunk {i + 1}]\n{chunk}" for i, chunk in enumerate(context_chunks)
    )
    prompt = (
        "You are an evaluation assistant. Your task is to judge whether an answer "
        "is faithfully grounded in the provided context chunks.\n\n"
        f"Context:\n{context_text}\n\n"
        f"Answer:\n{answer}\n\n"
        "Instructions:\n"
        "- Score 1.0 if every factual claim in the answer is directly supported by the context.\n"
        "- Score 0.0 if the answer contains fabricated or unsupported claims.\n"
        "- Use intermediate values (e.g. 0.5, 0.7) for partial grounding.\n"
        "- Respond with ONLY a single decimal number between 0.0 and 1.0, nothing else.\n\n"
        "Score:"
    )
    return _parse_score(client.complete(prompt))


def relevance_score(
    question: str,
    answer: str,
    client: JudgeClient,
) -> float:
    """Score whether the answer directly addresses the question.

    Args:
        question: The original question posed to the system.
        answer: The generated answer to evaluate.
        client: A JudgeClient that takes a prompt string and returns a string.

    Returns:
        Relevance score as a float in [0.0, 1.0].
    """
    prompt = (
        "You are an evaluation assistant. Your task is to judge how well an answer "
        "addresses the given question.\n\n"
        f"Question:\n{question}\n\n"
        f"Answer:\n{answer}\n\n"
        "Instructions:\n"
        "- Score 1.0 if the answer fully and directly addresses the question.\n"
        "- Score 0.0 if the answer is completely off-topic or does not address the question at all.\n"
        "- Use intermediate values (e.g. 0.5, 0.7) for partial relevance.\n"
        "- Respond with ONLY a single decimal number between 0.0 and 1.0, nothing else.\n\n"
        "Score:"
    )
    return _parse_score(client.complete(prompt))
