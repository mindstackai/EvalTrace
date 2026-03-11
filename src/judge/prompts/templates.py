from __future__ import annotations

import json
from typing import Any, Dict

from judge.rubrics.base import Rubric


def build_judge_prompt(payload: Dict[str, Any], rubric: Rubric) -> str:
    rubric_obj = rubric.as_dict() if hasattr(rubric, "as_dict") else {"name": rubric.name}
    return (
        "SYSTEM:\n"
        "You are an impartial evaluator. Follow the rubric strictly. "
        "Return ONLY valid JSON.\n\n"
        "RUBRIC:\n"
        f"{json.dumps(rubric_obj, ensure_ascii=False, indent=2)}\n\n"
        "EVALUATION_INPUT:\n"
        f"{json.dumps(payload, ensure_ascii=False, indent=2)}\n\n"
        "OUTPUT_JSON_SCHEMA:\n"
        "{\n"
        "  \"scores\": { \"<dimension_key>\": <float 0.0-1.0> },\n"
        "  \"overall\": <float 0.0-1.0>,\n"
        "  \"rationale\": \"<short reasoning>\"\n"
        "}\n"
    )
