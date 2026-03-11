from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class JudgeResult:
    trace_id: str
    rubric_name: str
    scores: Dict[str, float]
    overall: Optional[float] = None
    rationale: Optional[str] = None
    raw: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "trace_id": self.trace_id,
            "rubric_name": self.rubric_name,
            "scores": dict(self.scores),
            "overall": self.overall,
            "rationale": self.rationale,
            "raw": self.raw,
        }


def normalize_judge_output(trace_id: str, rubric_name: str, output: Dict[str, Any]) -> JudgeResult:
    scores = output.get("scores") or {}
    norm_scores: Dict[str, float] = {}
    for k, v in scores.items():
        try:
            norm_scores[str(k)] = float(v)
        except Exception:
            continue

    overall = output.get("overall")
    try:
        overall = float(overall) if overall is not None else None
    except Exception:
        overall = None

    rationale = output.get("rationale")
    if rationale is not None:
        rationale = str(rationale)

    return JudgeResult(
        trace_id=trace_id,
        rubric_name=rubric_name,
        scores=norm_scores,
        overall=overall,
        rationale=rationale,
        raw=output,
    )
