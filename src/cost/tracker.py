from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List

from spanrecorder.span import Span


# USD per 1M tokens  (prompt, completion)
PRICING: Dict[str, tuple] = {
    "gpt-4o":       (2.50, 10.00),
    "gpt-4o-mini":  (0.15,  0.60),
    "gpt-4-turbo":  (10.00, 30.00),
    "gpt-4":        (30.00, 60.00),
    "gpt-3.5-turbo":(0.50,  1.50),
}


@dataclass
class SpanCost:
    span_id: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    cost_usd: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "span_id": self.span_id,
            "model": self.model,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "cost_usd": self.cost_usd,
        }


@dataclass
class CostReport:
    total_tokens: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_cost_usd: float = 0.0
    per_span: List[SpanCost] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_tokens": self.total_tokens,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_cost_usd": self.total_cost_usd,
            "per_span": [s.to_dict() for s in self.per_span],
        }


def _compute_cost(model: str, prompt_tok: int, completion_tok: int) -> float:
    prices = PRICING.get(model)
    if prices is None:
        return 0.0
    prompt_price, completion_price = prices
    return (prompt_tok * prompt_price + completion_tok * completion_price) / 1_000_000


def extract_cost_from_spans(spans: List[Span]) -> CostReport:
    report = CostReport()
    for s in spans:
        attrs = s.attributes or {}
        model = attrs.get("llm.model")
        p_tok = attrs.get("llm.prompt_tokens")
        c_tok = attrs.get("llm.completion_tokens")
        if model is None or p_tok is None or c_tok is None:
            continue
        p_tok = int(p_tok)
        c_tok = int(c_tok)
        cost = _compute_cost(model, p_tok, c_tok)
        report.per_span.append(SpanCost(
            span_id=s.span_id,
            model=model,
            prompt_tokens=p_tok,
            completion_tokens=c_tok,
            cost_usd=cost,
        ))
        report.prompt_tokens += p_tok
        report.completion_tokens += c_tok
        report.total_tokens += p_tok + c_tok
        report.total_cost_usd += cost
    return report
