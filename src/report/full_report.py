from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from spanrecorder.span import Span
from judge.extract import build_judge_request
from judge.runner import JudgeClient, run_judge
from judge.scoring import JudgeResult
from judge.rubrics.base import Rubric
from latency.extract import extract_latency, LatencyFeatures
from latency.aggregate import aggregate_latency, LatencyReport
from latency.slo import LatencySLO, SLOResult, evaluate_latency_slo
from cost.tracker import extract_cost_from_spans, CostReport


@dataclass
class FullEvalReport:
    judge_results: List[JudgeResult] = field(default_factory=list)
    latency_features: List[LatencyFeatures] = field(default_factory=list)
    latency_report: Optional[LatencyReport] = None
    slo_results: List[SLOResult] = field(default_factory=list)
    cost_report: Optional[CostReport] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "judge_results": [j.to_dict() for j in self.judge_results],
            "latency_features": [l.to_dict() for l in self.latency_features],
            "latency_report": self.latency_report.to_dict() if self.latency_report else None,
            "slo_results": [
                {"slo_name": s.slo_name, "passed": s.passed,
                 "observed_p95_ms": s.observed_p95_ms, "budget_ms": s.budget_ms}
                for s in self.slo_results
            ],
            "cost_report": self.cost_report.to_dict() if self.cost_report else None,
        }


def run_full_eval(
    traces: List[List[Span]],
    judge_client: JudgeClient,
    rubric: Rubric,
    slo_configs: Optional[List[LatencySLO]] = None,
) -> FullEvalReport:
    report = FullEvalReport()

    all_spans: List[Span] = []
    for spans in traces:
        all_spans.extend(spans)

        # Judge
        try:
            req = build_judge_request(spans, rubric)
            result = run_judge(req, judge_client)
            report.judge_results.append(result)
        except Exception:
            pass

        # Latency per trace
        try:
            lf = extract_latency(spans)
            report.latency_features.append(lf)
        except Exception:
            pass

    # Aggregate latency
    if report.latency_features:
        report.latency_report = aggregate_latency(report.latency_features)

        # SLO evaluation
        if slo_configs:
            for slo in slo_configs:
                report.slo_results.append(
                    evaluate_latency_slo(report.latency_report, slo)
                )

    # Cost
    report.cost_report = extract_cost_from_spans(all_spans)

    return report


def format_report(report: FullEvalReport) -> str:
    lines: List[str] = []
    w = 55

    # --- Judge scores ---
    if report.judge_results:
        lines.append("=" * w)
        lines.append(" Judge Evaluation")
        lines.append("=" * w)
        for jr in report.judge_results:
            lines.append(f"  Trace: {jr.trace_id[:12]}...")
            for dim, score in jr.scores.items():
                lines.append(f"    {dim:<25} {score:>5}")
            if jr.overall is not None:
                lines.append(f"    {'overall':<25} {jr.overall:>5}")
            if jr.rationale:
                lines.append(f"    Rationale: {jr.rationale}")
            lines.append("")

        # Average scores across traces
        if len(report.judge_results) > 1:
            all_dims: Dict[str, List[int]] = {}
            for jr in report.judge_results:
                for dim, score in jr.scores.items():
                    all_dims.setdefault(dim, []).append(score)
            lines.append("  --- Averages ---")
            for dim, scores in all_dims.items():
                avg = sum(scores) / len(scores)
                lines.append(f"    {dim:<25} {avg:>5.2f}")
            lines.append("")

    # --- Latency ---
    if report.latency_report:
        lr = report.latency_report
        lines.append("=" * w)
        lines.append(" Latency Report")
        lines.append("=" * w)
        lines.append(f"  {'Metric':<30} {'Value':>15}")
        lines.append("-" * w)
        lines.append(f"  {'Traces':<30} {lr.n:>15}")
        lines.append(f"  {'P50 (ms)':<30} {lr.p50_ms:>15.1f}")
        lines.append(f"  {'P95 (ms)':<30} {lr.p95_ms:>15.1f}")
        lines.append(f"  {'P99 (ms)':<30} {lr.p99_ms:>15.1f}")
        lines.append(f"  {'Avg (ms)':<30} {lr.avg_ms:>15.1f}")
        if lr.component_avg_ms:
            lines.append("")
            lines.append("  Component breakdown (avg ms):")
            for comp, ms in sorted(lr.component_avg_ms.items()):
                lines.append(f"    {comp:<28} {ms:>15.1f}")
        lines.append("")

    # --- SLO ---
    if report.slo_results:
        lines.append("=" * w)
        lines.append(" SLO Results")
        lines.append("=" * w)
        for sr in report.slo_results:
            status = "PASS" if sr.passed else "FAIL"
            lines.append(
                f"  [{status}] {sr.slo_name:<25} "
                f"observed={sr.observed_p95_ms:.1f}ms  budget={sr.budget_ms:.1f}ms"
            )
        lines.append("")

    # --- Cost ---
    if report.cost_report and report.cost_report.total_tokens > 0:
        cr = report.cost_report
        lines.append("=" * w)
        lines.append(" Cost Report")
        lines.append("=" * w)
        lines.append(f"  {'Prompt tokens':<30} {cr.prompt_tokens:>15,}")
        lines.append(f"  {'Completion tokens':<30} {cr.completion_tokens:>15,}")
        lines.append(f"  {'Total tokens':<30} {cr.total_tokens:>15,}")
        lines.append(f"  {'Total cost (USD)':<30} {'$' + f'{cr.total_cost_usd:.6f}':>15}")
        lines.append("")

    return "\n".join(lines)
