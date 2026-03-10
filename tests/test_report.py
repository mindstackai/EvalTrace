import json

from spanrecorder import SpanRecorder
from judge.rubrics.rag_answer_quality import RagAnswerQualityRubric
from report.full_report import run_full_eval, format_report


class FakeJudgeClient:
    def complete(self, prompt: str) -> str:
        return json.dumps({
            "scores": {"correctness": 4, "grounding": 5, "completeness": 3, "clarity": 4},
            "overall": 4,
            "rationale": "Good answer",
        })


def _make_trace(recorder):
    """Create a single realistic trace and return its spans."""
    recorder.reset()
    with recorder.start_span("request", attrs={
        "component": "app",
        "user.query": "What is fleet management?",
    }):
        with recorder.start_span("retrieval.search", attrs={
            "component": "retriever",
            "retrieval.passages": ["Fleet management is...", "It involves..."],
        }):
            pass
        with recorder.start_span("llm.generation", attrs={
            "component": "llm",
            "assistant.answer": "Fleet management is the coordination of vehicles.",
            "llm.model": "gpt-4o-mini",
            "llm.prompt_tokens": 200,
            "llm.completion_tokens": 50,
        }):
            pass
    return recorder.get_spans()


def test_run_full_eval_single_trace():
    rec = SpanRecorder()
    spans = _make_trace(rec)

    report = run_full_eval(
        traces=[spans],
        judge_client=FakeJudgeClient(),
        rubric=RagAnswerQualityRubric(),
    )

    assert len(report.judge_results) == 1
    assert report.judge_results[0].scores["correctness"] == 4
    assert len(report.latency_features) == 1
    assert report.latency_report is not None
    assert report.cost_report is not None
    assert report.cost_report.total_tokens == 250


def test_run_full_eval_with_slo():
    from latency.slo import LatencySLO

    rec = SpanRecorder()
    spans = _make_trace(rec)

    slos = [
        LatencySLO(name="e2e_p95", p95_ms_max=50000),  # generous — should pass
        LatencySLO(name="tight_slo", p95_ms_max=0.001),  # tiny — should fail
    ]

    report = run_full_eval(
        traces=[spans],
        judge_client=FakeJudgeClient(),
        rubric=RagAnswerQualityRubric(),
        slo_configs=slos,
    )

    assert len(report.slo_results) == 2
    assert report.slo_results[0].passed is True
    assert report.slo_results[1].passed is False


def test_format_report_produces_output():
    rec = SpanRecorder()
    spans = _make_trace(rec)

    report = run_full_eval(
        traces=[spans],
        judge_client=FakeJudgeClient(),
        rubric=RagAnswerQualityRubric(),
    )

    text = format_report(report)
    assert "Judge Evaluation" in text
    assert "Latency Report" in text
    assert "Cost Report" in text
    assert "correctness" in text


def test_to_dict_roundtrip():
    rec = SpanRecorder()
    spans = _make_trace(rec)

    report = run_full_eval(
        traces=[spans],
        judge_client=FakeJudgeClient(),
        rubric=RagAnswerQualityRubric(),
    )

    d = report.to_dict()
    assert isinstance(d, dict)
    assert len(d["judge_results"]) == 1
    assert d["cost_report"]["total_tokens"] == 250
