from spanrecorder import SpanRecorder
from cost.tracker import extract_cost_from_spans, PRICING


def test_extract_cost_from_llm_spans():
    rec = SpanRecorder()
    with rec.start_span("request", attrs={"component": "app"}):
        with rec.start_span("llm.generation", attrs={
            "component": "llm",
            "llm.model": "gpt-4o-mini",
            "llm.prompt_tokens": 100,
            "llm.completion_tokens": 50,
        }):
            pass

    report = extract_cost_from_spans(rec.get_spans())
    assert report.prompt_tokens == 100
    assert report.completion_tokens == 50
    assert report.total_tokens == 150
    assert len(report.per_span) == 1
    assert report.per_span[0].model == "gpt-4o-mini"

    # Verify cost calculation: (100 * 0.15 + 50 * 0.60) / 1_000_000
    expected = (100 * 0.15 + 50 * 0.60) / 1_000_000
    assert abs(report.total_cost_usd - expected) < 1e-10


def test_extract_cost_skips_non_llm_spans():
    rec = SpanRecorder()
    with rec.start_span("retrieval", attrs={"component": "retriever"}):
        pass

    report = extract_cost_from_spans(rec.get_spans())
    assert report.total_tokens == 0
    assert report.total_cost_usd == 0.0
    assert len(report.per_span) == 0


def test_extract_cost_unknown_model():
    rec = SpanRecorder()
    with rec.start_span("llm.call", attrs={
        "llm.model": "unknown-model",
        "llm.prompt_tokens": 100,
        "llm.completion_tokens": 50,
    }):
        pass

    report = extract_cost_from_spans(rec.get_spans())
    assert report.total_tokens == 150
    assert report.total_cost_usd == 0.0  # unknown model = $0


def test_extract_cost_multiple_llm_spans():
    rec = SpanRecorder()
    with rec.start_span("request"):
        with rec.start_span("llm1", attrs={
            "llm.model": "gpt-4o-mini",
            "llm.prompt_tokens": 200,
            "llm.completion_tokens": 100,
        }):
            pass
        with rec.start_span("llm2", attrs={
            "llm.model": "gpt-4o",
            "llm.prompt_tokens": 300,
            "llm.completion_tokens": 150,
        }):
            pass

    report = extract_cost_from_spans(rec.get_spans())
    assert len(report.per_span) == 2
    assert report.prompt_tokens == 500
    assert report.completion_tokens == 250
    assert report.total_cost_usd > 0
