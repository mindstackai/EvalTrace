import json
import os
import tempfile

from spanrecorder import SpanRecorder
from storage.trace_store import JsonlTraceStore
from storage.result_store import JsonResultStore


def test_jsonl_trace_store_write_and_read():
    rec = SpanRecorder()
    with rec.start_span("request", attrs={"user.query": "test"}):
        with rec.start_span("retrieval", attrs={"component": "retriever"}):
            pass

    spans = rec.get_spans()

    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "traces.jsonl")
        store = JsonlTraceStore(path=path)
        store.write(spans)

        assert os.path.exists(path)
        with open(path) as f:
            lines = [l.strip() for l in f if l.strip()]
        assert len(lines) == 2

        # Verify each line is valid JSON with expected fields
        for line in lines:
            d = json.loads(line)
            assert "trace_id" in d
            assert "span_id" in d
            assert "name" in d

        # Verify read_all roundtrip
        loaded = store.read_all()
        assert len(loaded) == 2
        assert loaded[0].name == "request"
        assert loaded[0].attributes["user.query"] == "test"
        assert loaded[1].name == "retrieval"
        assert loaded[1].parent_id == loaded[0].span_id


def test_json_result_store_write_and_read():
    with tempfile.TemporaryDirectory() as tmpdir:
        store = JsonResultStore(directory=tmpdir)
        payload = {"scores": {"correctness": 4}, "overall": 4}

        store.write("trace123", "judge", payload)

        result = store.read("trace123", "judge")
        assert result["scores"]["correctness"] == 4
        assert result["overall"] == 4


def test_json_result_store_missing_file():
    import pytest
    with tempfile.TemporaryDirectory() as tmpdir:
        store = JsonResultStore(directory=tmpdir)
        with pytest.raises(FileNotFoundError):
            store.read("nonexistent", "judge")
