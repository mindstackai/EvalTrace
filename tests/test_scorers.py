"""Tests for judge.scorers — standalone LLM-as-judge scoring functions."""
from judge.scorers import _parse_score, correctness_score, faithfulness_score, relevance_score


class FakeClient:
    """Returns a predetermined score string."""

    def __init__(self, response: str):
        self.response = response
        self.last_prompt = None

    def complete(self, prompt: str) -> str:
        self.last_prompt = prompt
        return self.response


class TestParseScore:
    def test_clean_float(self):
        assert _parse_score("0.85") == 0.85

    def test_clean_int(self):
        assert _parse_score("1") == 1.0

    def test_with_whitespace(self):
        assert _parse_score("  0.7  \n") == 0.7

    def test_embedded_in_text(self):
        assert _parse_score("The score is 0.6 out of 1.0") == 0.6

    def test_clamp_above_one(self):
        assert _parse_score("1.5") == 1.0

    def test_clamp_below_zero(self):
        assert _parse_score("-0.3") == 0.0

    def test_no_number(self):
        assert _parse_score("no score here") == 0.0

    def test_zero(self):
        assert _parse_score("0.0") == 0.0


class TestCorrectnessScore:
    def test_returns_score(self):
        client = FakeClient("0.9")
        score = correctness_score("Paris", "Paris is the capital of France", client)
        assert score == 0.9

    def test_prompt_contains_inputs(self):
        client = FakeClient("0.5")
        correctness_score("my answer", "expected answer", client)
        assert "my answer" in client.last_prompt
        assert "expected answer" in client.last_prompt


class TestFaithfulnessScore:
    def test_returns_score(self):
        client = FakeClient("0.8")
        score = faithfulness_score("grounded answer", ["chunk 1", "chunk 2"], client)
        assert score == 0.8

    def test_prompt_contains_chunks(self):
        client = FakeClient("0.7")
        faithfulness_score("answer", ["first chunk", "second chunk"], client)
        assert "first chunk" in client.last_prompt
        assert "second chunk" in client.last_prompt


class TestRelevanceScore:
    def test_returns_score(self):
        client = FakeClient("0.75")
        score = relevance_score("What is X?", "X is a thing", client)
        assert score == 0.75

    def test_prompt_contains_inputs(self):
        client = FakeClient("0.6")
        relevance_score("the question", "the answer", client)
        assert "the question" in client.last_prompt
        assert "the answer" in client.last_prompt
