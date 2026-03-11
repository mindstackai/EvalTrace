"""Tests for metrics.overlap — text overlap and content-based matching."""
from metrics.overlap import text_overlap_ratio, is_content_match, content_reciprocal_rank


class TestTextOverlapRatio:
    def test_identical(self):
        assert text_overlap_ratio("hello world", "hello world") == 1.0

    def test_no_overlap(self):
        assert text_overlap_ratio("hello world", "foo bar") == 0.0

    def test_partial_overlap(self):
        ratio = text_overlap_ratio("the cat sat", "the dog sat on a mat")
        # shorter = {"the", "cat", "sat"} (3 words)
        # overlap with longer = {"the", "sat"} (2 words)
        assert abs(ratio - 2 / 3) < 1e-9

    def test_case_insensitive(self):
        assert text_overlap_ratio("Hello World", "hello world") == 1.0

    def test_empty_a(self):
        assert text_overlap_ratio("", "hello") == 0.0

    def test_empty_b(self):
        assert text_overlap_ratio("hello", "") == 0.0

    def test_both_empty(self):
        assert text_overlap_ratio("", "") == 0.0

    def test_subset(self):
        # shorter is fully contained in longer
        assert text_overlap_ratio("cat", "the cat sat on a mat") == 1.0


class TestIsContentMatch:
    def test_match(self):
        assert is_content_match("the cat sat", ["the cat sat on a mat"]) is True

    def test_no_match(self):
        assert is_content_match("hello world", ["foo bar baz"]) is False

    def test_empty_references(self):
        assert is_content_match("hello", []) is False

    def test_threshold(self):
        # overlap ratio = 2/3 ≈ 0.667
        assert is_content_match("the cat sat", ["the dog sat on mat"], threshold=0.5) is True
        assert is_content_match("the cat sat", ["the dog sat on mat"], threshold=0.9) is False

    def test_multiple_references(self):
        # First ref doesn't match, second does
        refs = ["completely different text here", "the cat sat on a mat"]
        assert is_content_match("the cat sat", refs) is True


class TestContentReciprocalRank:
    def test_first_rank(self):
        retrieved = ["the cat sat on mat", "the dog ran"]
        expected = ["the cat sat on a mat"]
        assert content_reciprocal_rank(retrieved, expected) == 1.0

    def test_second_rank(self):
        retrieved = ["something else entirely different words", "the cat sat on mat"]
        expected = ["the cat sat on a mat"]
        assert content_reciprocal_rank(retrieved, expected) == 0.5

    def test_no_match(self):
        retrieved = ["foo bar", "baz qux"]
        expected = ["the cat sat on a mat"]
        assert content_reciprocal_rank(retrieved, expected) == 0.0

    def test_empty_retrieved(self):
        assert content_reciprocal_rank([], ["some text"]) == 0.0
