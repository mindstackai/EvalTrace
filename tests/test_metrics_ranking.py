"""Tests for metrics.ranking — general IR ranking metrics."""
from metrics.ranking import hit_rate, mrr, recall_at_k, precision_at_k, average_precision


class TestHitRate:
    def test_hit(self):
        assert hit_rate(["a", "b", "c"], ["b"]) == 1.0

    def test_miss(self):
        assert hit_rate(["a", "b", "c"], ["x"]) == 0.0

    def test_empty_relevant(self):
        assert hit_rate(["a", "b"], []) == 0.0

    def test_empty_retrieved(self):
        assert hit_rate([], ["a"]) == 0.0

    def test_first_position(self):
        assert hit_rate(["a"], ["a"]) == 1.0


class TestMRR:
    def test_first_rank(self):
        assert mrr(["a", "b", "c"], ["a"]) == 1.0

    def test_second_rank(self):
        assert mrr(["a", "b", "c"], ["b"]) == 0.5

    def test_third_rank(self):
        assert abs(mrr(["a", "b", "c"], ["c"]) - 1 / 3) < 1e-9

    def test_no_hit(self):
        assert mrr(["a", "b", "c"], ["x"]) == 0.0

    def test_multiple_relevant_returns_first(self):
        # Should return reciprocal rank of the *first* relevant item
        assert mrr(["a", "b", "c"], ["b", "c"]) == 0.5

    def test_empty_retrieved(self):
        assert mrr([], ["a"]) == 0.0


class TestRecallAtK:
    def test_full_recall(self):
        assert recall_at_k(["a", "b", "c"], ["a", "b"], k=3) == 1.0

    def test_partial_recall(self):
        assert recall_at_k(["a", "b", "c"], ["a", "x"], k=3) == 0.5

    def test_zero_recall(self):
        assert recall_at_k(["a", "b", "c"], ["x", "y"], k=3) == 0.0

    def test_k_limits_results(self):
        # "c" is relevant but outside top-2
        assert recall_at_k(["a", "b", "c"], ["c"], k=2) == 0.0

    def test_empty_relevant(self):
        assert recall_at_k(["a", "b"], [], k=2) == 0.0

    def test_k_larger_than_retrieved(self):
        assert recall_at_k(["a"], ["a", "b"], k=5) == 0.5


class TestPrecisionAtK:
    def test_all_relevant(self):
        assert precision_at_k(["a", "b"], ["a", "b", "c"], k=2) == 1.0

    def test_half_relevant(self):
        assert precision_at_k(["a", "x"], ["a", "b"], k=2) == 0.5

    def test_none_relevant(self):
        assert precision_at_k(["x", "y"], ["a", "b"], k=2) == 0.0

    def test_k_zero(self):
        assert precision_at_k(["a"], ["a"], k=0) == 0.0


class TestAveragePrecision:
    def test_perfect(self):
        # Both relevant items at rank 1 and 2
        assert average_precision(["a", "b", "c"], ["a", "b"]) == 1.0

    def test_one_relevant_at_rank_1(self):
        assert average_precision(["a", "b", "c"], ["a"]) == 1.0

    def test_one_relevant_at_rank_3(self):
        # precision at rank 3 = 1/3, AP = (1/3) / 1 = 1/3
        assert abs(average_precision(["x", "y", "a"], ["a"]) - 1 / 3) < 1e-9

    def test_two_relevant_spread(self):
        # "a" at rank 1 → precision = 1/1
        # "c" at rank 3 → precision = 2/3
        # AP = (1 + 2/3) / 2 = 5/6
        ap = average_precision(["a", "x", "c"], ["a", "c"])
        assert abs(ap - 5 / 6) < 1e-9

    def test_no_relevant(self):
        assert average_precision(["a", "b"], ["x"]) == 0.0

    def test_empty_relevant(self):
        assert average_precision(["a", "b"], []) == 0.0
