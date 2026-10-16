"""指标数学：评测框架的地基。"""

from shopmind.eval.metrics import mrr, recall_at_k


def test_recall_at_k():
    expected = {"A", "B", "C"}
    hits = ["A", "X", "B"]
    assert recall_at_k(expected, hits, k=2) == 1 / 3
    assert recall_at_k(expected, hits, k=3) == 2 / 3
    assert recall_at_k(expected, ["X", "Y"], k=5) == 0.0
    assert recall_at_k(set(), hits, k=5) == 0.0


def test_mrr_uses_best_rank_in_expected_set():
    assert mrr({"B"}, ["X", "B", "A"]) == 0.5
    assert mrr({"A"}, ["A", "B"]) == 1.0
    assert mrr({"Z"}, ["A", "B"]) == 0.0
