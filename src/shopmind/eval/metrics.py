"""评测指标：Recall@K、MRR、澄清触发准确率。

口径说明（README 报告与面试都要讲清楚）：
- expected_ids 是人工标注的「应命中商品集合」，命中 = 出现在检索返回的 top-K
- MRR 用 expected 集合中最好名次计算（1/最佳排名）
- 澄清触发准确率 = should_clarify 标签与图路由实际决策的一致率；
  决策在代码（compute_missing）里而非 LLM，所以这个指标是稳定可复现的
"""

from __future__ import annotations

from dataclasses import dataclass, field


def recall_at_k(expected: set[str], hits: list[str], k: int) -> float:
    if not expected:
        return 0.0
    return len(expected & set(hits[:k])) / len(expected)


def mrr(expected: set[str], hits: list[str]) -> float:
    for rank, hit in enumerate(hits, start=1):
        if hit in expected:
            return 1.0 / rank
    return 0.0


@dataclass
class ClarifyCase:
    query_id: str
    should_clarify: bool
    actual_clarify: bool


@dataclass
class EvalReport:
    meta: dict
    metrics: dict = field(default_factory=dict)
    rows: list[dict] = field(default_factory=list)
    badcases: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "meta": self.meta,
            "metrics": self.metrics,
            "rows": self.rows,
            "badcases": self.badcases,
        }
