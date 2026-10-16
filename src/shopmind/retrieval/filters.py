"""结构化硬过滤：预算 / 类目 / 排除项，确定性代码而非 LLM。

逐级放宽阶梯（每级只放宽一个约束，记录放宽原因进 Trace）：
  L1  预算 ±20% + 类目 + 排除项
  L2  预算 ±40% + 类目 + 排除项      ← 放宽预算
  L3  预算 ±40% + 排除项             ← 忽略类目
  L4  仅排除项                       ← 全交语义召回把关

设计动机（面试考点）：硬过滤保证候选集质量和后续算力成本；
放宽兜底保证「永远有结果」；最终排序里语义相似度仍然把关，
所以放宽不会把完全不相关的商品顶到前面。
"""

from __future__ import annotations

from dataclasses import dataclass, field

from shopmind.schemas import Product, Requirements


@dataclass
class FilterResult:
    products: list[Product] = field(default_factory=list)
    relaxations: list[str] = field(default_factory=list)  # 记录走过的放宽阶梯


def hard_filter(
    products: list[Product],
    req: Requirements,
    budget_tolerance: float = 0.20,
    relax_tolerance: float = 0.40,
) -> FilterResult:
    keep = [p for p in products if p.brand not in req.exclusions]

    # L1：全约束
    tier1 = _budget_in(keep, req.budget_min, req.budget_max, budget_tolerance)
    tier1 = _category_match(tier1, req.category)
    if tier1:
        return FilterResult(tier1)

    # L2：放宽预算
    tier2 = _budget_in(keep, req.budget_min, req.budget_max, relax_tolerance)
    tier2 = _category_match(tier2, req.category)
    if tier2:
        return FilterResult(tier2, relaxations=[f"预算放宽至 ±{int(relax_tolerance * 100)}%"])

    # L3：忽略类目
    tier3 = _budget_in(keep, req.budget_min, req.budget_max, relax_tolerance)
    if tier3:
        return FilterResult(tier3, relaxations=["忽略类目限制"])

    # L4：只剩排除项约束，语义召回兜底
    if keep:
        return FilterResult(keep, relaxations=["结构化约束全部放宽，按语义相关度排序"])
    return FilterResult(keep, relaxations=["目录中无满足排除项的商品"])


def _budget_in(
    products: list[Product], budget_min: float | None, budget_max: float | None, tol: float
) -> list[Product]:
    if budget_min is None and budget_max is None:
        return products
    lo = (budget_min or 0.0) * (1 - tol)
    hi = (budget_max if budget_max is not None else float("inf")) * (1 + tol)
    return [p for p in products if lo <= p.price <= hi]


def _category_match(products: list[Product], category: str | None) -> list[Product]:
    """类目匹配：精确相等或双向子串包含（「耳机」可命中「无线耳机」）。"""
    if not category:
        return products
    return [
        p
        for p in products
        if p.category == category or category in p.category or p.category in category
    ]
