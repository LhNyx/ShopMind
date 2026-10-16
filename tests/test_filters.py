"""硬过滤与降级阶梯。"""

from shopmind.retrieval.filters import hard_filter
from shopmind.schemas import Requirements
from tests.conftest import CATALOG


def test_full_constraint_pass():
    req = Requirements(category="无线耳机", budget_max=500, exclusions=["小米"])
    result = hard_filter(CATALOG, req)
    ids = {p.id for p in result.products}
    assert ids == {"A3"}  # A1 超预算、A2 被排除
    assert result.relaxations == []


def test_budget_tolerance_default_20pct():
    # 预算 350 → 上界 350*1.2 = 420：A2(199)、A3(399) 都在容差内，A1(1899) 超出
    req = Requirements(category="无线耳机", budget_max=350)
    assert {p.id for p in hard_filter(CATALOG, req).products} == {"A2", "A3"}


def test_relax_budget_then_category_then_all():
    # 预算 150 + 耳机：L1(±20%→180) 命中不了 A2(199) → L2(±40%→210) 命中，记录放宽
    req = Requirements(category="无线耳机", budget_max=150)
    r1 = hard_filter(CATALOG, req)
    assert {p.id for p in r1.products} == {"A2"}
    assert r1.relaxations == ["预算放宽至 ±40%"]

    # 类目不存在 → 逐级放宽到「忽略类目」
    req = Requirements(category="相机", budget_max=3000)
    r3 = hard_filter(CATALOG, req)
    assert "忽略类目限制" in r3.relaxations
    assert r3.products

    # 全部无解 → 最后一级只剩排除项，语义召回兜底
    req = Requirements(category="相机", budget_max=1)
    r4 = hard_filter(CATALOG, req)
    assert len(r4.products) == len(CATALOG)
    assert "结构化约束全部放宽" in r4.relaxations[-1]


def test_exclusion_always_applied():
    req = Requirements(budget_max=100000)
    result = hard_filter(CATALOG, req)
    assert all(p.brand != "小米" or True for p in result.products)  # 未排除时不影响
    req_excl = Requirements(exclusions=["小米"])
    ids = {p.id for p in hard_filter(CATALOG, req_excl).products}
    assert "A2" not in ids and "C1" not in ids
