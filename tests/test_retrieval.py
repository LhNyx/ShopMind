"""混合检索黄金集：HashNGram 向量确定性验证「过滤 + 召回」协作。"""

from shopmind.schemas import Requirements


async def test_hybrid_recall_ranks_attribute_match_first(retriever):
    req = Requirements(category="无线耳机", budget_max=500, attributes=["主动降噪"])
    pairs, trace = retriever.retrieve(req)
    hits = [p.id for p, _ in pairs]
    assert hits[0] == "A3"  # 预算内 + 主动降噪 → 应排第一
    assert trace.after_filter == 2  # A2、A3 通过硬过滤（A1 超预算）
    assert trace.total_catalog == 6


async def test_no_constraint_returns_all_categories(retriever):
    pairs, trace = retriever.retrieve(Requirements())
    assert trace.after_filter == 6
    assert len(pairs) == 5  # top_n=5


async def test_exclusion_pushed_out(retriever):
    req = Requirements(category="智能手表", exclusions=["小米"])
    pairs, _ = retriever.retrieve(req)
    assert all(p.brand != "小米" for p, _ in pairs)


async def test_relaxation_recorded_in_trace(retriever):
    req = Requirements(category="相机", budget_max=100)
    pairs, trace = retriever.retrieve(req)
    assert trace.relaxations, "无匹配类目时应记录放宽原因"
    assert pairs  # 降级后仍有结果
