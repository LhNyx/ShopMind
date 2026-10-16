"""需求合并语义：多轮反馈闭环的确定性核心。"""

from shopmind.schemas import Requirements, compute_missing, merge_requirements


def test_first_turn_returns_new():
    new = Requirements(category="无线耳机", budget_max=500)
    assert merge_requirements(None, new) is new


def test_scalar_override_and_list_union():
    prev = Requirements(category="无线耳机", budget_max=500, attributes=["主动降噪"])
    new = Requirements(budget_max=400, attributes=["长续航"], scene="通勤")
    merged = merge_requirements(prev, new)
    assert merged.category == "无线耳机"          # 新消息没提 → 保留
    assert merged.budget_max == 400               # 新值覆盖
    assert merged.attributes == ["主动降噪", "长续航"]  # 并集保序
    assert merged.scene == "通勤"


def test_exclusions_accumulate_and_dedupe():
    prev = Requirements(exclusions=["小米", "苹果"])
    new = Requirements(exclusions=["小米", "华为"])
    merged = merge_requirements(prev, new)
    assert merged.exclusions == ["小米", "苹果", "华为"]


def test_compute_missing_priority():
    assert compute_missing(Requirements()) == ["budget", "category"]
    assert compute_missing(Requirements(budget_max=500)) == ["category"]
    # 有属性描述即使没类目也放行（带缺口检索好过追问）
    assert compute_missing(Requirements(attributes=["主动降噪"])) == ["budget"]
    assert compute_missing(Requirements(budget_max=500, category="手机")) == []
