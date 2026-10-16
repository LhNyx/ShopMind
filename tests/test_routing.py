"""路由护栏：追问硬上限是防死循环的关键，单独测。"""

from shopmind.graph.build import route_after_understand


def test_clarify_when_needed_and_under_cap():
    state = {"need_clarify": True, "clarify_rounds": 0}
    assert route_after_understand(state, max_rounds=2) == "clarify"
    state = {"need_clarify": True, "clarify_rounds": 1}
    assert route_after_understand(state, max_rounds=2) == "clarify"


def test_guardrail_forces_retrieve_at_cap():
    # 第 2 轮后强制放行——即使需求仍不完整
    state = {"need_clarify": True, "clarify_rounds": 2}
    assert route_after_understand(state, max_rounds=2) == "retrieve"


def test_no_clarify_when_not_needed():
    assert route_after_understand({"need_clarify": False, "clarify_rounds": 0}, 2) == "retrieve"


def test_removed_guardrail_loops_forever():
    """破坏性实验（课程练习 2 的镜像）：上限改 10，第 3 轮仍会追问——护栏必须存在。"""
    state = {"need_clarify": True, "clarify_rounds": 3}
    assert route_after_understand(state, max_rounds=10) == "clarify"
