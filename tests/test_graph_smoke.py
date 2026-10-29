"""全图冒烟：fake 端口注入，验证两条路径与护栏行为（CI 无 LLM 可跑）。"""

from __future__ import annotations

from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from shopmind.config import Settings
from shopmind.graph.build import build_graph
from shopmind.retrieval.retriever import HybridRetriever
from shopmind.schemas import Requirements


class FakeExtractor:
    def __init__(self, result: Requirements | Exception) -> None:
        self.result = result

    async def extract(self, messages, prev):
        if isinstance(self.result, Exception):
            raise self.result
        return self.result


class FakeQuestioner:
    def __init__(self) -> None:
        self.calls = 0

    async def question(self, missing, messages) -> str:
        self.calls += 1
        return f"请问您的{missing[0]}？"


class FakeReasoner:
    async def generate(self, req, products):
        return [(p.id, f"{p.brand}{p.category}，{p.price:.0f}元符合需求") for p in products]


def make_graph(extractor, store, embedder):
    settings = Settings()
    return build_graph(
        extractor=extractor,
        questioner=FakeQuestioner(),
        reasoner=FakeReasoner(),
        retriever=HybridRetriever(store, embedder, top_n=5),
        settings=settings,
        checkpointer=InMemorySaver(),
    )


async def test_complete_info_goes_direct(store, embedder):
    graph = make_graph(
        FakeExtractor(Requirements(category="无线耳机", budget_max=500, attributes=["主动降噪"])),
        store,
        embedder,
    )
    final = await graph.ainvoke(
        {"messages": [HumanMessage(content="预算500的主动降噪耳机")], "clarify_rounds": 0},
        {"configurable": {"thread_id": "t1"}},
    )
    assert final["clarify_rounds"] == 0
    assert 1 <= len(final["recommendations"]) <= 5
    assert final["recommendations"][0].product.id == "A3"  # 主动降噪+预算内排最前


async def test_missing_info_clarifies_then_guardrail_releases(store, embedder):
    graph = make_graph(FakeExtractor(Requirements()), store, embedder)  # 永远提不出信息
    config = {"configurable": {"thread_id": "t2"}}
    await graph.ainvoke(
        {"messages": [HumanMessage(content="帮我挑个礼物")], "clarify_rounds": 0}, config
    )
    # 第 1 轮追问后 interrupt 挂起，等人回答——不是一轮连问两遍
    snap = await graph.aget_state(config)
    assert snap.next, "缺信息时应停在 await_answer 等用户"
    assert snap.values["clarify_rounds"] == 1

    await graph.ainvoke(Command(resume="随便"), config)
    snap = await graph.aget_state(config)
    assert snap.next, "仍缺信息 → 第 2 轮继续追问"
    assert snap.values["clarify_rounds"] == 2

    await graph.ainvoke(Command(resume="随便"), config)
    final = (await graph.aget_state(config)).values
    # 第 2 轮后护栏放行：即使仍缺信息也出结果
    assert final["clarify_rounds"] == 2
    assert final["recommendations"], "护栏放行后必须出结果（带缺口检索）"
    assert len(final["candidates"]) == 5  # 无约束 → 全目录进入检索，top_n=5 截断


async def test_extraction_failure_falls_back_to_rules(store, embedder):
    graph = make_graph(FakeExtractor(RuntimeError("LLM 挂了")), store, embedder)
    final = await graph.ainvoke(
        {"messages": [HumanMessage(content="预算500买个耳机")], "clarify_rounds": 0},
        {"configurable": {"thread_id": "t3"}},
    )
    # 规则降级应抽出预算与类目 → 不追问，直接检索
    req = final["requirements"]
    assert req.budget_max == 500
    assert req.category == "无线耳机"
    assert final["recommendations"]
