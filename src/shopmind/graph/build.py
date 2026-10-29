"""组图：understand → (条件边) clarify → await_answer(挂起) → retrieve → rank_and_explain。

防死循环护栏（面试考点）：need_clarify 且追问轮数 < 硬上限才走 clarify；
到上限后带着缺口强制放行去检索——带着缺口出结果，好过无限追问。
await_answer 用 interrupt() 真正挂起等人，checkpoint 恢复后继续，不会一轮连问两遍。
"""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from shopmind.config import Settings
from shopmind.graph.nodes import (
    make_await_answer_node,
    make_clarify_node,
    make_rank_and_explain_node,
    make_retrieve_node,
    make_understand_node,
)
from shopmind.graph.ports import ClarifyQuestioner, ReasonGenerator, RequirementsExtractor
from shopmind.graph.state import RecommendationState
from shopmind.retrieval.retriever import HybridRetriever


def route_after_understand(state: RecommendationState, max_rounds: int) -> str:
    """条件边路由：需要追问且未到上限 → clarify，否则 → retrieve。"""
    if state.get("need_clarify") and state.get("clarify_rounds", 0) < max_rounds:
        return "clarify"
    return "retrieve"


def build_graph(
    *,
    extractor: RequirementsExtractor,
    questioner: ClarifyQuestioner,
    reasoner: ReasonGenerator,
    retriever: HybridRetriever,
    settings: Settings,
    checkpointer=None,
):
    """依赖注入式组图：LLM 实现与测试 fake 通过参数替换，图结构不变。"""
    builder = StateGraph(RecommendationState)
    builder.add_node("understand", make_understand_node(extractor))
    builder.add_node("clarify", make_clarify_node(questioner))
    builder.add_node("await_answer", make_await_answer_node())
    builder.add_node("retrieve", make_retrieve_node(retriever))
    builder.add_node("rank_and_explain", make_rank_and_explain_node(reasoner, settings))

    builder.add_edge(START, "understand")
    builder.add_conditional_edges(
        "understand",
        lambda state: route_after_understand(state, settings.clarify_max_rounds),
        {"clarify": "clarify", "retrieve": "retrieve"},
    )
    builder.add_edge("clarify", "await_answer")
    builder.add_edge("await_answer", "understand")  # 用户回答后重新理解（循环的来源）
    builder.add_edge("retrieve", "rank_and_explain")
    builder.add_edge("rank_and_explain", END)

    return builder.compile(checkpointer=checkpointer)
