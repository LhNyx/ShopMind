"""LangGraph State：全链路共享的一等公民。

节点只返回增量 dict、绝不原地改 State——checkpoint 断点续跑
和多用户会话隔离都建立在增量合并之上。
"""

from __future__ import annotations

from typing import Annotated, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages

from shopmind.schemas import Product, Recommendation, Requirements, RetrievalTrace


class RecommendationState(TypedDict, total=False):
    messages: Annotated[list[AnyMessage], add_messages]
    requirements: Requirements | None
    need_clarify: bool
    clarify_rounds: int           # 已追问轮数，硬上限见 Settings.clarify_max_rounds
    clarify_question: str | None
    candidates: list[Product]
    similarities: list[float]     # 与 candidates 对齐的语义相似度（排序权重之一）
    recommendations: list[Recommendation]
    retrieval_trace: RetrievalTrace | None
