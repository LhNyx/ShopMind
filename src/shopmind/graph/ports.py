"""图层的依赖端口：节点只依赖这三个小接口，LLM 实现与测试 fake 同权。

依赖注入的价值（面试点）：CI / 离线评测不需要真模型——
评测框架用 golden requirements 走检索栈，单测用确定性 fake 走全图。
"""

from __future__ import annotations

from typing import Protocol

from langchain_core.messages import AnyMessage

from shopmind.schemas import Product, Requirements


class RequirementsExtractor(Protocol):
    """从消息流提取结构化需求；解析失败由实现方自行重试降级。"""

    async def extract(
        self, messages: list[AnyMessage], prev: Requirements | None
    ) -> Requirements: ...


class ClarifyQuestioner(Protocol):
    """针对最关键的信息缺口生成一句追问。"""

    async def question(self, missing: list[str], messages: list[AnyMessage]) -> str: ...


class ReasonGenerator(Protocol):
    """为 top-K 商品逐条生成推荐理由，返回 (product_id, reason)，
    只允许引用商品字段——幻觉抑制在生成侧的约束。"""

    async def generate(
        self, req: Requirements, products: list[Product]
    ) -> list[tuple[str, str]]: ...
