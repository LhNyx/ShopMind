"""结构化需求提取的 LLM 实现：Pydantic Schema 约束 → 重试 → JSON 兜底。

三级防线（对应简历「解析失败自动重试降级」）：
1. with_structured_output(Requirements) —— function calling 约束
2. 失败 → 提示词要求纯 JSON 输出，手工剥离代码围栏后 model_validate
3. 再失败 → 抛 ExtractionError，节点层降级到规则提取器（graph/rules.py）
"""

from __future__ import annotations

import json
import re

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AnyMessage, HumanMessage, SystemMessage

from shopmind.schemas import Requirements

_EXTRACTION_SYSTEM = """你是购物助手的需求理解模块。从对话中提取结构化购物需求。

规则：
- 只提取用户明确表达的信息，不猜测、不编造
- 结合「当前已确认的需求」与「用户最新消息」输出合并后的完整需求：
  相对调整要换算成具体数字（如已确认预算上限 500，用户说"再便宜点"，输出 400 左右）
- 用户说「不要X / 排除X」时把 X 放入 exclusions
- category 取这些值之一：无线耳机/手机/笔记本/智能手表/平板/蓝牙音箱；不确定则不填
- 不要输出 missing_fields，它由系统计算
"""

_JSON_RETRY_SUFFIX = "\n\n请只输出符合 schema 的 JSON 对象，不要任何其他文字或代码围栏。"

_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


class ExtractionError(Exception):
    """三级提取全部失败，由节点层降级规则提取。"""


class LLMRequirementsExtractor:
    def __init__(self, chat_model: BaseChatModel) -> None:
        self._model = chat_model

    async def extract(self, messages: list[AnyMessage], prev: Requirements | None) -> Requirements:
        prompt = self._build_messages(messages, prev, retry_suffix="")

        # 第一级：结构化输出（function calling 约束，替代自由文本拼接）
        try:
            result = await self._model.with_structured_output(Requirements).ainvoke(prompt)
            return _normalize(result)
        except Exception:  # noqa: BLE001 — 兼容端点不支持 function calling 等情况
            pass

        # 第二级：纯 JSON 输出 + 手工解析
        try:
            retry_prompt = self._build_messages(messages, prev, _JSON_RETRY_SUFFIX)
            raw = await self._model.ainvoke(retry_prompt)
            content = raw.content if isinstance(raw.content, str) else str(raw.content)
            return _parse_json_requirements(content)
        except Exception as exc:  # noqa: BLE001
            raise ExtractionError(str(exc)) from exc

    def _build_messages(
        self, messages: list[AnyMessage], prev: Requirements | None, retry_suffix: str
    ) -> list[AnyMessage]:
        prev_block = (
            "当前已确认的需求："
            + json.dumps(
                prev.model_dump(exclude={"missing_fields"}, exclude_none=True),
                ensure_ascii=False,
            )
            if prev
            else "当前已确认的需求：无（首轮）"
        )
        return [
            SystemMessage(content=_EXTRACTION_SYSTEM + prev_block + retry_suffix),
            *messages[-6:],  # 只带最近 6 条，控制 token 成本
        ]


def _parse_json_requirements(text: str) -> Requirements:
    m = _FENCE_RE.search(text)
    payload = m.group(1) if m else text
    start, end = payload.find("{"), payload.rfind("}")
    if start == -1 or end == -1:
        raise ValueError(f"输出中没有 JSON：{text[:80]}")
    return Requirements.model_validate(json.loads(payload[start : end + 1]))


def _normalize(req: Requirements) -> Requirements:
    """防御性清洗：空串归 None、列表去空白。"""
    req.category = (req.category or None) and req.category.strip() or None
    req.scene = (req.scene or None) and req.scene.strip() or None
    req.recipient = (req.recipient or None) and req.recipient.strip() or None
    req.attributes = [a.strip() for a in req.attributes if a and a.strip()]
    req.exclusions = [e.strip() for e in req.exclusions if e and e.strip()]
    if req.budget_min is not None and req.budget_min <= 0:
        req.budget_min = None
    if req.budget_max is not None and req.budget_max <= 0:
        req.budget_max = None
    req.missing_fields = []
    return req


def latest_user_text(messages: list[AnyMessage]) -> str:
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            return str(msg.content)
    return str(messages[-1].content) if messages else ""
