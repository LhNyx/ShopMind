"""追问生成的 LLM 实现：针对一个缺口问一句话，失败降级固定模板。

追问是用户可见文本，走流式；模板兜底保证澄清链路永不空转。
"""

from __future__ import annotations

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AnyMessage, HumanMessage, SystemMessage

_FALLBACK_QUESTIONS = {
    "budget": "请问您的预算大概在什么范围呢？",
    "category": "您想看哪个品类？比如无线耳机、手机、笔记本、智能手表都可以～",
}

_LABEL = {"budget": "预算区间（元）", "category": "商品品类"}


class LLMClarifyQuestioner:
    def __init__(self, chat_model: BaseChatModel) -> None:
        self._model = chat_model

    async def question(self, missing: list[str], messages: list[AnyMessage]) -> str:
        if not missing:
            return ""
        field = missing[0]
        system = (
            "你是购物助手。用户的需求缺少信息，请生成一句自然的追问，要求：\n"
            f"- 只追问缺失字段：{_LABEL.get(field, field)}\n"
            "- 一句话，不超过 25 字，语气友好不啰嗦\n"
            "- 直接输出问题本身，不要引号和前缀"
        )
        try:
            content = ""
            prompt = [
                SystemMessage(content=system),
                *messages[-4:],
                HumanMessage(content="请生成追问。"),
            ]
            async for chunk in self._model.astream(prompt):
                piece = chunk.content if isinstance(chunk.content, str) else ""
                content += piece
            content = content.strip().strip('"「」')
            return content or _FALLBACK_QUESTIONS.get(field, "能再描述一下您的需求吗？")
        except Exception:  # noqa: BLE001 — 追问失败不影响主链路
            return _FALLBACK_QUESTIONS.get(field, "能再描述一下您的需求吗？")
