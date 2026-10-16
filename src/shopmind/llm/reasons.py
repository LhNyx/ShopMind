"""推荐理由生成的 LLM 实现：逐行协议 + 流式产出 + 字段约束。

幻觉抑制的落地手段（简历 bullet）：
- 提示词强约束「只能引用给定商品字段，不得出现字段之外的价格/参数」
- 行协议 "id|理由" 逐行产出，节点层做价格一致性校验，不过则降级模板理由
"""

from __future__ import annotations

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from shopmind.graph.nodes import _emit
from shopmind.schemas import Product, Requirements

_REASON_SYSTEM = """你是购物助手的推荐理由模块。为给定商品各写一句推荐理由。

硬性要求：
- 只能引用「商品清单」中给出的字段（名称/品牌/类目/价格/属性/场景），不得编造任何字段之外的信息
- 提到价格时只能写清单里的价格数字，格式如「329元」
- 结合用户需求说清「为什么匹配」，每句不超过 30 字
- 每行一条，严格格式：商品ID|理由
- 不要输出其他任何内容
"""


class LLMReasonGenerator:
    def __init__(self, chat_model: BaseChatModel) -> None:
        self._model = chat_model

    async def generate(
        self, req: Requirements, products: list[Product]
    ) -> list[tuple[str, str]]:
        catalog_block = "\n".join(
            f"- id={p.id} | {p.name} | 品牌:{p.brand} | 类目:{p.category} | "
            f"价格:{p.price:.0f}元 | 属性:{'、'.join(p.attributes.values()) or '无'} | "
            f"场景:{'、'.join(p.scene_tags) or '无'}"
            for p in products
        )
        req_block = req.model_dump(exclude={"missing_fields"}, exclude_none=True)
        prompt = [
            SystemMessage(content=_REASON_SYSTEM),
            HumanMessage(
                content=f"用户需求：{req_block}\n\n商品清单：\n{catalog_block}\n\n请逐行输出理由。"
            ),
        ]

        results: list[tuple[str, str]] = []
        buffer = ""
        try:
            async for chunk in self._model.astream(prompt):
                delta = chunk.content if isinstance(chunk.content, str) else ""
                if not delta:
                    continue
                buffer += delta
                while "\n" in buffer:
                    line, buffer = buffer.split("\n", 1)
                    parsed = _parse_line(line)
                    if parsed:
                        results.append(parsed)
                        _emit(
                            {
                                "type": "reason",
                                "product_id": parsed[0],
                                "reason": parsed[1],
                                "source": "llm",
                            }
                        )
            if buffer.strip():
                parsed = _parse_line(buffer)
                if parsed:
                    results.append(parsed)
        except Exception:  # noqa: BLE001 — 理由生成失败时由节点层模板兜底
            return results
        return results


def _parse_line(line: str) -> tuple[str, str] | None:
    line = line.strip().lstrip("-• ")
    if "|" not in line:
        return None
    pid, _, reason = line.partition("|")
    pid, reason = pid.strip(), reason.strip()
    if pid and reason:
        return pid, reason
    return None
