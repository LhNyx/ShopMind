"""五个图节点：understand / clarify / await_answer / retrieve / rank_and_explain。

节点是纯函数：读 State → 调依赖 → 返回增量 dict。
流式事件通过 langgraph 的 custom stream writer 发给 SSE 层，
节点本身对传输层无感知（writer 不存在时静默降级为离线模式）。
"""

from __future__ import annotations

import re

from langchain_core.messages import AIMessage, HumanMessage

from shopmind.config import Settings
from shopmind.graph.ports import ClarifyQuestioner, ReasonGenerator, RequirementsExtractor
from shopmind.graph.rules import extract_requirements_by_rules
from shopmind.retrieval.retriever import HybridRetriever
from shopmind.schemas import (
    Recommendation,
    compute_missing,
    merge_requirements,
)

try:  # langgraph 1.x；不存在时为离线/测试模式，静默降级
    from langgraph.config import get_stream_writer
except ImportError:  # pragma: no cover
    get_stream_writer = None

from langgraph.types import interrupt

_CLARIFY_LABEL = {"budget": "预算", "category": "品类"}


def _emit(event: dict) -> None:
    if get_stream_writer is None:
        return
    try:
        get_stream_writer()(event)
    except Exception:  # noqa: BLE001 — 非 astream 上下文没有 writer
        pass


# ── understand：结构化提取 + 合并 + 确定性缺口判定 ──────────────────────────


def make_understand_node(extractor: RequirementsExtractor):
    async def understand(state: dict) -> dict:
        _emit({"type": "stage", "stage": "understand", "message": "正在理解需求…"})
        prev = state.get("requirements")
        messages = state["messages"]
        try:
            extracted = await extractor.extract(messages, prev)
        except Exception as exc:  # noqa: BLE001 — 结构化输出两连败后降级
            _emit(
                {
                    "type": "stage",
                    "stage": "understand",
                    "message": f"结构化解析失败，降级规则提取（{exc}）",
                }
            )
            latest = messages[-1].content if messages else ""
            extracted = extract_requirements_by_rules(str(latest))

        merged = merge_requirements(prev, extracted)
        merged.missing_fields = compute_missing(merged)
        return {
            "requirements": merged,
            "need_clarify": bool(merged.missing_fields),
        }

    return understand


# ── clarify：针对最关键缺口生成一句追问 ─────────────────────────────────────


def make_clarify_node(questioner: ClarifyQuestioner):
    async def clarify(state: dict) -> dict:
        rounds = state.get("clarify_rounds", 0) + 1
        missing = state["requirements"].missing_fields
        label = _CLARIFY_LABEL.get(missing[0], missing[0]) if missing else "需求"
        _emit({"type": "stage", "stage": "clarify", "message": f"需要补充{label}信息"})
        question = await questioner.question(missing, state["messages"])
        _emit({"type": "clarify_question", "round": rounds, "text": question})
        return {
            "clarify_rounds": rounds,
            "clarify_question": question,
            "messages": [AIMessage(content=question)],
        }

    return clarify


# ── await_answer：interrupt 挂起，等用户回答后由 checkpoint 恢复 ────────────


def make_await_answer_node():
    """把「追问后等回答」做成独立节点：问题生成只跑一次，恢复时不重复调 LLM。"""

    async def await_answer(state: dict) -> dict:
        _emit(
            {
                "type": "awaiting_answer",
                "round": state.get("clarify_rounds", 0),
                "text": state.get("clarify_question"),
            }
        )
        answer = interrupt(
            {"question": state.get("clarify_question"), "round": state.get("clarify_rounds", 0)}
        )
        return {
            "messages": [HumanMessage(content=str(answer))],
            "clarify_question": None,
        }

    return await_answer


# ── retrieve：硬过滤 + 语义召回 ─────────────────────────────────────────────


def make_retrieve_node(retriever: HybridRetriever):
    async def retrieve(state: dict) -> dict:
        _emit({"type": "stage", "stage": "retrieve", "message": "正在筛选商品…"})
        pairs, trace = retriever.retrieve(state["requirements"])
        _emit(
            {
                "type": "stage",
                "stage": "retrieve",
                "message": (
                    f"候选 {trace.total_catalog} → 过滤后 {trace.after_filter}"
                    + (f"（{'；'.join(trace.relaxations)}）" if trace.relaxations else "")
                    + f" → 召回 {trace.recalled}"
                ),
            }
        )
        return {
            "candidates": [p for p, _ in pairs],
            "similarities": [sim for _, sim in pairs],
            "retrieval_trace": trace,
        }

    return retrieve


# ── rank_and_explain：确定性打分 + 可溯源理由 ───────────────────────────────


def _budget_fit(price: float, req_budget_min: float | None, req_budget_max: float | None) -> float:
    """预算贴合度：越靠近预算中心得分越高，区间外单调衰减。"""
    if req_budget_max is None:
        return 0.5
    low = req_budget_min or 0.0
    center = (low + req_budget_max) / 2
    span = max(req_budget_max - low, 1.0)
    return max(0.0, 1.0 - abs(price - center) / span)


def make_rank_and_explain_node(
    reasoner: ReasonGenerator,
    settings: Settings,
):
    async def rank_and_explain(state: dict) -> dict:
        _emit({"type": "stage", "stage": "rank_and_explain", "message": "正在排序并生成推荐理由…"})
        req = state["requirements"]
        candidates = state["candidates"]
        sims = state.get("similarities") or [0.0] * len(candidates)

        # 确定性打分排序（LLM 不参与排序——排序必须可解释、可回归）
        semantic = {p.id: sim for p, sim in zip(candidates, sims, strict=False)}

        def total_score(p) -> float:
            s = settings.score_weight_semantic * semantic.get(p.id, 0.0)
            s += settings.score_weight_budget * _budget_fit(
                p.price, req.budget_min, req.budget_max
            )
            if p.rating is not None:
                s += settings.score_weight_rating * (p.rating - 3.0) / 2.0
            return s

        ranked = sorted(candidates, key=total_score, reverse=True)[: settings.recommend_top_k]

        # LLM 逐条理由（只允许引用目录字段），校验失败降级模板
        reasons = dict(await reasoner.generate(req, ranked))
        recommendations = []
        for p in ranked:
            reason = reasons.get(p.id) or _template_reason(p, req)
            source = "llm" if p.id in reasons and _reason_consistent(reason, p) else "template"
            if source == "template":
                reason = _template_reason(p, req)
            recommendations.append(
                Recommendation(
                    product=p,
                    score=round(total_score(p), 4),
                    reason=reason,
                    reason_source=source,
                )
            )
            if p.id not in reasons:  # LLM 流式阶段已发过 reason 事件，这里只补发模板降级
                _emit(
                    {
                        "type": "reason",
                        "product_id": p.id,
                        "name": p.name,
                        "price": p.price,
                        "reason": reason,
                        "source": source,
                    }
                )

        summary = f"为你找到 {len(recommendations)} 件：\n" + "\n".join(
            f"{i}. {r.product.name}（¥{r.product.price:.0f}）—— {r.reason}"
            for i, r in enumerate(recommendations, 1)
        )
        return {"recommendations": recommendations, "messages": [AIMessage(content=summary)]}

    return rank_and_explain


def _reason_consistent(reason: str, product) -> bool:
    """理由一致性校验：理由里出现的「价格样数字」必须与目录一致（防幻觉的落地手段）。"""
    numbers = [float(n) for n in re.findall(r"(?:¥|￥|价格\s*)?(\d+(?:\.\d+)?)\s*元", reason)]
    return all(abs(n - product.price) < 0.51 for n in numbers)


def _template_reason(p, req) -> str:
    attrs = "、".join(list(p.attributes.values())[:2])
    scenes = [s for s in p.scene_tags if req.scene and s == req.scene]
    head = f"贴合「{scenes[0]}」场景" if scenes else f"{p.brand} {p.category}"
    detail = f"，{attrs}" if attrs else ""
    return f"{head}{detail}，¥{p.price:.0f} 在你的预算范围内。"


__all__ = [
    "make_await_answer_node",
    "make_clarify_node",
    "make_rank_and_explain_node",
    "make_retrieve_node",
    "make_understand_node",
]
