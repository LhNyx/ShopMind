"""CLI 端到端演示：两条场景跑完整图（真 LLM，需 .env）。

场景 1（信息完整 → 直通推荐）  场景 2（信息不足 → 追问 2 轮后护栏放行）
用法：uv run scripts/run_demo.py [自定义 query]
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from shopmind.config import get_settings
from shopmind.graph.build import build_graph
from shopmind.graph.checkpointing import make_serde
from shopmind.llm.clarify import LLMClarifyQuestioner
from shopmind.llm.extractor import LLMRequirementsExtractor
from shopmind.llm.reasons import LLMReasonGenerator
from shopmind.runtime import load_runtime


def show(state: dict) -> None:
    req = state.get("requirements")
    if req:
        filled = {
            k: v
            for k, v in req.model_dump(exclude={"missing_fields"}, exclude_none=True).items()
            if v
        }
        print(f"  需求: {filled}")
        print(f"  缺口: {req.missing_fields or '无'}")
    if state.get("clarify_question") and not state.get("recommendations"):
        print(f"  追问: {state['clarify_question']}")
    for rec in state.get("recommendations", []) or []:
        print(f"  ▸ {rec.product.name}  ¥{rec.product.price:.0f}  [{rec.reason_source}]")
        print(f"    {rec.reason}")


def new_graph():
    settings = get_settings()
    runtime = load_runtime(settings)
    return build_graph(
        extractor=LLMRequirementsExtractor(runtime.chat_model),
        questioner=LLMClarifyQuestioner(runtime.chat_model),
        reasoner=LLMReasonGenerator(runtime.chat_model),
        retriever=runtime.retriever,
        settings=settings,
        checkpointer=InMemorySaver(serde=make_serde()),
    )


def run_turn(graph, thread_id: str, text: str) -> tuple[dict, bool]:
    """跑一轮：上一轮停在 await_answer 就是 resume，否则作为新输入。

    返回 (state, awaiting)——awaiting=True 表示图被 interrupt 挂起，在等用户回答。
    """

    async def _run() -> tuple[dict, bool]:
        config = {"configurable": {"thread_id": thread_id}}
        if (await graph.aget_state(config)).next:
            await graph.ainvoke(Command(resume=text), config)
        else:
            await graph.ainvoke({"messages": [HumanMessage(content=text)]}, config)
        snap = await graph.aget_state(config)
        return snap.values, bool(snap.next)

    return asyncio.run(_run())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("query", nargs="?", default="预算1500左右，买个主动降噪的无线耳机，通勤用")
    args = parser.parse_args()

    graph = new_graph()

    print("═══ 场景 1：信息完整 → 直通 ═══")
    print(f"用户: {args.query}")
    state, _ = run_turn(graph, "demo-direct", args.query)
    show(state)

    print("\n═══ 场景 2：信息不足 → 追问（interrupt 挂起，护栏 ≤2 轮）═══")
    print("用户: 帮我妈挑个礼物")
    tid = "demo-clarify"
    state, awaiting = run_turn(graph, tid, "帮我妈挑个礼物")
    show(state)

    # 最多再回答 2 次；每次回答后图要么再追问（挂起），要么出结果
    for i in range(2):
        if not awaiting:
            break
        answer = input(f"\n你的回答 {i + 1}（直接回车 = 跳过，验证护栏放行）: ").strip()
        if not answer:
            answer = "随便，你看着办"
        print(f"用户: {answer}")
        state, awaiting = run_turn(graph, tid, answer)
        show(state)
    print(f"\n交代：追问 {state.get('clarify_rounds', 0)} 轮后"
          + ("仍在等用户回答" if awaiting else "护栏放行，带缺口出结果"))


if __name__ == "__main__":
    main()
