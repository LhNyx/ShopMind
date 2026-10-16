"""FastAPI + SSE 流式推荐服务。

- POST /api/chat/stream：SSE 流式（stage / clarify_question / reason / result / done）
- POST /api/chat：非流式（评测与脚本用）
- SqliteSaver checkpoint：thread_id 即会话 ID，重启不丢、多用户隔离
"""

from __future__ import annotations

import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

import aiosqlite
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from pydantic import BaseModel, Field

from shopmind.config import Settings, get_settings
from shopmind.graph.build import build_graph
from shopmind.graph.checkpointing import make_serde
from shopmind.llm.clarify import LLMClarifyQuestioner
from shopmind.llm.extractor import LLMRequirementsExtractor
from shopmind.llm.reasons import LLMReasonGenerator
from shopmind.runtime import load_runtime


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=2000)
    thread_id: str | None = None  # 会话 ID：多轮反馈与隔离的钥匙


def result_event(state: dict) -> dict:
    """从图终态抽取对外结果。追问轮与推荐轮返回不同形状。"""
    req = state.get("requirements")
    payload: dict[str, Any] = {
        "type": "result",
        "requirements": req.model_dump(exclude={"missing_fields"}) if req else None,
        "clarify_rounds": state.get("clarify_rounds", 0),
    }
    if state.get("clarify_question") and not state.get("recommendations"):
        payload["clarify_question"] = state["clarify_question"]
    if state.get("recommendations"):
        payload["recommendations"] = [
            {
                "id": r.product.id,
                "name": r.product.name,
                "brand": r.product.brand,
                "category": r.product.category,
                "price": r.product.price,
                "score": r.score,
                "reason": r.reason,
                "reason_source": r.reason_source,
            }
            for r in state["recommendations"]
        ]
    if state.get("retrieval_trace"):
        payload["retrieval_trace"] = state["retrieval_trace"].model_dump()
    return payload


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        runtime = load_runtime(settings)
        conn = await aiosqlite.connect(str(settings.data_dir / "checkpoints.sqlite"))
        checkpointer = AsyncSqliteSaver(conn, serde=make_serde())
        graph = build_graph(
            extractor=LLMRequirementsExtractor(runtime.chat_model),
            questioner=LLMClarifyQuestioner(runtime.chat_model),
            reasoner=LLMReasonGenerator(runtime.chat_model),
            retriever=runtime.retriever,
            settings=settings,
            checkpointer=checkpointer,
        )
        app.state.graph = graph
        app.state.settings = settings
        app.state.runtime = runtime
        app.state._conn = conn
        yield
        await conn.close()

    app = FastAPI(title="ShopMind 商品推荐导购 Agent", version="0.1.0", lifespan=lifespan)
    app.add_middleware(
        CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
    )

    @app.get("/api/health")
    async def health() -> dict:
        return {"status": "ok", "catalog_version": app.state.runtime.store.version}

    @app.get("/api/catalog/meta")
    async def catalog_meta() -> dict:
        return app.state.runtime.store.meta

    async def _run_graph(app: FastAPI, message: str, thread_id: str) -> dict:
        inputs = {"messages": [HumanMessage(content=message)]}
        config = {"configurable": {"thread_id": thread_id}}
        final = await app.state.graph.ainvoke(inputs, config)
        return result_event(final)

    @app.post("/api/chat")
    async def chat(body: ChatRequest) -> dict:
        thread_id = body.thread_id or str(uuid.uuid4())
        result = await _run_graph(app, body.message, thread_id)
        result["thread_id"] = thread_id
        return result

    @app.post("/api/chat/stream")
    async def chat_stream(body: ChatRequest) -> StreamingResponse:
        thread_id = body.thread_id or str(uuid.uuid4())

        async def event_stream() -> AsyncIterator[str]:
            def sse(event: dict) -> str:
                import json

                return f"data: {json.dumps(event, ensure_ascii=False)}\n\n"

            yield sse({"type": "start", "thread_id": thread_id})
            inputs = {"messages": [HumanMessage(content=body.message)]}
            config = {"configurable": {"thread_id": thread_id}}
            try:
                # custom：节点内 writer 事件实时透传；updates：取终态组 result
                async for mode, chunk in app.state.graph.astream(
                    inputs, config, stream_mode=["custom", "updates"]
                ):
                    if mode == "custom":
                        yield sse(chunk)
                final = await app.state.graph.aget_state(config)
                result = result_event(final.values)
                result["thread_id"] = thread_id
                yield sse(result)
            except Exception as exc:  # noqa: BLE001 — 错误也以事件形式给客户端
                yield sse({"type": "error", "message": str(exc)})
            yield sse({"type": "done"})

        return StreamingResponse(
            event_stream(), media_type="text/event-stream", headers={"Cache-Control": "no-cache"}
        )

    return app


app = create_app()
