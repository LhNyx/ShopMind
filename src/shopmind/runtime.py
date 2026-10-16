"""运行时装配：配置 → 快照 → embedding → 检索 → LLM 端口 → 图。

服务、CLI、评测共用同一装配路径，保证「评测的对象就是部署的对象」。
"""

from __future__ import annotations

from dataclasses import dataclass

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_openai import ChatOpenAI

from shopmind.catalog.store import CatalogStore
from shopmind.config import Settings
from shopmind.embedding.local_bge import LocalBgeEmbedder
from shopmind.retrieval.retriever import HybridRetriever


@dataclass
class Runtime:
    settings: Settings
    store: CatalogStore
    retriever: HybridRetriever
    chat_model: BaseChatModel


def build_chat_model(settings: Settings) -> ChatOpenAI:
    if not settings.openai_api_key:
        raise RuntimeError(
            "未配置 OPENAI_API_KEY：请复制 .env.example 为 .env 并填写（或用 mock 模式跑离线评测）"
        )
    return ChatOpenAI(
        model=settings.openai_model,
        api_key=settings.openai_api_key,
        base_url=settings.openai_base_url or None,
        temperature=0.2,
        streaming=True,
    )


def load_runtime(settings: Settings, *, embedder=None) -> Runtime:
    """加载快照与检索器。embedder 可注入（评测/测试用 HashNGram 替代真模型）。"""
    store = CatalogStore(settings.catalog_dir).load()
    if embedder is None:
        model_dir = settings.models_dir / settings.embedding_model
        embedder = LocalBgeEmbedder(
            model_dir,
            hf_endpoint=settings.hf_endpoint,
            dim=settings.embedding_dim,
            max_len=settings.embedding_max_len,
            ort_threads=settings.ort_threads,
        )
    retriever = HybridRetriever(
        store,
        embedder,
        budget_tolerance=settings.budget_tolerance,
        relax_tolerance=settings.relax_tolerance,
        top_n=settings.retrieve_top_n,
        fetch_depth=settings.fetch_depth,
        attr_boost=settings.rerank_attr_boost,
        scene_boost=settings.rerank_scene_boost,
    )
    return Runtime(
        settings=settings,
        store=store,
        retriever=retriever,
        chat_model=build_chat_model(settings),
    )
