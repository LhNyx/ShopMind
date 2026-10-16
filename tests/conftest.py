"""共享 fixture：小型确定性目录 + HashNGram 向量（测试不依赖网络与真模型）。"""

from __future__ import annotations

import pytest

from shopmind.catalog.source import doc_text
from shopmind.catalog.store import CatalogStore
from shopmind.embedding.base import HashNGramEmbedder
from shopmind.retrieval.retriever import HybridRetriever
from shopmind.schemas import Product

CATALOG = [
    Product(
        id="A1", name="索尼 WH-1000 降噪耳机 头戴式 主动降噪",
        brand="索尼", category="无线耳机", price=1899,
        attributes={"降噪": "主动降噪", "佩戴": "头戴式"},
        scene_tags=["通勤", "旅行"], rating=4.8,
    ),
    Product(
        id="A2", name="小米 RedmiBuds 无线耳机 半入耳式",
        brand="小米", category="无线耳机", price=199,
        attributes={"降噪": "无降噪", "佩戴": "半入耳式"},
        scene_tags=["通勤"], rating=4.2,
    ),
    Product(
        id="A3", name="漫步者 NeoBuds 无线耳机 主动降噪 入耳式 24小时续航",
        brand="漫步者", category="无线耳机", price=399,
        attributes={"降噪": "主动降噪", "续航": "24小时续航"},
        scene_tags=["通勤", "运动"], rating=4.5,
    ),
    Product(
        id="B1", name="华为 MatePad 平板 11英寸 128G",
        brand="华为", category="平板", price=2599,
        attributes={"屏幕": "11英寸", "存储": "128G"},
        scene_tags=["学习", "网课"], rating=4.6,
    ),
    Product(
        id="B2", name="苹果 iPad 平板 256G 手写笔支持",
        brand="苹果", category="平板", price=4999,
        attributes={"存储": "256G", "特色": "手写笔支持"},
        scene_tags=["绘画", "学习"], rating=4.9,
    ),
    Product(
        id="C1", name="小米 Watch 智能手表 心率监测 14天续航",
        brand="小米", category="智能手表", price=899,
        attributes={"健康": "心率监测", "续航": "14天续航"},
        scene_tags=["运动", "健康"], rating=4.4,
    ),
]


@pytest.fixture
def embedder() -> HashNGramEmbedder:
    return HashNGramEmbedder(dim=64)


@pytest.fixture
def store(tmp_path, embedder) -> CatalogStore:
    store = CatalogStore(tmp_path / "catalog")
    embeddings = embedder.embed_documents([doc_text(p) for p in CATALOG])
    store.save(CATALOG, embeddings, embedder.name)
    return store.load()


@pytest.fixture
def retriever(store, embedder) -> HybridRetriever:
    return HybridRetriever(store, embedder, top_n=5)
