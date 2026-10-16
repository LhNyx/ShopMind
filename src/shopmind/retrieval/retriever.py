"""混合检索：结构化硬过滤在前缩小候选集 → FAISS 子集内语义召回。

「先过滤后召回」的理由（简历 bullet 的灵魂三问）：
1. 预算外的相似商品不会污染 top 结果（语义检索不知道价格）
2. 候选集从全目录缩到几十几百，向量计算量随之下降
3. 评测上两者解耦：Recall 掉了能归因是过滤错杀还是召回不准（看 Trace）
"""

from __future__ import annotations

import faiss

from shopmind.catalog.source import query_text_from_requirements
from shopmind.catalog.store import CatalogStore
from shopmind.embedding.base import BaseEmbedder
from shopmind.retrieval.filters import FilterResult, hard_filter
from shopmind.schemas import Product, Requirements, RetrievalTrace


class HybridRetriever:
    def __init__(
        self,
        store: CatalogStore,
        embedder: BaseEmbedder,
        *,
        budget_tolerance: float = 0.20,
        relax_tolerance: float = 0.40,
        top_n: int = 20,
        fetch_depth: int = 400,
        attr_boost: float = 0.5,
        scene_boost: float = 0.2,
    ) -> None:
        self._store = store
        self._embedder = embedder
        self._budget_tol = budget_tolerance
        self._relax_tol = relax_tolerance
        self._top_n = top_n
        self._fetch_depth = fetch_depth
        self._attr_boost = attr_boost
        self._scene_boost = scene_boost

    def retrieve(self, req: Requirements) -> tuple[list[tuple[Product, float]], RetrievalTrace]:
        """返回 [(商品, 余弦相似度)]（按综合分降序）与检索留痕。

        粗排 → 精排两段式（电商检索标准形态）：
        FAISS 余弦粗排取 fetch_depth 个，再用属性/场景命中做确定性精排——
        纯语义排序对「有没有某个属性」不够稳（模板化文本相似度挤在一起），
        属性命中是结构化信号，归排序不该归向量。
        """
        filtered: FilterResult = hard_filter(
            self._store.products, req, self._budget_tol, self._relax_tol
        )
        trace = RetrievalTrace(
            total_catalog=len(self._store.products),
            after_filter=len(filtered.products),
            relaxations=filtered.relaxations,
            recalled=0,
        )
        if not filtered.products:
            return [], trace

        # 硬过滤后的子集上做 FAISS 检索：每次按过滤结果取行建临时 IndexFlatIP，
        # 2k 规模毫秒级，且天然避免「全库检索后再丢弃越界结果」的浪费。
        query_vec = self._embedder.embed_query(
            query_text_from_requirements(req.category, req.attributes, req.scene, req.recipient)
        ).reshape(1, -1)
        subset = self._store.rows_for_ids([p.id for p in filtered.products])
        index = faiss.IndexFlatIP(subset.shape[1])
        index.add(subset)
        k = min(self._fetch_depth, subset.shape[0])
        sims, positions = index.search(query_vec, k)

        rescored: list[tuple[Product, float, float]] = []
        for pos, sim in zip(positions[0], sims[0], strict=True):
            if pos < 0:  # faiss 不足 k 时用 -1 填充
                continue
            product = filtered.products[int(pos)]
            score = float(sim) + self._attr_boost * _attr_overlap(req, product)
            score += self._scene_boost * _scene_overlap(req, product)
            rescored.append((product, float(sim), score))
        rescored.sort(key=lambda item: item[2], reverse=True)
        results = [(p, sim) for p, sim, _ in rescored[: self._top_n]]
        trace.recalled = len(results)
        return results, trace


def _attr_overlap(req: Requirements, product: Product) -> float:
    """需求属性在商品属性值中的命中率（确定性特征）。"""
    if not req.attributes:
        return 0.0
    values = list(product.attributes.values())
    return sum(1 for a in req.attributes if a in values) / len(req.attributes)


def _scene_overlap(req: Requirements, product: Product) -> float:
    if not req.scene:
        return 0.0
    return 1.0 if req.scene in product.scene_tags else 0.0
