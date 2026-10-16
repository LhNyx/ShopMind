"""版本化商品快照 + FAISS 本地向量索引的持久化与加载。

快照目录结构（data/catalog/）：
- snapshot.jsonl   商品全集（每行一个 Product JSON）
- embeddings.npy   与 snapshot 行序对齐的 L2 归一化向量矩阵
- meta.json        目录版本（内容 sha256）、SKU 数、embedding 模型等

版本号绑定评测报告：指标变化时先确认目录没变，归因才有意义。
"""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from pydantic import ValidationError

from shopmind.schemas import Product


class CatalogStore:
    def __init__(self, catalog_dir: Path) -> None:
        self._dir = catalog_dir
        self.products: list[Product] = []
        self.embeddings: np.ndarray | None = None  # (N, dim) float32 已归一化
        self.meta: dict = {}
        self._row_by_id: dict[str, int] = {}
        self._faiss_index = None

    # ── 写入（离线管线）────────────────────────────────────
    def save(self, products: list[Product], embeddings: np.ndarray, model_name: str) -> None:
        self._dir.mkdir(parents=True, exist_ok=True)
        snapshot_path = self._dir / "snapshot.jsonl"
        with snapshot_path.open("w", encoding="utf-8") as fh:
            for p in products:
                fh.write(p.model_dump_json() + "\n")
        np.save(self._dir / "embeddings.npy", embeddings.astype(np.float32))
        self.meta = {
            "version": hashlib.sha256(snapshot_path.read_bytes()).hexdigest()[:16],
            "built_at": datetime.now(UTC).isoformat(timespec="seconds"),
            "count": len(products),
            "embedding_model": model_name,
            "dim": int(embeddings.shape[1]),
        }
        (self._dir / "meta.json").write_text(
            json.dumps(self.meta, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        self._set_products(products, embeddings)

    # ── 读取（在线服务）────────────────────────────────────
    def load(self) -> CatalogStore:
        snapshot_path = self._dir / "snapshot.jsonl"
        if not snapshot_path.exists():
            raise FileNotFoundError(
                f"目录快照不存在：{snapshot_path}。先运行 scripts/fetch_catalog.py 生成。"
            )
        products: list[Product] = []
        for line in snapshot_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                products.append(Product.model_validate_json(line))
            except ValidationError:
                continue  # 脏行跳过，快照生成侧已做质量过滤
        embeddings = np.load(self._dir / "embeddings.npy")
        self.meta = json.loads((self._dir / "meta.json").read_text(encoding="utf-8"))
        return self._set_products(products, embeddings)

    @property
    def version(self) -> str:
        return self.meta.get("version", "unknown")

    @property
    def faiss_index(self):
        """IndexFlatIP + L2 归一化向量 = 余弦相似度。2k 规模暴力精确检索最快最准，
        近似索引（IVF/HNSW）是规模上来之后的事。"""
        if self._faiss_index is None:
            import faiss

            index = faiss.IndexFlatIP(self.embeddings.shape[1])
            index.add(self.embeddings)
            self._faiss_index = index
        return self._faiss_index

    def rows_for_ids(self, ids: list[str]) -> np.ndarray:
        """按商品 id 取向量行——子集检索（硬过滤后的候选池）用。"""
        rows = [self._row_by_id[i] for i in ids]
        return self.embeddings[rows]

    # ── 内部 ───────────────────────────────────────────────
    def _set_products(self, products: list[Product], embeddings: np.ndarray) -> CatalogStore:
        self.products = products
        self.embeddings = embeddings
        self._row_by_id = {p.id: i for i, p in enumerate(products)}
        self._faiss_index = None
        return self
