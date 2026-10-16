"""生成商品快照并构建向量索引。

用法：
  uv run scripts/fetch_catalog.py --source mock            # seeded mock 目录（1200 SKU）
  uv run scripts/fetch_catalog.py --source jd              # 京东联盟快照（需 .env 配置密钥）
  uv run scripts/fetch_catalog.py --source jd --per-keyword 60 --keywords 耳机 手机
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from shopmind.catalog.jd_source import JdUnionSource
from shopmind.catalog.mock_source import SOURCE_NAME as MOCK_NAME
from shopmind.catalog.mock_source import generate_catalog
from shopmind.catalog.source import doc_text
from shopmind.catalog.store import CatalogStore
from shopmind.config import get_settings
from shopmind.embedding.local_bge import LocalBgeEmbedder

# 3C 垂类抓取关键词（扩品类 = 改这里，架构不动）
DEFAULT_KEYWORDS = ["无线耳机", "手机", "笔记本", "智能手表", "平板", "蓝牙音箱"]


def build_mock_snapshot(settings, embedder) -> None:
    products = generate_catalog()
    print(f"[mock] 生成 {len(products)} 个 SKU（seed=42，可复现）")
    embeddings = embedder.embed_documents([doc_text(p) for p in products])
    store = CatalogStore(settings.catalog_dir)
    store.save(products, embeddings, embedder.name)
    print(f"[mock] 快照落盘 {settings.catalog_dir}，版本 {store.version}")


def build_jd_snapshot(settings, embedder, keywords: list[str], per_keyword: int) -> None:
    source = JdUnionSource(
        settings.jd_union_app_key, settings.jd_union_secret_key, settings.jd_union_gateway
    )
    products: list[Product] = []
    for keyword in keywords:
        batch = source.fetch(keyword, per_keyword)
        print(f"[jd] 「{keyword}」拉取 {len(batch)} 个 SKU")
        products.extend(batch)
    if len({p.id for p in products}) != len(products):
        seen, deduped = set(), []
        for p in products:
            if p.id not in seen:
                seen.add(p.id)
                deduped.append(p)
        products = deduped
    print(f"[jd] 共 {len(products)} SKU，开始向量化…")
    embeddings = embedder.embed_documents([doc_text(p) for p in products])
    store = CatalogStore(settings.catalog_dir)
    store.save(products, embeddings, embedder.name)
    print(f"[jd] 快照落盘，版本 {store.version}")


def main() -> None:
    parser = argparse.ArgumentParser(description="生成商品快照 + FAISS 索引")
    parser.add_argument("--source", choices=["mock", "jd"], default="mock")
    parser.add_argument("--keywords", nargs="*", default=DEFAULT_KEYWORDS)
    parser.add_argument("--per-keyword", type=int, default=60, help="jd 模式每个关键词拉取条数")
    args = parser.parse_args()

    settings = get_settings()
    model_dir = settings.models_dir / settings.embedding_model
    embedder = LocalBgeEmbedder(
        model_dir,
        hf_endpoint=settings.hf_endpoint,
        dim=settings.embedding_dim,
        max_len=settings.embedding_max_len,
        ort_threads=settings.ort_threads,
    )

    if args.source == MOCK_NAME:
        build_mock_snapshot(settings, embedder)
    else:
        build_jd_snapshot(settings, embedder, args.keywords, args.per_keyword)

    # 建索引并冒烟验证
    store = CatalogStore(settings.catalog_dir).load()
    _ = store.faiss_index
    print(f"[done] 目录版本 {store.version}，{len(store.products)} SKU，索引就绪")


if __name__ == "__main__":
    from shopmind.schemas import Product  # noqa: F401  类型引用

    main()
