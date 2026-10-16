"""跑评测并输出报告。

用法：
  uv run scripts/run_eval.py                          # retrieval 模式（无 LLM，CI 可跑）
  uv run scripts/run_eval.py --tag v2                 # 报告命名，供 eval_compare 对比
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from shopmind.catalog.store import CatalogStore
from shopmind.config import get_settings
from shopmind.embedding.local_bge import LocalBgeEmbedder
from shopmind.eval.runner import load_dataset, run_retrieval_eval
from shopmind.retrieval.retriever import HybridRetriever


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", default="base", help="报告标签，用于回归对比")
    args = parser.parse_args()

    settings = get_settings()
    dataset = load_dataset(settings.data_dir / "eval" / "queries.jsonl")
    store = CatalogStore(settings.catalog_dir).load()
    embedder = LocalBgeEmbedder(
        settings.models_dir / settings.embedding_model,
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

    print(f"[eval] 目录版本 {store.version}，{len(store.products)} SKU，{len(dataset)} 条 query")
    report = run_retrieval_eval(dataset, retriever, store)

    settings.reports_dir.mkdir(parents=True, exist_ok=True)
    out = settings.reports_dir / f"metrics_{args.tag}.json"
    out.write_text(json.dumps(report.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n═══ 评测结果 ═══")
    for key, value in report.metrics.items():
        print(f"  {key:<20} {value}")
    print(f"\n坏例 {len(report.badcases)} 条，报告 → {out}")


if __name__ == "__main__":
    main()
