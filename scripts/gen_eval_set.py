"""构造评测集：约束反推 + 全集相关标注，输出 data/eval/queries.jsonl。

标注口径（README/面试要讲清楚）：
- query 由「结构化约束 + 模板」反向构造：先抽类目/预算/属性/场景，再生成自然语言
- **expected_ids = 满足 query 全部明示约束的商品全集**（类目匹配、价格在预算容差内、
  属性值全含、场景含标签），不是少数锚点——锚点标注在同类商品成百上千时没有区分信号
- 预算容差与线上硬过滤一致（±20%），保证「按设计算分」而非「按理想算分」
- should_clarify 由 compute_missing 确定性判定，标注与实现同源但口径独立于 LLM

用法：uv run scripts/gen_eval_set.py [--size 120]
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from shopmind.catalog.mock_source import generate_catalog
from shopmind.config import get_settings
from shopmind.retrieval.filters import hard_filter
from shopmind.schemas import Requirements, compute_missing

_TEMPLATES = [
    "预算{budget}元以内，想买{category}，要求{attrs}，用来{scene}",
    "帮我挑个{category}，{attrs}，价格控制在{budget}块以内",
    "想要{attrs}的{category}，预算{budget}左右，主要{scene}用",
    "想入手{category}，要{attrs}，{scene}场景，预算不超过{budget}",
    "{category}求推荐，必须{attrs}，预算上限{budget}元",
]
_SHORT_TEMPLATES = [
    "预算{budget}以内买个{category}",
    "{category}，{budget}元以内",
    "帮我看下{category}，预算{budget}",
]


def _relevant_set(catalog, req: Requirements, attrs: list[str], scene: str | None) -> list:
    """相关集合 = 满足全部明示约束的商品（预算容差与线上过滤一致）。"""
    filtered = hard_filter(catalog, req).products
    out = []
    for p in filtered:
        values = list(p.attributes.values())
        if not all(a in values for a in attrs):
            continue
        if scene and scene not in p.scene_tags:
            continue
        out.append(p)
    return out


def build_dataset(seed: int = 7, size: int = 120) -> list[dict]:
    rng = random.Random(seed)
    catalog = generate_catalog()
    by_category: dict[str, list] = {}
    for p in catalog:
        by_category.setdefault(p.category, []).append(p)

    dataset: list[dict] = []
    idx = 0
    n_pos = size * 3 // 4  # 其余为追问场景
    attempts = 0
    while len(dataset) < n_pos and attempts < size * 50:
        attempts += 1
        category = rng.choice(list(by_category))
        pool = by_category[category]
        anchors = rng.sample(pool, k=min(3, len(pool)))
        budget_max = int(max(p.price for p in anchors) * rng.choice([1.0, 1.1, 1.25]))
        # 属性词取自锚点（保证相关集非空），1~2 个
        anchor_values = [v for p in anchors for v in p.attributes.values()]
        attrs = rng.sample(sorted(set(anchor_values)), k=min(2, len(set(anchor_values))))
        scene = rng.choice([s for p in anchors for s in p.scene_tags])

        use_attrs = rng.random() < 0.7  # 30% 的 query 只提预算类目（短模板）
        req = Requirements(
            category=category,
            budget_max=float(budget_max),
            attributes=attrs if use_attrs else [],
            scene=scene if use_attrs else None,
        )
        use_scene = scene if use_attrs else None
        relevant = _relevant_set(catalog, req, attrs if use_attrs else [], use_scene)
        # 相关集过大（query 区分度差）或过小（无意义）的行都不入库
        if not (1 <= len(relevant) <= 12):
            continue

        template = rng.choice(_TEMPLATES if use_attrs else _SHORT_TEMPLATES)
        query = template.format(
            budget=budget_max,
            category=category,
            attrs="、".join(attrs) if use_attrs else category,
            scene=scene,
        )
        dataset.append(
            {
                "id": f"pos-{idx:03d}",
                "query": query,
                "requirements": req.model_dump(exclude_none=True),
                "expected_ids": sorted(p.id for p in relevant),
                "should_clarify": bool(compute_missing(req)),
            }
        )
        idx += 1

    # 追问场景：缺预算 / 缺类目 / 全缺 / 仅人群
    clarify_specs = [
        ("想要主动降噪的耳机", Requirements(attributes=["主动降噪"])),
        ("预算2000以内买点什么好", Requirements(budget_max=2000.0)),
        ("帮我挑个礼物", Requirements()),
        ("我妈生日，送她点实用的", Requirements(recipient="送妈妈")),
    ]
    i = 0
    while len(dataset) < size:
        query, req = clarify_specs[i % len(clarify_specs)]
        dataset.append(
            {
                "id": f"clarify-{i:03d}",
                "query": query,
                "requirements": req.model_dump(exclude_none=True),
                "expected_ids": [],
                "should_clarify": bool(compute_missing(req)),
            }
        )
        i += 1
    return dataset


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, default=120)
    args = parser.parse_args()

    dataset = build_dataset(size=args.size)
    out = get_settings().data_dir / "eval" / "queries.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as fh:
        for row in dataset:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    n_clarify = sum(1 for d in dataset if d["should_clarify"])
    sizes = [len(d["expected_ids"]) for d in dataset if d["expected_ids"]]
    print(
        f"[eval-set] {len(dataset)} 条（追问 {n_clarify}）→ {out}\n"
        f"[eval-set] 相关集大小 min/median/max = "
        f"{min(sizes)}/{sorted(sizes)[len(sizes)//2]}/{max(sizes)}"
    )


if __name__ == "__main__":
    main()
