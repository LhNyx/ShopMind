"""数据源抽象：京东联盟快照与 seeded mock 目录实现同一接口（Plan B 的落点）。

切换数据源 = 换 adapter 实现，业务代码零改动——这是「离线快照」方案里
API 审批周期不受控时的风险对冲。
"""

from __future__ import annotations

from typing import Protocol


class CatalogSource(Protocol):
    """商品目录数据源协议：按关键词分页拉取，输出统一 schema 的商品。"""

    name: str

    def fetch(self, keyword: str, max_items: int) -> list[Product]: ...


from shopmind.schemas import Product  # noqa: E402  (置于底部避免循环导入提示)


def doc_text(p: Product) -> str:
    """商品 → embedding 输入文本：名称 + 类目 + 品牌 + 属性 + 场景标签。"""
    attrs = "，".join(f"{k}：{v}" for k, v in p.attributes.items())
    scenes = "、".join(p.scene_tags)
    parts = [p.name, f"类目：{p.category}", f"品牌：{p.brand}"]
    if attrs:
        parts.append(f"属性：{attrs}")
    if scenes:
        parts.append(f"场景：{scenes}")
    return "。".join(parts)


def query_text_from_requirements(
    category: str | None,
    attributes: list[str],
    scene: str | None,
    recipient: str | None,
) -> str:
    """结构化需求 → 查询向量文本。预算是数字不是语义，刻意不放进去。"""
    parts = [category, *attributes, scene, recipient]
    return " ".join(p for p in parts if p)
