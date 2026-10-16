"""领域模型：商品、结构化需求、推荐结果与需求合并语义。

合并语义放在代码里而不是提示词里（面试考点）：
LLM 只负责从「最新一条消息」提取新增/变更的需求，多轮合并的规则是确定性的、
可单测的——这是澄清触发准确率可以被量化的前提。
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class Product(BaseModel):
    """统一商品 schema：京东联盟快照与 mock 目录共用同一形状。"""

    id: str
    name: str
    brand: str = "未知"
    category: str
    price: float
    attributes: dict[str, str] = Field(default_factory=dict)
    scene_tags: list[str] = Field(default_factory=list)
    rating: float | None = None
    source: str = "mock"  # mock / jd，溯源字段


class Requirements(BaseModel):
    """结构化需求：LLM 结构化输出的目标 schema，也是硬过滤的唯一输入。"""

    category: str | None = None
    budget_min: float | None = None
    budget_max: float | None = None
    attributes: list[str] = Field(default_factory=list)
    scene: str | None = None
    recipient: str | None = None
    exclusions: list[str] = Field(default_factory=list)
    missing_fields: list[str] = Field(default_factory=list)  # 由 compute_missing 计算，LLM 不填

    model_config = {"extra": "ignore"}


class RetrievalTrace(BaseModel):
    """检索过程留痕：放宽了什么、候选量变化，坏例归因与面试演示都靠它。"""

    total_catalog: int
    after_filter: int
    relaxations: list[str] = Field(default_factory=list)
    recalled: int


class Recommendation(BaseModel):
    product: Product
    score: float
    reason: str
    reason_source: str = "llm"  # llm / template（一致性校验不过时降级）


# ── 多轮需求合并（确定性规则，对应简历「反馈合并回结构化需求」）────────────


def merge_requirements(prev: Requirements | None, new: Requirements) -> Requirements:
    """把新提取的需求合并进历史需求。

    规则：
    - 标量字段（category / scene / recipient / 预算端点）：新值覆盖旧值
    - 列表字段（attributes / exclusions）：并集、保序、去重
    - exclusions 只增不减（用户说"不要小米"后不会因为后续消息恢复）
    """
    if prev is None:
        return new

    return Requirements(
        category=new.category or prev.category,
        budget_min=new.budget_min if new.budget_min is not None else prev.budget_min,
        budget_max=new.budget_max if new.budget_max is not None else prev.budget_max,
        attributes=_union(prev.attributes, new.attributes),
        scene=new.scene or prev.scene,
        recipient=new.recipient or prev.recipient,
        exclusions=_union(prev.exclusions, new.exclusions),
    )


def _union(a: list[str], b: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in [*a, *b]:
        key = item.strip().lower()
        if key and key not in seen:
            seen.add(key)
            out.append(item.strip())
    return out


# 缺口优先级：预算 > 类目（一次只追问最关键的一个）
MISSING_PRIORITY = ["budget", "category"]


def compute_missing(req: Requirements) -> list[str]:
    """确定性计算信息缺口——澄清触发准确率能被量化的关键：决策在代码不在 LLM。"""
    missing: list[str] = []
    if req.budget_min is None and req.budget_max is None:
        missing.append("budget")
    if not req.category and not req.attributes:
        missing.append("category")
    return [m for m in MISSING_PRIORITY if m in missing]
