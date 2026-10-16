"""推荐理由：价格一致性校验（防幻觉的落地手段）与模板降级。"""

from shopmind.graph.nodes import _reason_consistent, _template_reason
from shopmind.schemas import Product, Requirements


def _product(price: float = 329.0) -> Product:
    return Product(
        id="X1",
        name="漫步者 无线耳机",
        brand="漫步者",
        category="无线耳机",
        price=price,
        attributes={"降噪": "主动降噪"},
        scene_tags=["通勤"],
    )


def test_reason_price_consistent():
    assert _reason_consistent("329元到手，主动降噪通勤神器", _product())
    assert _reason_consistent("券后价格 329.0 元", _product())
    assert not _reason_consistent("299元到手超值", _product())  # 编造价格 → 拦截


def test_reason_without_price_is_accepted():
    assert _reason_consistent("主动降噪，通勤舒适", _product())


def test_template_reason_cites_real_fields():
    reason = _template_reason(_product(), Requirements(scene="通勤", budget_max=500))
    assert "329" in reason  # 引用目录真实价格
    assert "通勤" in reason  # 引用场景标签
