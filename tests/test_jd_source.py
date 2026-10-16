"""京东联盟 adapter：签名算法与字段归一化（纯单元测试，不联网）。"""

import re

import pytest

from shopmind.catalog.jd_source import JdUnionSource


@pytest.fixture
def source() -> JdUnionSource:
    return JdUnionSource("test-key", "test-secret")


def test_requires_credentials():
    with pytest.raises(ValueError):
        JdUnionSource("", "")


def test_md5_sign_format(source):
    params = source._signed_params('{"goodsReqDTO":{"keyword":"耳机"}}')
    # 系统参数齐全、签名为 32 位大写十六进制
    assert params["method"] == "jd.union.open.goods.query"
    assert params["sign_method"] == "md5"
    assert re.fullmatch(r"[0-9A-F]{32}", params["sign"])
    # 参与签名的原始参数不包含 sign 本身
    assert "sign" not in params["360buy_param_json"]


def test_normalize_prefers_coupon_price(source):
    raw = {
        "sku": 100_01,
        "goodsName": "索尼 WH-1000XM5 头戴式无线降噪耳机",
        "priceInfo": {"lowestPrice": 2199.0, "lowestCouponPrice": 1899.0},
        "brandInfo": {"brandName": "索尼"},
        "categoryInfo": {"cidName": "数码;影音娱乐;耳机"},
        "couponInfo": {"couponList": [{"price": 300, "isBest": True}]},
    }
    product = source._normalize(raw, "耳机")
    assert product.id == "JD-10001"
    assert product.price == 1899.0  # 券后价优先
    assert product.brand == "索尼"
    assert product.category == "耳机"  # cidName 取最后一级
    assert product.attributes["优惠"] == "300元券"


def test_normalize_rejects_dirty_row(source):
    assert source._normalize({"sku": 1}, "耳机") is None
    assert source._normalize({"sku": 2, "goodsName": "", "priceInfo": {}}, "耳机") is None
