"""京东联盟开放 API adapter（jd.union.open.goods.query）。

鉴权：MD5 签名（系统参数按 key ASCII 排序拼接 k=v，secretKey 包夹后取大写 MD5）。
产出：归一化到统一 Product schema 的商品快照（券后价优先）。

注意事项（面试可讲）：
- 联盟接口对个人应用有 QPS 配额 → 批量拉取一次落盘为快照，在线链路零依赖
- 返回字段里佣金/推广链接与推荐无关，归一化时丢弃，只留目录字段
"""

from __future__ import annotations

import hashlib
import json
import time
from datetime import datetime

import httpx

from shopmind.schemas import Product

SOURCE_NAME = "jd"
_METHOD = "jd.union.open.goods.query"
_PAGE_SIZE = 50


class JdUnionSource:
    name = SOURCE_NAME

    def __init__(
        self,
        app_key: str,
        secret_key: str,
        gateway: str = "https://api.jd.com/routerjson",
        *,
        request_interval: float = 0.35,  # 礼貌性限速，个人应用 QPS 配额有限
        timeout: float = 15.0,
    ) -> None:
        if not app_key or not secret_key:
            raise ValueError("京东联盟 app_key / secret_key 未配置（.env 的 JD_UNION_*）")
        self._app_key = app_key
        self._secret_key = secret_key
        self._gateway = gateway
        self._interval = request_interval
        self._timeout = timeout

    def fetch(self, keyword: str, max_items: int) -> list[Product]:
        """按关键词分页拉取商品并归一化。失败重试 2 次后抛出，由调用方决定降级。"""
        products: list[Product] = []
        seen: set[str] = set()
        page = 1
        while len(products) < max_items:
            batch = self._search_page(keyword, page)
            if not batch:
                break
            for raw in batch:
                product = self._normalize(raw, keyword)
                if product is not None and product.id not in seen:
                    seen.add(product.id)
                    products.append(product)
                    if len(products) >= max_items:
                        break
            page += 1
            time.sleep(self._interval)
        return products

    # ── 协议细节 ───────────────────────────────────────────
    def _search_page(self, keyword: str, page: int) -> list[dict]:
        biz = {"goodsReqDTO": {"keyword": keyword, "pageIndex": page, "pageSize": _PAGE_SIZE}}
        params = self._signed_params(json.dumps(biz, ensure_ascii=False, separators=(",", ":")))
        last_error: Exception | None = None
        for attempt in range(3):
            try:
                resp = httpx.post(self._gateway, data=params, timeout=self._timeout)
                resp.raise_for_status()
                body = resp.json()
                error = body.get("error_response")
                if error:
                    raise RuntimeError(f"京东联盟接口报错：{error}")
                result = body["jd_union_open_goods_query_responce"]["queryResult"]
                return json.loads(result).get("data", [])
            except Exception as exc:  # noqa: BLE001 — 网络/协议异常统一重试
                last_error = exc
                time.sleep(0.5 * (attempt + 1))
        raise RuntimeError(
            f"jd.union.open.goods.query 重试 3 次仍失败：{last_error}"
        ) from last_error

    def _signed_params(self, param_json: str) -> dict:
        sys_params = {
            "method": _METHOD,
            "app_key": self._app_key,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "format": "json",
            "v": "1.0",
            "sign_method": "md5",
            "360buy_param_json": param_json,
        }
        ordered = sorted(sys_params.items())
        plain = self._secret_key + "".join(f"{k}{v}" for k, v in ordered) + self._secret_key
        sys_params["sign"] = hashlib.md5(plain.encode("utf-8")).hexdigest().upper()
        return sys_params

    # ── 归一化：平台字段 → 统一 schema ─────────────────────
    def _normalize(self, raw: dict, keyword: str) -> Product | None:
        try:
            sku = str(raw["sku"])
            name = str(raw.get("goodsName", "")).strip()
            price_info = raw.get("priceInfo", {})
            price = (
                price_info.get("lowestCouponPrice")
                or price_info.get("lowestPrice")
                or price_info.get("price")
            )
            if not name or price is None or float(price) <= 0:
                return None
            brand = (
                raw.get("brandInfo", {}).get("brandName")
                or raw.get("brandName")
                or name.split()[0]
            )
            category = raw.get("categoryInfo", {}).get("cidName", "").split(";")[-1] or keyword
            coupon = (raw.get("couponInfo", {}) or {}).get("couponList", [])
            attrs: dict[str, str] = {}
            if coupon and coupon[0].get("isBest"):
                attrs["优惠"] = f"{coupon[0].get('price', '')}元券"
            return Product(
                id=f"JD-{sku}",
                name=name,
                brand=str(brand).strip(),
                category=str(category).strip() or keyword,
                price=round(float(price), 2),
                attributes=attrs,
                scene_tags=[],
                rating=None,
                source=SOURCE_NAME,
            )
        except (KeyError, TypeError, ValueError):
            return None  # 单条脏数据不影响整页
