"""Seeded mock 目录：3C 数码垂类，随机种子固定 → 完全可复现。

为什么 mock 数据源是正式能力而不是权宜之计（面试点）：
- 评测与单测需要确定性的目录，在线 API 做不到
- 属性 schema 与真实联盟快照完全同构，切换数据源零改码
- 属性/场景/价格带分布可控，能构造预算边界、同价竞品等评测边界 case
"""

from __future__ import annotations

import random

from shopmind.schemas import Product

SOURCE_NAME = "mock"

# 每个子类目的生成参数：品牌池 / 系列词 / 价格带 / 属性取值池 / 场景池
_CATEGORY_SPECS: dict[str, dict] = {
    "无线耳机": {
        "brands": ["索尼", "苹果", "华为", "小米", "三星", "OPPO", "漫步者", "铁三角"],
        "price": (99, 2999),
        "attrs": {
            "降噪": ["主动降噪", "被动降噪", "无降噪"],
            "续航": ["24小时续航", "30小时续航", "40小时续航", "8小时续航"],
            "佩戴": ["入耳式", "头戴式", "半入耳式"],
            "防水": ["IPX4防水", "IPX5防水", "不防水"],
        },
        "scenes": ["通勤", "运动", "办公", "游戏", "旅行"],
    },
    "手机": {
        "brands": ["华为", "小米", "苹果", "三星", "OPPO", "vivo", "一加", "荣耀"],
        "price": (599, 12999),
        "attrs": {
            "存储": ["256G", "512G", "1T"],
            "内存": ["8G", "12G", "16G"],
            "屏幕": ["OLED屏", "LCD屏", "2K屏", "高刷屏"],
            "电池": ["5000mAh", "5500mAh", "6000mAh大电池"],
            "特色": ["5G", "快充", "无线充电", "卫星通信"],
        },
        "scenes": ["游戏", "拍照", "商务", "学生", "送礼"],
    },
    "笔记本": {
        "brands": ["联想", "华为", "苹果", "戴尔", "华硕", "小米", "惠普"],
        "price": (2499, 21999),
        "attrs": {
            "处理器": ["i5", "i7", "R7", "M3芯片"],
            "内存": ["16G", "32G", "8G"],
            "硬盘": ["512G固态", "1T固态"],
            "屏幕": ["14英寸", "15.6英寸", "16英寸", "2.8K屏"],
            "特色": ["轻薄本", "游戏本", "办公本", "触控屏"],
        },
        "scenes": ["办公", "学习", "游戏", "设计", "编程"],
    },
    "智能手表": {
        "brands": ["华为", "苹果", "小米", "三星", "佳明", "OPPO"],
        "price": (149, 4999),
        "attrs": {
            "续航": ["14天续航", "7天续航", "36小时续航"],
            "屏幕": ["AMOLED屏", "LCD屏", "1.4英寸屏"],
            "防水": ["5ATM防水", "IP68防水", "50米防水"],
            "健康": ["心率监测", "血氧监测", "睡眠监测", "ECG心电图"],
        },
        "scenes": ["运动", "健康", "户外", "商务", "送礼"],
    },
    "平板": {
        "brands": ["苹果", "华为", "小米", "三星", "联想", "OPPO"],
        "price": (999, 8999),
        "attrs": {
            "屏幕": ["11英寸", "12.6英寸", "13英寸", "2.5K屏", "144Hz高刷"],
            "存储": ["128G", "256G", "512G"],
            "特色": ["手写笔支持", "键盘支持", "护眼认证", "儿童模式"],
        },
        "scenes": ["学习", "网课", "绘画", "娱乐", "办公"],
    },
    "蓝牙音箱": {
        "brands": ["小米", "华为", "JBL", "索尼", "哈曼卡顿", "漫步者"],
        "price": (59, 2999),
        "attrs": {
            "功率": ["30W", "60W", "100W大功率"],
            "续航": ["12小时续航", "20小时续航", "24小时续航"],
            "防水": ["IPX7防水", "IP67防水", "不防水"],
        },
        "scenes": ["居家", "户外", "聚会", "旅行"],
    },
}

# 类目 → 品牌 → 系列词（无映射的品牌不带系列词，避免「OPPO iPhone」这类乱配）
_SERIES: dict[str, dict[str, list[str]]] = {
    "手机": {
        "苹果": ["iPhone"], "华为": ["Mate", "Pura"], "小米": ["Xiaomi", "Redmi"],
        "三星": ["Galaxy S", "Galaxy A"], "OPPO": ["Reno", "Find"], "vivo": ["X", "S"],
        "一加": ["Ace"], "荣耀": ["Magic"],
    },
    "无线耳机": {
        "苹果": ["AirPods"], "华为": ["FreeBuds"], "小米": ["RedmiBuds"],
        "索尼": ["WH", "WF"], "三星": ["Galaxy Buds"], "OPPO": ["Enco"],
        "漫步者": ["NeoBuds", "W"], "铁三角": ["ATH"],
    },
    "笔记本": {
        "苹果": ["MacBook Air", "MacBook Pro"], "联想": ["小新", "拯救者", "ThinkBook"],
        "华为": ["MateBook"], "戴尔": ["XPS", "灵越"], "华硕": ["天选", "无双"],
        "小米": ["RedmiBook"], "惠普": ["星Book"],
    },
    "智能手表": {
        "苹果": ["Watch"], "华为": ["Watch GT"], "小米": ["Watch", "Redmi Watch"],
        "三星": ["Galaxy Watch"], "佳明": ["Forerunner", "Venu"], "OPPO": ["Watch"],
    },
    "平板": {
        "苹果": ["iPad", "iPad Air"], "华为": ["MatePad"], "小米": ["Redmi Pad", "Xiaomi Pad"],
        "三星": ["Galaxy Tab"], "联想": ["小新 Pad"], "OPPO": ["Pad"],
    },
    "蓝牙音箱": {
        "小米": ["Sound"], "华为": ["Sound"], "JBL": ["Flip", "Charge", "Pulse"],
        "索尼": ["SRS"], "哈曼卡顿": ["Onyx", "Go"], "漫步者": ["M"],
    },
}

# 跨类目通用的人群词（recipient 提取词典也用它）
_RECIPIENTS = ["送妈妈", "送爸爸", "送女友", "送男友", "送孩子", "送老人", "自用"]


def generate_catalog(seed: int = 42, per_category: int = 200) -> list[Product]:
    """生成固定种子的 mock 目录。同参数必出同一目录——评测可复现的基石。"""
    rng = random.Random(seed)
    products: list[Product] = []
    seq = 0
    for category, spec in _CATEGORY_SPECS.items():
        for _ in range(per_category):
            seq += 1
            brand = rng.choice(spec["brands"])
            series = rng.choice(_SERIES[category].get(brand, [""]))
            num = rng.randint(1, 9)
            # 挑 2~4 个属性
            n_attrs = rng.randint(2, min(4, len(spec["attrs"])))
            attr_keys = rng.sample(list(spec["attrs"]), k=n_attrs)
            attributes = {k: rng.choice(spec["attrs"][k]) for k in attr_keys}
            attr_hint = " ".join(list(attributes.values())[:2])
            parts = [brand]
            if series:
                parts.append(f"{series}{num}")
            parts += [category, attr_hint]
            name = " ".join(p for p in parts if p)
            lo, hi = spec["price"]
            products.append(
                Product(
                    id=f"MOCK-{seq:04d}",
                    name=name,
                    brand=brand,
                    category=category,
                    price=round(rng.uniform(lo, hi), 0),
                    attributes=attributes,
                    scene_tags=rng.sample(spec["scenes"], k=rng.randint(1, 3)),
                    rating=round(rng.uniform(3.6, 5.0), 1),
                    source=SOURCE_NAME,
                )
            )
    return products
