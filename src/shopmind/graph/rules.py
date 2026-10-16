"""规则降级提取器：LLM 结构化输出两连败后的最后防线。

能抽多少抽多少：预算数字、类目词典、品牌黑名单、人群词、场景词。
保证链路永不因 LLM 输出问题中断——对应简历「解析失败自动重试降级」。
"""

from __future__ import annotations

import re

from shopmind.schemas import Requirements

# 类目同义词 → 标准类目（与 mock 目录 / 京东关键词对齐）
_CATEGORY_LEXICON = {
    "无线耳机": ["耳机", "蓝牙耳机", "降噪耳机", "耳麦"],
    "手机": ["手机", "5g手机", "旗舰机", "安卓机"],
    "笔记本": ["笔记本", "电脑", "手提", "轻薄本", "游戏本"],
    "智能手表": ["手表", "智能表", "手环"],
    "平板": ["平板", "ipad", "pad"],
    "蓝牙音箱": ["音箱", "音响", "喇叭"],
}

_BRAND_LEXICON = [
    "苹果", "华为", "小米", "三星", "OPPO", "vivo", "一加", "荣耀", "联想",
    "戴尔", "华硕", "惠普", "索尼", "铁三角", "漫步者", "JBL", "佳明", "哈曼卡顿",
]

_RECIPIENT_WORDS = ["妈妈", "爸爸", "女友", "男友", "孩子", "老人", "自己"]
_SCENE_WORDS = [
    "通勤", "运动", "游戏", "办公", "学习", "送礼", "旅行", "居家",
    "拍照", "网课", "绘画", "健康", "户外", "聚会", "设计", "编程", "商务",
]
_ATTR_HINTS = [
    "主动降噪", "被动降噪", "长续航", "大电池", "防水", "轻薄", "高刷",
    "快充", "无线充电", "手写笔", "护眼", "心率监测", "血氧监测", "睡眠监测",
]

# 预算：抓「500 以内 / 1500 左右 / 预算3000 / 3000元以下」等表达
_BUDGET_MAX_RE = re.compile(
    r"(?:预算|价格)?(\d{2,6})\s*(?:元|块)?\s*(?:以内|以下|之内|左右|上下|以内吗)?"
)
_BUDGET_MIN_RE = re.compile(r"(\d{2,6})\s*(?:元|块)?\s*(?:以上|起步)")
_EXCLUDE_RE = re.compile(r"(?:不要|排除|别|不看|去掉)\s*([\u4e00-\u9fa5A-Za-z0-9]{1,10})")


def extract_requirements_by_rules(text: str) -> Requirements:
    normalized = text.lower().replace(" ", "")
    req = Requirements()

    # 类目：词典最长命中
    for category, aliases in _CATEGORY_LEXICON.items():
        if category in normalized or any(a in normalized for a in aliases):
            req.category = category
            break

    # 预算：优先「以上」下限，再取第一个数字当上限（中文表达里上限远多于下限）
    if (m := _BUDGET_MIN_RE.search(normalized)) and 50 <= int(m.group(1)) <= 100000:
        req.budget_min = float(m.group(1))
    elif (m := _BUDGET_MAX_RE.search(normalized)) and 50 <= int(m.group(1)) <= 100000:
        req.budget_max = float(m.group(1))

    # 品牌排除：「不要小米」「除了华为」
    for m in _EXCLUDE_RE.finditer(normalized):
        tail = m.group(1)
        for brand in _BRAND_LEXICON:
            if brand.lower() in tail.lower():
                req.exclusions.append(brand)

    # 人群 / 场景 / 属性提示
    for word in _RECIPIENT_WORDS:
        if word in normalized:
            req.recipient = f"送{word}" if word != "自己" else "自用"
            break
    for word in _SCENE_WORDS:
        if word in normalized:
            req.scene = word
            break
    req.attributes = [hint for hint in _ATTR_HINTS if hint in normalized]

    return req
