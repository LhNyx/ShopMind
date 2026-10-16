"""checkpoint 序列化白名单：显式允许领域模型进出 SqliteSaver。

langgraph 正在收紧 msgpack 反序列化的默认放行模式，白名单是官方推荐姿势；
只放行 shopmind.schemas 四个模型，不放行任意模块。
"""

from __future__ import annotations

from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer

from shopmind.schemas import Product, Recommendation, Requirements, RetrievalTrace

_ALLOWED_TYPES = [Product, Requirements, Recommendation, RetrievalTrace]


def make_serde() -> JsonPlusSerializer:
    return JsonPlusSerializer(allowed_msgpack_modules=_ALLOWED_TYPES)
