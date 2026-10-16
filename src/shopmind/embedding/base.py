"""Embedding 抽象：本地 ONNX bge（生产）与字符 n-gram 哈希（测试/降级）同接口。"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class BaseEmbedder(ABC):
    dim: int
    name: str

    @abstractmethod
    def embed_documents(self, texts: list[str]) -> np.ndarray: ...

    @abstractmethod
    def embed_query(self, text: str) -> np.ndarray: ...


def normalize(mat: np.ndarray) -> np.ndarray:
    """L2 归一化：归一化后内积 = 余弦相似度，配合 FAISS IndexFlatIP。"""
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (mat / norms).astype(np.float32)


class HashNGramEmbedder(BaseEmbedder):
    """确定性字符 bigram 哈希向量。

    用途：单元测试与模型缺失时的降级。共享字面片段的文本会有相近向量，
    因此 golden 检索测试无需下载模型即可覆盖语义召回路径。
    """

    def __init__(self, dim: int = 512) -> None:
        self.dim = dim
        self.name = "hash-bigram-512"

    def _encode_one(self, text: str) -> np.ndarray:
        vec = np.zeros(self.dim, dtype=np.float32)
        cleaned = "".join(text.split())
        for i in range(len(cleaned) - 1):
            gram = cleaned[i : i + 2]
            slot = int.from_bytes(gram.encode("utf-8")[:4], "little") % self.dim
            vec[slot] += 1.0
        return vec

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        return normalize(np.stack([self._encode_one(t) for t in texts]))

    def embed_query(self, text: str) -> np.ndarray:
        return self.embed_documents([text])[0]
