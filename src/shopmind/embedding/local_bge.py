"""本地 ONNX bge-small-zh-v1.5 embedding：离线推理 + 模型自动下载。

设计对应简历/面试点：
- ONNX Runtime 本地推理，离线可复现，真正兑现「FAISS 本地向量索引，零外部依赖」
- 推理放线程池（asyncio.to_thread），不阻塞事件循环
- 模型流式下载支持 HF_ENDPOINT 镜像（国内网络），磁盘空间预检
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import httpx
import numpy as np

from shopmind.embedding.base import BaseEmbedder, normalize

MODEL_REPO = "Xenova/bge-small-zh-v1.5"  # 官方 BAAI 仓库无 ONNX 导出，Xenova 转换版权重一致
# 仓库内相对路径 → 本地文件名
_MODEL_FILES = {
    "onnx/model.onnx": "model.onnx",
    "tokenizer.json": "tokenizer.json",
}
_MIN_MODEL_BYTES = 50 * 1024 * 1024  # model.onnx 约 91MB，低于阈值视为下载不完整


class LocalBgeEmbedder(BaseEmbedder):
    def __init__(
        self,
        model_dir: Path,
        *,
        hf_endpoint: str = "https://hf-mirror.com",
        dim: int = 512,
        max_len: int = 512,
        ort_threads: int = 4,
    ) -> None:
        self.name = "bge-small-zh-v1.5"
        self.dim = dim
        self._model_dir = model_dir
        self._hf_endpoint = hf_endpoint.rstrip("/")
        self._max_len = max_len
        self._ort_threads = ort_threads
        self._session = None  # 懒加载：首次 embed 时初始化

    # ── 对外接口 ───────────────────────────────────────────
    def embed_documents(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        self._ensure_loaded()
        last_state = self._forward(texts)
        return normalize(last_state[:, 0, :])  # CLS pooling

    def embed_query(self, text: str) -> np.ndarray:
        return self.embed_documents([text])[0]

    async def embed_documents_async(self, texts: list[str]) -> np.ndarray:
        return await asyncio.to_thread(self.embed_documents, texts)

    async def embed_query_async(self, text: str) -> np.ndarray:
        return await asyncio.to_thread(self.embed_query, text)

    # ── 内部实现 ───────────────────────────────────────────
    def _ensure_loaded(self) -> None:
        if self._session is not None:
            return
        self._download_if_missing()
        import onnxruntime as ort
        from tokenizers import Tokenizer

        self._tokenizer = Tokenizer.from_file(str(self._model_dir / "tokenizer.json"))
        self._tokenizer.enable_truncation(max_length=self._max_len)
        # 动态 padding 到批内最长：ONNX 需要规整的 (B, L) 输入，pad_id=0 即 [PAD]
        self._tokenizer.enable_padding(direction="right", pad_id=0, pad_type_id=0)
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = self._ort_threads
        self._session = ort.InferenceSession(
            str(self._model_dir / "model.onnx"), opts, providers=["CPUExecutionProvider"]
        )
        self._input_names = {i.name for i in self._session.get_inputs()}

    def _forward(self, texts: list[str]) -> np.ndarray:
        encodings = self._tokenizer.encode_batch(texts)
        input_ids = np.array([e.ids for e in encodings], dtype=np.int64)
        attention_mask = np.array([e.attention_mask for e in encodings], dtype=np.int64)
        feed = {"input_ids": input_ids, "attention_mask": attention_mask}
        if "token_type_ids" in self._input_names:
            feed["token_type_ids"] = np.zeros_like(input_ids)
        (last_state,) = self._session.run(None, feed)
        return last_state

    def _download_if_missing(self) -> None:
        model_path = self._model_dir / "model.onnx"
        if model_path.exists() and model_path.stat().st_size >= _MIN_MODEL_BYTES:
            return
        self._model_dir.mkdir(parents=True, exist_ok=True)
        free_mb = _free_space_mb(self._model_dir)
        if free_mb < 300:
            sys.exit(f"[embedder] 磁盘剩余 {free_mb}MB 不足，模型需要约 300MB，中止。")
        for remote, local in _MODEL_FILES.items():
            target = self._model_dir / local
            if target.exists() and target.stat().st_size > 0:
                continue
            url = f"{self._hf_endpoint}/{MODEL_REPO}/resolve/main/{remote}"
            print(f"[embedder] 下载 {url} → {target}")
            with httpx.stream("GET", url, follow_redirects=True, timeout=60) as resp:
                resp.raise_for_status()
                tmp = target.with_suffix(target.suffix + ".part")
                with tmp.open("wb") as fh:
                    for chunk in resp.iter_bytes(1024 * 1024):
                        fh.write(chunk)
                tmp.rename(target)
            print(f"[embedder] 完成：{target.stat().st_size // 1024 // 1024}MB")


def _free_space_mb(path: Path) -> float:
    import shutil

    return shutil.disk_usage(path).free / 1024 / 1024
