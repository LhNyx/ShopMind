# ── 构建阶段：uv 解析依赖 ──────────────────────────────────
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim AS builder
WORKDIR /app
COPY pyproject.toml uv.lock ./
COPY src ./src
RUN uv sync --frozen --no-dev --no-install-project

# ── 运行阶段 ───────────────────────────────────────────────
FROM python:3.12-slim-bookworm
WORKDIR /app
ENV PATH="/app/.venv/bin:$PATH" \
    HF_ENDPOINT=https://hf-mirror.com

COPY --from=builder /app/.venv /app/.venv
COPY src ./src
COPY scripts ./scripts
# 商品快照随镜像走；向量模型（data/models）挂 volume 首次启动自动下载
COPY data/catalog ./data/catalog
COPY data/eval ./data/eval

EXPOSE 8000
VOLUME ["/app/data/models"]
CMD ["uvicorn", "shopmind.service.app:app", "--host", "0.0.0.0", "--port", "8000"]
