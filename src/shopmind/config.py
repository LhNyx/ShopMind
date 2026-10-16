"""ShopMind 配置：.env / 环境变量加载，全项目唯一直接读环境的地方。"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv
from pydantic_settings import BaseSettings, SettingsConfigDict

# 项目根目录（src/shopmind/config.py → 上三级）
PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    """全部可调参数集中在此，测试与评测通过实例化覆盖，不改全局状态。"""

    model_config = SettingsConfigDict(
        env_file=(PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ── LLM（OpenAI 兼容端点）──────────────────────────────
    openai_base_url: str = ""
    openai_api_key: str = ""
    openai_model: str = "glm-5.3-flash"

    # ── 京东联盟（可选，未配置则用 mock 数据源）─────────────
    jd_union_app_key: str = ""
    jd_union_secret_key: str = ""
    jd_union_gateway: str = "https://api.jd.com/routerjson"

    # ── Embedding ──────────────────────────────────────────
    hf_endpoint: str = "https://huggingface.co"  # 被墙时改 https://hf-mirror.com
    embedding_model: str = "bge-small-zh-v1.5"
    embedding_dim: int = 512
    embedding_max_len: int = 512
    ort_threads: int = 4

    # ── 检索 / 排序 ────────────────────────────────────────
    budget_tolerance: float = 0.20      # 硬过滤默认预算容差 ±20%
    relax_tolerance: float = 0.40       # 降级时放宽到 ±40%
    retrieve_top_n: int = 20            # 精排后保留的候选数
    fetch_depth: int = 400              # FAISS 粗排取回深度（2k 规模下≈全量精排，暴力精确）
    rerank_attr_boost: float = 0.5      # 精排：属性命中加分（评测扫描 0.15→0.5 饱和）
    rerank_scene_boost: float = 0.2     # 精排：场景命中加分
    recommend_top_k: int = 5            # 最终推荐条数
    score_weight_semantic: float = 0.70
    score_weight_budget: float = 0.20
    score_weight_rating: float = 0.10

    # ── 图护栏 ─────────────────────────────────────────────
    clarify_max_rounds: int = 2         # 追问硬上限，防死循环

    # ── 路径 ───────────────────────────────────────────────
    data_dir: Path = PROJECT_ROOT / "data"

    @property
    def catalog_dir(self) -> Path:
        return self.data_dir / "catalog"

    @property
    def models_dir(self) -> Path:
        return self.data_dir / "models"

    @property
    def reports_dir(self) -> Path:
        return self.data_dir / "reports"


@lru_cache
def get_settings() -> Settings:
    load_dotenv(PROJECT_ROOT / ".env")
    return Settings()
