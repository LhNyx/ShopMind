# ShopMind — 商品推荐导购 Agent

> 用户用自然语言描述需求，Agent 从真实商品目录中检索、筛选、排序，输出**带理由、可溯源**的推荐清单。
> 设计哲学：**AI 不编造商品，只从真实目录里选；每条推荐可溯源到目录条目。**

技术栈：Python 3.12 / FastAPI / LangChain / LangGraph / OpenAI-Compatible API / FAISS / ONNX Runtime（bge-small-zh）/ Pydantic / Docker

架构细节与面试防守点见 [ARCHITECTURE.md](./ARCHITECTURE.md)。

## 快速开始

```bash
uv sync                                          # 安装依赖（含 faiss / onnxruntime）

# 1. 生成商品快照 + 向量索引（--source jd 用京东联盟真实数据，见下文）
uv run scripts/fetch_catalog.py --source mock    # seeded mock 目录，1200 SKU，可复现

# 2. 生成评测集 + 跑评测
uv run scripts/gen_eval_set.py
uv run scripts/run_eval.py --tag final

# 3. CLI 端到端演示（真 LLM，需 .env）
uv run scripts/run_demo.py

# 4. 起服务
uv run uvicorn shopmind.service.app:app --port 8000
```

## API

**SSE 流式推荐**（逐段看到「理解需求 → 筛选 → 推荐理由」）：

```bash
curl -N -X POST localhost:8000/api/chat/stream \
  -H "Content-Type: application/json" \
  -d '{"message": "预算300以内，买个能测心率的智能手表", "thread_id": "u1"}'
```

事件流：`start → stage(正在理解需求) → stage(候选 1200 → 过滤后 10 → 召回 10) → reason×5（逐条流出）→ result → done`

**非流式**（评测/脚本用）：`POST /api/chat`，body 相同。
**多轮反馈**：带同一 `thread_id` 再发「再便宜点，不要三星」——需求合并（预算收紧、黑名单累加）后重新检索。
**断点续跑**：重启服务后再发消息，会话状态从 SqliteSaver checkpoint 恢复，对话不丢。

## 数据源：联盟开放 API → 离线快照

在线链路只查本地 FAISS 索引（毫秒级、零配额），商品目录由**京东联盟开放 API 批量拉取**落盘为版本化快照：

```bash
# .env 配置 JD_UNION_APP_KEY / JD_UNION_SECRET_KEY（union.jd.com 入驻 → 开放平台建应用）
uv run scripts/fetch_catalog.py --source jd --keywords 无线耳机 手机 笔记本 --per-keyword 60
```

为什么离线快照而不是请求时在线调 API：评测可复现（目录漂移会让 Recall 无法归因）、QPS 配额、行业惯例（离线索引 + 在线 serving）。数据源是 adapter 接口，京东联盟与 seeded mock 同 schema，切换零改码；API 审批期间 mock 不阻塞开发。

## 评测

**口径**：expected_ids = 满足 query 全部明示约束的商品全集（预算容差与线上过滤一致 ±20%，相关集 ≤12 保证区分度）；should_clarify 与图路由实际决策比对。检索栈评测用 golden requirements 驱动，隔离 LLM 波动，CI 可跑。

| 指标 | 最终值 | 说明 |
|---|---|---|
| hit_rate@5 | **1.00** | 每条正例 top-5 至少命中一件相关商品 |
| recall@5 | 0.68 | 相关集平均 4 件，top-5 覆盖 68% |
| MRR | 0.75 | 首条相关商品平均排名 ≈ 1.3 |
| 澄清触发准确率 | **1.00** | 澄清决策在确定性代码（compute_missing），可测 |
| 检索延迟 | 2.4 ms | 1200 SKU，过滤+粗排+精排全链路 |

**评测驱动的迭代过程**（`scripts/eval_compare.py` 回归对比，坏例逐条归因）：

| 版本 | 改动 | recall@5 | MRR |
|---|---|---|---|
| base | 纯语义排序（FAISS 余弦直出） | 0.13 | 0.26 |
| v2 | +属性/场景命中精排特征（0.15/0.05） | 0.44 | 0.63 |
| v3 | 标注口径修正（锚点→约束全集≤12）+ 权重扫描定 0.5/0.2 | 0.56 | 0.72 |
| v4 | 精排池扩到全量（2k 规模暴力精确） | **0.71** | 0.75 |

v3 → v4 同目录同评测集：**修复 29 条坏例**，指标变化与坏例清单全部留档（`data/reports/`）。诊断方法：粗排漏召 4/90 vs 精排挤掉 42/90 → 瓶颈在精排池深度，不在向量召回——这个归因过程本身就是评测框架的价值。

## 简历对照（每条 bullet 的验证位置）

| 简历描述 | 代码 | 验证 |
|---|---|---|
| LangGraph 编排图 + 条边 + 2 轮护栏 + 纯函数节点 | `graph/build.py` `graph/nodes.py` | `pytest tests/test_routing.py` 场景2追问2轮放行：`run_demo.py` |
| Pydantic Schema 结构化提取 + 重试降级 | `llm/extractor.py` `graph/rules.py` | `test_graph_smoke.py::test_extraction_failure_falls_back_to_rules` |
| 混合检索：硬过滤在前 + FAISS 语义召回 | `retrieval/filters.py` `retrieval/retriever.py` | `run_eval.py` 指标 + `test_retrieval.py` |
| 可溯源推荐（仅引用目录字段抑制幻觉） | `llm/reasons.py` `nodes._reason_consistent` | `test_reasons.py`（编造价格→拦截降级模板） |
| 多轮反馈合并回结构化需求 | `schemas.merge_requirements` | `test_merge.py` + demo 两轮对话 |
| Recall@K / MRR / 澄清准确率评测框架 | `eval/` `scripts/run_eval.py` | `data/reports/` 报告 + 上表 |
| FastAPI+SSE 流式 + checkpoint 断点续跑 | `service/app.py` `graph/checkpointing.py` | curl SSE；重启后同 thread_id 续跑（上文实测） |

## 目录结构

```
src/shopmind/
├── config.py            # 全部可调参数（.env 加载）
├── schemas.py           # Product / Requirements / 合并语义（确定性，单测覆盖）
├── catalog/             # source adapter（jd/mock 同构）+ 版本化快照 + FAISS 索引
├── embedding/           # ONNX bge 本地推理（线程池）+ 测试用 Hash 向量
├── retrieval/           # 硬过滤 + 逐级放宽 + 粗排→精排
├── graph/               # State / 四节点 / 组图 / 依赖端口 / checkpoint 白名单
├── llm/                 # 端口的 langchain 实现（提取/追问/理由）
├── service/             # FastAPI + SSE
└── eval/                # 指标 + runner
scripts/                 # fetch_catalog / gen_eval_set / run_eval / eval_compare / run_demo
data/                    # catalog(快照+索引) / eval(标注集) / reports(评测报告)
tests/                   # 28 个单测，CI 无 LLM 可跑
```

## 工程化

```bash
uv run pytest          # 28 passed（fake 端口注入，无网络依赖）
uv run ruff check .    # 代码规范
docker build -t shopmind . && docker run -p 8000:8000 -v shopmind-models:/app/data/models shopmind
```
