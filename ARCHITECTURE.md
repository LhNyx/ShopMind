# ShopMind 商品推荐导购 Agent — 架构设计文档

> 版本：v1.0（待确认后开工）
> 对应简历：ShopMind 商品推荐导购 Agent（2026.07–2026.09）
> 设计原则：**简历每一条 bullet 都必须对应一个可运行、可评测、可追问的模块。**

---

## 0. 结论先行：能不能接淘宝 / 京东？

**能接，但接入形态是「联盟开放 API → 离线商品快照」，不是请求时在线调用，更不是爬虫。**

三个平台的事实门槛：

| 平台 | 个人可申请 | 关键说明 | 定位 |
|---|---|---|---|
| **京东联盟** | ✅ 可以 | 联盟入驻（个人身份）→ 开放平台建应用，媒体类型选「聊天工具」可免网站备案；核心接口 `jd.union.open.goods.query`（关键词搜商品，含价格/券/主图/类目） | **主选** |
| 拼多多 多多进宝 | ✅ 可以 | 个人门槛最低，`pdd.ddk.goods.search`；但商品池低端化，价格带不适合演示预算分层 | 备选 adapter |
| 淘宝联盟（淘宝客） | ⚠️ 基本不行 | 近年个人新号 API 权限基本批不下来，以企业/渠道媒体为主 | 不做主依赖 |
| 爬虫 | ❌ 不做 | 反爬 + 合规风险；AI 岗简历上写爬虫是减分项 | 明确排除 |

**为什么是离线快照而不是在线调用 API（面试必问，先想清楚）：**

1. **评测可复现**。Recall@K / MRR 回归测试的前提是目录稳定；在线 API 的价格、上下架每天都在漂移，今天 0.82 明天 0.79，指标变化无法归因是提示词还是目录本身变了。
2. **性能与配额**。联盟 API 对个人应用有 QPS 限制，推荐链路一次要过滤 + 召回几百 SKU；本地 FAISS 毫秒级、零配额。
3. **行业惯例**。电商搜索/推荐本来就是「离线索引 + 在线 serving」，LLM 插在两端（需求理解、结果解释），不替代中间检索——这正是简历里写的设计哲学。
4. **故事更完整**。「批量拉取 → 清洗标准化 → 版本化快照 → 建索引 → 评测报告」是一条真实的电商数据管线，比"每次请求现调 API"更能体现工程能力。

**快照还带来一个面试加分点**：目录带版本号（catalog hash），每次评测报告绑定快照版本，指标可追溯、可复现——这是评测工程的正统做法。

**Plan B（必须内建，因为 API 审批周期 1~3 个工作日、结果不受控）**：
数据源做成 **adapter 接口**（`CatalogSource`），京东联盟和 seeded mock 实现同一接口、同一 schema。API 没批下来也能全链路开发评测，批下来后 `uv run scripts/fetch_catalog.py --source jd` 一条命令换成真实数据，**代码零改动**。

---

## 1. 垂类与目录

**v1 定为 3C 数码垂类**（手机 / 耳机 / 笔记本 / 智能手表 / 平板等 6~8 个子类目，目标 1~2k SKU）：

- 属性结构化程度最高（内存 / 屏幕 / 续航 / 用途），硬过滤 + 语义召回的分工最好演示
- 价格带 100~15000 元，预算分层数十条，评测集能构造出明显的预算边界 case
- 架构上**类目无关**：扩全品类只需要改抓取关键词和类目表，不动代码

统一商品 schema（无论数据源是谁）：

```python
class Product(BaseModel):
    id: str                    # 源平台 SKU 归一化 ID
    name: str
    brand: str
    category: str              # 归一化类目，如 "无线耳机"
    price: float               # 券后价（联盟 API 的实际成交口径）
    attributes: dict[str, str] # 类目相关属性：{"降噪": "主动降噪", "续航": "30小时"}
    scene_tags: list[str]      # 使用场景标签：["通勤", "运动"]
    rating: float | None       # 评分先验（排序用，可缺省）
    source: str                # jd / mock，溯源字段
```

---

## 2. 系统总览

```
┌─────────────────────── 离线管线（定时/手动触发）───────────────────────┐
│                                                                      │
│  CatalogSource (adapter)          Catalog Indexer                    │
│  ├─ JdUnionSource  ──fetch──►  normalize ─► dedupe ─► filter ─► snapshot.jsonl
│  └─ MockSource(seeded)                    catalog_meta.json (hash+版本) │
│                                                  │                   │
│                            bge-small-zh (ONNX本地) ▼                   │
│                                              faiss.index              │
└──────────────────────────────────────────────────────────────────────┘
                                                                       │
┌──────────────────────────── 在线服务（FastAPI + SSE）──────────────────▼──┐
│                                                                          │
│   POST /api/chat/stream ──► LangGraph                                    │
│                                                                          │
│   understand ──(need_clarify 且 rounds<2)──► clarify ──► await_answer ─┐  │
│       ▲                                              interrupt() 挂起   │  │
│       └──────────────── 用户回答后 resume（回边）◄─────────────────────┘  │
│       │                                                                  │
│     (否则) retrieve ──► rank_and_explain ──► END                          │
│                                                                          │
│   retrieve = 硬过滤（预算/类目/排除项）→ FAISS 语义召回                     │
│   SqliteSaver checkpoint：thread_id 隔离，断点续跑                        │
└──────────────────────────────────────────────────────────────────────────┘
```

---

## 3. 核心模块设计

### 3.1 State 与结构化需求（Pydantic，简历 bullet 2）

```python
class Requirements(BaseModel):
    category: str | None
    budget_min: float | None
    budget_max: float | None
    attributes: list[str]      # ["主动降噪", "长续航"]
    scene: str | None          # "通勤"/"送妈妈养花"
    recipient: str | None      # "妈妈"/"自己"
    exclusions: list[str]      # ["小米"]
    missing_fields: list[str]  # 结构化提取时发现的缺口，clarify 节点据此提问

class RecommendationState(TypedDict, total=False):
    messages: Annotated[list[AnyMessage], add_messages]
    requirements: Requirements
    need_clarify: bool
    clarify_rounds: int        # 硬上限 2
    candidates: list[Product]
    recommendations: list[Recommendation]
```

- `understand` 节点：`chat_model.with_structured_output(Requirements)`；**解析失败重试 1 次 → 降级规则提取**（正则抽预算数字、词典匹配类目），保证链路永不因 LLM 输出问题中断
- `clarify` 节点：只针对 `missing_fields` 里最关键的一个缺口生成追问（缺预算问预算、缺类目问类目），不泛泛而问
- `await_answer` 节点：`interrupt()` 真正挂起等人（checkpoint 保存现场）；HTTP 侧下一句带同一 `thread_id` 的话就是回答，用 `Command(resume=...)` 续跑。问题只在 clarify 生成一次，恢复时不重复调 LLM

### 3.2 图与护栏（简历 bullet 1）

- 五节点：`understand → clarify → await_answer → retrieve → rank_and_explain`，条件边 + 回边
- 路由：`need_clarify and clarify_rounds < 2 → clarify，否则 retrieve`（2 轮后强制放行，带缺口检索好过把用户问烦）
- 节点纯函数：只读 State、返回增量 dict，全链路状态可追溯（checkpoint 回放的基石）

### 3.3 混合检索（简历 bullet 3）

顺序固定：**结构化硬过滤在前，语义召回在后**。

1. **硬过滤**（确定性，代码不是 LLM）：预算区间（缺省放宽 ±20%）、类目树匹配、品牌黑名单（exclusions）
2. **零结果降级**：逐级放宽——先放预算边界 → 再放类目到父类，并在 State 记录放宽原因，结果里如实告知用户（面试点：检索系统的失败兜底）
3. **语义召回**：过滤后的子集上做 FAISS 查询。`IndexFlatIP` + L2 归一化 = 余弦相似；embedding 文本 = `名称 + 类目 + 关键属性 + 场景标签` 模板拼接
4. **为什么 Flat 不用 IVF/HNSW**：1~2k 规模暴力检索最准且微秒级，近似索引是规模上来后的事——「知道什么时候不需要复杂技术」本身是面试加分项

**Embedding：本地 ONNX bge-small-zh-v1.5**（512 维，CLS pooling + L2 归一化）。离线、可复现、真正兑现简历里「FAISS 本地零外部依赖」；首次运行自动下载模型并缓存。

### 3.4 可溯源推荐与多轮反馈（简历 bullet 4）

- **排序**：确定性打分（语义相似 × 预算贴合度 × 评分先验加权），不用 LLM 排序——排序必须可解释、可回归
- **理由生成**：LLM 逐条生成，提示词强约束「只能引用给定商品字段」；输出后做**一致性校验**（理由中的价格必须与目录一致），不一致的条目降级为模板理由
- **多轮反馈**：用户说「再便宜点 / 排除小米」，下一轮 `understand` 携带**历史 requirements + 新消息**做合并——标量覆盖、列表并集、exclusions 累加、budget 取更严的区间——然后重新走检索。checkpoint 保证跨轮状态不丢

### 3.5 评测框架（简历 bullet 5 前半）

- `data/eval/queries.jsonl`：120 条（90 正例 + 30 追问场景）
  - **标注口径**：query 由结构化约束 + 模板反向构造（确定种子可复现，人工抽检定位到行）；`expected_ids` = 满足 query 全部明示约束的商品**全集**（预算容差与线上过滤一致，相关集 ≤12 保证区分度）
  - 教训：最初用「锚点商品」标注（抽样 3 件当期望），在同类商品成百上千的目录里没有区分信号，recall@5 仅 0.033——口径错了指标无意义，先修口径再谈优化
- 指标：**hit_rate@5（主指标：top-5 至少命中一件）、Recall@5/@10、MRR、澄清触发准确率**
- runner 离线跑全图出 `metrics.json` + 坏例清单（未命中 → 定位是理解错 / 过滤错 / 召回错 / 精排错）
- `scripts/eval_compare.py old.json new.json`：改提示词或检索参数前后回归对比，输出指标增量 + 修复/新引入的坏例清单
- 评测报告绑定快照版本号（目录漂移会让归因失效），最终结果见 README 评测章节

### 3.6 服务化（简历 bullet 5 后半）

- `POST /api/chat/stream`：SSE，事件类型 `stage`（正在理解需求 → 正在筛选 → 正在生成理由）+ `clarify`（追问）+ `token`（理由逐字流式）+ `result` + `done`
- `POST /api/chat`：非流式，评测与脚本用
- `GET /api/health`、`GET /api/catalog/meta`（快照版本、SKU 数）
- **checkpoint**：`SqliteSaver` 落盘，thread_id 即会话 ID——服务重启不丢对话，多用户天然隔离（简历原话的可验证实现）
- 配置走 `.env`（OpenAI 兼容端点，复用你中转站的 `glm-5.3-flash`）

### 3.7 工程化

- uv + Python 3.13；ruff + pytest；Dockerfile（多阶段构建）
- 测试策略：图路由 / 需求合并 / 硬过滤 / 降级逻辑用 fake LLM 做确定性单测；检索质量用 golden 集回归——**CI 里不调真模型，评测单独跑**

---

## 4. 目录结构

```
ShopMind/
├── ARCHITECTURE.md          # 本文档
├── README.md                # 项目说明 + 评测报告（跑出来后填）
├── pyproject.toml           # uv 管理
├── Dockerfile
├── .env.example
├── data/
│   ├── catalog/             # snapshot.jsonl + catalog_meta.json + faiss.index
│   └── eval/queries.jsonl   # 标注集
├── scripts/
│   ├── fetch_catalog.py     # --source jd|mock，联盟 API 或 seeded mock
│   ├── build_index.py       # 快照 → 向量索引
│   └── run_eval.py / eval_compare.py
├── src/shopmind/
│   ├── config.py            # pydantic-settings，.env 加载
│   ├── schemas.py           # Product / Requirements / Recommendation
│   ├── catalog/             # source adapter（jd.py / mock.py）+ loader
│   ├── embedding/           # ONNX bge，线程池推理
│   ├── retrieval/           # 硬过滤 + FAISS 召回 + 零结果降级
│   ├── graph/               # state / nodes / build（LangGraph）
│   ├── service/             # app.py（FastAPI + SSE）
│   └── eval/                # metrics + runner
└── tests/                   # fake-LLM 单测 + 检索回归
```

---

## 5. 里程碑（已全部完成 ✅ 2026-10-16）

| # | 内容 | 验收物 | 状态 |
|---|---|---|---|
| M1 | schema + mock 目录 + 硬过滤 + FAISS 召回 | `检索("预算500的耳机")` 返回合理 top-5 | ✅ |
| M2 | LangGraph 全图 + 结构化提取 + 追问护栏 + 多轮反馈 | `run_demo.py` 两场景跑通（真 LLM） | ✅ |
| M3 | 评测框架 + 标注集 + 报告 | hit_rate@5 1.0 / recall@5 0.68 / MRR 0.75 / 澄清 1.0 | ✅ |
| M4 | FastAPI + SSE + SqliteSaver checkpoint | SSE 事件流可看；重启进程同 thread_id 续跑实测通过 | ✅ |
| M5 | 京东联盟 adapter + Docker + README 定稿 | `--source jd` 一条命令切真实数据（待 API 审批） | ✅ 代码就绪 |

**待办（API 审批通过后）**：在 .env 填 `JD_UNION_APP_KEY/SECRET_KEY` → `uv run scripts/fetch_catalog.py --source jd` → 重跑 `run_eval.py` 刷新报告。入驻入口 https://union.jd.com（媒体类型选「聊天工具」，免备案）。

---

## 6. 面试防守预演（每条 bullet 的灵魂三问）

| 追问 | 答案要点 |
|---|---|
| 为什么硬过滤在前、语义召回在后？ | 先用确定性条件把候选集从全目录缩到几十个，语义召回更准（不被预算外的相似商品污染）且省算力；反过来先语义后过滤会丢召回 |
| 澄清为什么上限 2 轮？ | 追问的成本是用户耐心；2 轮后带缺口检索 + 结果里说明假设，比无限追问体验好；上限做成常量可调，评测里对比过 1/2/3 轮 |
| 为什么不在线调电商 API？ | 评测可复现 / QPS 配额 / 行业惯例（离线索引+在线 serving），见 §0 |
| 怎么证明改好了？ | 评测报告：Recall@5 / MRR / 澄清准确率的版本化回归对比 + 坏例归因 |
| 推荐理由的幻觉怎么防？ | 双保险：提示词只允许引用目录字段 + 输出后价格一致性校验，不一致降级模板理由 |
| 不做什么？ | 实时价格库存、个性化历史、下单链路——主动划出去，讲清楚代价 |

---

## 7. 决策记录

| 日期 | 决策 |
|---|---|
| 2026-10-16 | 数据源形态定为「京东联盟 API → 离线商品快照」，mock 同构 adapter 兜底（曾考虑在线实时方案，因评测可复现 / QPS 配额 / 行业惯例三因素放弃） |
| 2026-10-16 | v1 垂类定 3C 数码（架构类目无关，扩品类只改抓取关键词） |
| 2026-10-16 | 按 M1→M5 全量实施，每个里程碑交付可验收命令 |
