# MarketAgent · 上市公司公告情报官

> 基于 A2A（Agent2Agent）协议的跨框架多智能体投研系统
>
> 三个独立 agent 通过 A2A 协议协作，用自然语言查询 A 股公告、公司信息与市场数据。
> **所有答案可溯源**到公告 ID / 股票代码 / 数据来源。

---

## 目录

- [1. 项目亮点](#1-项目亮点)
- [2. 架构设计](#2-架构设计)
- [3. 环境要求](#3-环境要求)
- [4. 快速开始](#4-快速开始)
- [5. 核心模块](#5-核心模块)
- [6. 数据说明](#6-数据说明)
- [7. 日志系统](#7-日志系统)
- [8. 数据库设计](#8-数据库设计)
- [9. 评估体系](#9-评估体系)
- [10. 实测踩到的坑](#10-实测踩到的坑)
- [11. 成本](#11-成本)
- [12. 常见问题](#12-常见问题)
- [13. 路线图](#13-路线图)

---

## 1. 项目亮点

| 亮点 | 说明 |
|---|---|
| **真·异构多智能体** | 三个 agent 是三个独立进程（5005/5006/5007），只通过 A2A 协议通信 |
| **官方 A2A SDK** | 用 `a2a-sdk 1.3.0`（protobuf 版），不是第三方 `python-a2a` |
| **动态能力发现** | Orchestrator 从 Agent Card 拉取能力，**新增 agent 无需改代码** |
| **两阶段检索** | 向量召回 Top 50 → rerank 精排，相关度从 0.61 提升到 0.86 |
| **真实数据可验证** | 13.4 万条公告，每条带 announcementId 可上巨潮核对 |
| **完整评估体系** | 18 条用例全部通过，TSR 100%，P50 延迟 6.79s |

**演示问题**：

```
> 最近有哪些公司回购股份
> 浦发银行最近有什么公告
> 贵州燃气发了多少公告
> 今天大盘怎么样，浦发银行是什么公司
```

---

## 2. 架构设计

### 2.1 三层架构

```
┌──────────────────────────────────────────────────────────┐
│  用户界面层                                                │
│  · CLI:  python -m a2a_server.orchestrator "问题"         │
│  · Web:  python -m a2a_server.web   → localhost:8085     │
└────────────────────────┬─────────────────────────────────┘
                         │
┌────────────────────────▼─────────────────────────────────┐
│  编排层 Orchestrator (a2a_server/orchestrator.py)         │
│  1. 动态发现 agent（拉取 Agent Card）                      │
│  2. LLM 意图识别（qwen3.8-27b）                            │
│  3. 并发调度（asyncio.gather）                             │
│  4. LLM 结果汇总                                          │
└───┬──────────────────┬──────────────────┬────────────────┘
    │ A2A JSON-RPC     │ A2A JSON-RPC     │ A2A JSON-RPC
    │ (SendMessage)    │                  │
┌───▼──────────┐ ┌─────▼────────┐ ┌───────▼──────────┐
│ 公告检索      │ │ 公司画像      │ │ 市场数据          │
│ :5005        │ │ :5006        │ │ :5007            │
│              │ │              │ │                  │
│ 向量召回 50   │ │ SQL 查询      │ │ MySQL 聚合        │
│  → rerank 10 │ │              │ │                  │
└───┬──────────┘ └─────┬────────┘ └───────┬──────────┘
    │                  │                  │
┌───▼──────┐    ┌──────▼─────┐    ┌───────▼──────┐
│ Milvus   │    │  MySQL     │    │  MySQL       │
│ 13.4万向量│    │  companies │    │ quotes       │
│          │    │announcements│   │ fx/gold      │
└──────────┘    └────────────┘    └──────────────┘
```

### 2.2 为什么用 A2A 而不是进程内函数调用

三个 agent 的**依赖、数据源、更新节奏完全不同**：

| Agent | 依赖 | 数据源 | 独立更新的理由 |
|---|---|---|---|
| 公告检索 | Milvus + Embedding API | 巨潮资讯 | 向量库重建时不影响其他 agent |
| 公司画像 | 仅 MySQL | 上交所 | 数据表结构变化时独立发布 |
| 市场数据 | 仅 MySQL | 新浪/货币网/黄金所 | 行情采集频率不同 |

**A2A 的价值**：Orchestrator 不需要知道 agent 内部实现，只需读 Agent Card 就能调用。新增一个「研报分析 agent」时，Orchestrator 一行代码都不用改。

### 2.3 A2A 协议版本说明

本项目使用 **A2A v1.0**（`a2a-sdk 1.3.0` protobuf 版），与网上大量 v0.3 教程不同：

| 项目 | v0.3（旧） | **v1.0（本项目）** |
|---|---|---|
| JSON-RPC 方法名 | `message/send` | **`SendMessage`**（gRPC 风格） |
| 字段命名 | `message_id` | **`messageId`**（camelCase） |
| role 枚举 | `user` | **`ROLE_USER`** |
| 版本头 | 无 | **必须带 `A2A-Version: 1.0`** |
| AgentCard 字段 | `url` | **`supported_interfaces`** |
| HTTP 层 | `A2AStarletteApplication` | **`create_*_routes` + `add_a2a_routes_to_fastapi`** |
| 同步响应 | Task + TaskUpdater | **直接 enqueue 一个 `Message`** |

---

## 3. 环境要求

### 3.1 必需

| 组件 | 版本 | 本项目实测 | 说明 |
|---|---|---|---|
| **Python** | 3.10+ | **3.12.14** | 推荐 3.12，3.14 部分依赖编译失败 |
| **MySQL** | 5.7+/8.0+ | **8.0.12** | 存公告、公司、行情 |
| **Milvus** | 2.4+ | **2.4.4** | 向量检索 |
| **Docker** | 20.10+ | **29.6.2** | 运行 Milvus（etcd + MinIO + Milvus）|

**Milvus 内存要求**：约 8 GB 可用内存。

### 3.2 Python 依赖

```powershell
pip install requests pymysql pymilvus fastapi uvicorn "a2a-sdk[http-server]"
```

| 包 | 用途 |
|---|---|
| `requests` | HTTP 请求（数据采集 + LLM 调用）|
| `PyMySQL` | MySQL 驱动 |
| `pymilvus` | Milvus 客户端 |
| `fastapi` + `uvicorn` | A2A 服务 + Web 界面 |
| `a2a-sdk[http-server]` | A2A 协议实现 |

**不需要** `openai` SDK（直接用 requests 调 OpenAI 兼容接口）。

### 3.3 磁盘

| 项目 | 大小 |
|---|---|
| 代码 + 虚拟环境 | ~250 MB |
| `data/raw/`（97 个日文件）| **57 MB** |
| MySQL 数据（13.4 万公告）| ~150 MB |
| Milvus 数据（13.4 万 × 1024 维）| **~550 MB** |
| Docker 镜像 | ~3 GB |
| **合计** | **约 4 GB** |

---

## 4. 快速开始

### 4.1 环境变量

```powershell
# LLM / Embedding（必需）
$env:QIANWEN_API_KEY = [Environment]::GetEnvironmentVariable('QIANWEN_API_KEY','User')
$env:QIANWEN_BASE_URL = [Environment]::GetEnvironmentVariable('QIANWEN_BASE_URL','User')

# 可选：切换向量化模型
$env:EMBEDDING_MODEL = 'qwen3.7-text-embedding-flash'   # 默认，便宜
# $env:EMBEDDING_MODEL = 'qwen3.7-text-embedding'       # 标准版，更准

# 可选：调日志级别
$env:LOG_LEVEL = 'INFO'
```

> **MySQL 需要设置密码** —— 通过 `MYSQL_PASSWORD` 环境变量传入，代码中不保留任何凭据。
> Milvus 默认连 `127.0.0.1:19530`，无需额外配置。

### 4.2 启动 Milvus

```powershell
cd C:\market-agent\docker
docker compose up -d
# 等待约 60 秒
docker compose ps    # 三个容器都应为 healthy
```

### 4.3 一键启动

```powershell
cd C:\market-agent

# 环境检查
.\.venv\Scripts\python.exe scripts\start_all.py --check

# 启动三个 agent
.\.venv\Scripts\python.exe scripts\start_all.py
```

输出：

```
[OK] MySQL 8.0.12  133,957 条公告
[OK] Milvus 集合 118,468
[OK] LLM 模型 qwen3.8-27b
[OK] Embedding 模型 qwen3.7-text-embedding-flash

[启动] 公告检索 (端口 5005)
[启动] 公司画像 (端口 5006)
[启动] 市场数据 (端口 5007)
```

### 4.4 使用

```powershell
# CLI 查询
.\.venv\Scripts\python.exe -m a2a_server.orchestrator "浦发银行最近有什么公告"

# Web 界面
.\.venv\Scripts\python.exe -m a2a_server.web
# 浏览器打开 http://127.0.0.1:8085
```

### 4.5 完整数据准备（首次）

```powershell
# 1. 建库建表
.\.venv\Scripts\python.exe utils\db.py init

# 2. 采集数据（共约 25 分钟）
.\.venv\Scripts\python.exe scripts\fetch_announcements.py   # 公告（22 分钟）
.\.venv\Scripts\python.exe scripts\fetch_companies.py       # 公司（5 秒）
.\.venv\Scripts\python.exe scripts\fetch_fx_gold.py         # 汇率+黄金（3 秒）
.\.venv\Scripts\python.exe scripts\fetch_quotes.py          # 行情（2 秒）

# 3. 导入 MySQL
.\.venv\Scripts\python.exe scripts\load_to_mysql.py all

# 4. 创建 Milvus 集合
.\.venv\Scripts\python.exe utils\milvus_client.py init

# 5. 向量化（11.8 万条，实测 57 分钟 / 约 0.25 元）
.\.venv\Scripts\python.exe scripts\embed_announcements.py --level L1,L2
```

---

## 5. 核心模块

### 5.0 Web 界面

```powershell
.\.venv\Scripts\python.exe -m a2a_server.web
# 浏览器打开 http://127.0.0.1:8085
```

**界面截图**：

![欢迎页](docs/ui-shot.png)

![对话页](docs/ui-chat.png)

**布局**：左侧栏 + 主对话区

左侧栏（三段）：
- **数据规模** — 实时从 MySQL/Milvus 读取（缓存 5 分钟）
- **Agent 状态** — 三个 agent 的在线指示灯，绿点=在线
- **试试这样问** — 5 个预设问题，点击即问

主对话区：
- 欢迎屏：渐变标题 + 三张能力卡片
- 消息流：用户消息右侧蓝色气泡，机器人左侧带头像与 agent 标签
- 输入区：多行输入框，`Enter` 发送 / `Shift+Enter` 换行

**渲染细节**（都是踩过才知道要处理的）：

| 情况 | 处理 |
|---|---|
| `【结构化数据】` 后的原始 JSON | 折叠成「查看结构化数据」，默认收起 |
| `[2026-10-30] 公告标题` 缩进行 | 渲染成带日期标签的列表，日期与标题分列 |
| `【贵州燃气（600903）】` 整行 | 渲染成加粗标题，不加左侧竖线 |
| Markdown 表格 | 独立容器横向滚动，数字列等宽对齐 |
| 耗时显示 | 用后端返回的 `elapsed`，不用前端 `Date.now()` |

> **最后一条的原因**：无头浏览器截图时前端时钟被虚拟化，`Date.now()` 差值是 0，
> 界面上会显示"耗时 0.0s"。改成读后端字段后既准确又不依赖客户端时钟。

**其它特性**：
- 深色主题，无任何外部 CDN 依赖（字体用系统字体），完全离线可用
- 支持 `http://127.0.0.1:8085/?q=你的问题` 直接提问（便于分享链接和自动化截图）
- 加载动画带分阶段文案（识别意图 → 调度 agent → 汇总结果）与实时计时

**接口**：

| 路径 | 方法 | 说明 |
|---|---|---|
| `/` | GET | 聊天界面（支持 `?q=` 预填提问）|
| `/api/stats` | GET | 数据规模（公告数/公司数/向量数，缓存 5 分钟）|
| `/api/status` | GET | 检查 agent 在线状态 |
| `/api/ask` | POST | 核心问答（body: `{"query": "..."}`）|

**实测响应**（查"贵州燃气发了多少公告"）：

```
状态: 200   耗时: 10.7s   意图: ['company']

【贵州燃气（600903）】
公司全称：贵州燃气集团股份有限公司
所属行业：电力、热力、燃气及水生产和供应业
注册地区：贵州省
上市日期：2017-11-07

公告统计（2026-08-04 ~ 2026-10-30）：
  总计 108 条
  核心 L1：40 条
  重要 L2：60 条
  模板 L3：8 条
```

### 5.1 三个 A2A Agent

| Agent | 端口 | 文件 | 能力 |
|---|---|---|---|
| **公告检索** | 5005 | `a2a_server/announcement_agent.py` | 语义检索 + 公司/日期/层级过滤 |
| **公司画像** | 5006 | `a2a_server/company_agent.py` | 基本信息 + 公告统计 + 同业对比 |
| **市场数据** | 5007 | `a2a_server/market_agent.py` | 指数 / 汇率 / 黄金 |

每个 agent 暴露：
- `/.well-known/agent-card.json` — 能力声明
- `/` — JSON-RPC 端点（`SendMessage` 方法）
- `/health` — 健康检查

### 5.2 两阶段检索

```
用户 query
    ↓
[阶段 1] Embedding → Milvus 向量召回 Top 50
    ↓
[阶段 2] qwen3.7-text-rerank 精排 → Top 10
    ↓
返回带公告 ID 的结果
```

**效果对比**（实测）：

| 方式 | "回购股份" 查询的平均相似度 |
|---|---|
| 纯向量检索 | 0.61 |
| **向量 + rerank** | **0.86** |

rerank 的分数区分度也远高于余弦相似度：

```
query: "回购股份"
  "关于回购公司股份的公告"  → 0.6276    ✅
  "关于2026年第三季度报告..."  → 0.2253
  "关于董事会换届选举的公告"   → 0.1560    ← 差 4 倍
```

**全量 118,468 条入库后的实测**（2026-11-02 21:45）：

| 查询 | 余弦 Top1 | rerank Top1 | rerank 分数 |
|---|---|---|---|
| 最近有哪些公司回购股份 | 0.7955 | 关于回购公司A股股份的进展公告 | 0.4528 |
| 重大资产重组 | 0.8506 | 董事会关于本次交易构成重大资产重组但不构成重组上市的说明 | 0.6966 |
| 股东减持 | 0.8210 | 关于股东减持股份的预披露公告 | 0.6510 |

注意余弦相似度的 Top1 和 Top3 往往只差 0.002（0.7955 / 0.7936 / 0.7936），
几乎无法区分；rerank 分数在同类与异类之间能拉开数倍差距，这才是可排序的信号。

### 5.3 动态能力发现

Orchestrator 启动时拉取所有 Agent Card，用 skills 描述生成 LLM 路由提示：

```python
# orchestrator.py
def build_routing_prompt(self) -> str:
    lines = []
    for name, card in self._cards.items():
        skills = "；".join(f"{s['name']}（{s['description']}）" for s in card["skills"][:3])
        lines.append(f"- {name}: {card['name']} —— {skills}")
    return "\n".join(lines)
```

**新增 agent 时**：只需注册端口 + 启动服务，Orchestrator **零改动**。

#### 路由提示里的「宁少勿多」原则

动态生成的能力清单只是"有哪些 agent"，还需要告诉 LLM **什么时候不该多选**。

```python
## 多选原则（重要，宁少勿多）
只有当用户一句话里存在两个及以上彼此独立的问题时，才选多个 agent。
判断方法：能不能把这句话拆成两个各自独立、单独问也成立的句子？

只选一个的例子（看起来涉及多个领域，但实质是一个问题）：
  "贵州燃气发了多少公告"  → 只选 company（统计能力直接给数）
  "688728 最近有什么公告" → 只选 announcement（代码识别在内部完成）

核心判断：某个 agent 能独立完成整个诉求时，就只选它，不要为了"更全面"而叠加。
```

没有这段约束时，LLM 会把"查某公司公告"这类问题拆成 `announcement + company`，
多调一个 agent、多一次 LLM 汇总，既慢又容易在汇总环节引入噪声。

### 5.4 公司名识别

用户输入"贵州燃气发了多少公告"，需要识别出 `600903 贵州燃气`。

**做法**：连续中文段做滑窗子串匹配（从长到短），先精确匹配、再前缀匹配。

```python
segments = re.findall(r"[\u4e00-\u9fff]+", query)   # ["贵州燃气发了多少公告"]
candidates = {seg[i:i+L] for L in range(8,1,-1) for i in ...}  # 滑窗
# → 精确匹配 "贵州燃气" 命中
```

> **踩坑记录**：最初用 `re.sub` 去掉疑问词后整体匹配，得到"贵州燃气发多少"，匹配失败。滑窗方案才正确。

---

## 6. 数据说明

### 6.1 数据源（全部免费、无需 Key）

| 数据源 | 内容 | 量 | 接口 |
|---|---|---|---|
| **巨潮资讯** | 上市公司公告 | **133,957 条** | `cninfo.com.cn/new/hisAnnouncement/query` |
| **上交所** | 公司基本信息 | **5,619 家** | `query.sse.com.cn` |
| **中国货币网** | 人民币汇率中间价 | 25 组 | `chinamoney.com.cn/.../ccpr.json` |
| **上金所** | 黄金日行情 | 2,392 条 | `sge.com.cn/graph/Dailyhq?instid=Au99.99` |
| **新浪财经** | 实时行情 | 5 个指数 | `hq.sinajs.cn` |

### 6.2 公告分层

| 层级 | 判定 | 数量 | 处理 |
|---|---|---|---|
| **L1 核心** | 回购、重组、减持、业绩预告、诉讼、分红… | 16,282 | PDF + 向量化 |
| **L2 重要** | 其余非模板公告 | 102,186 | 向量化 |
| **L3 模板** | 监事会决议、股东大会通知、持续督导… | 15,489 | 只存元数据 |

规则在 `utils/config.py`：
- `CORE_KEYWORDS` 定义 L1
- `TEMPLATE_PATTERNS` 定义 L3

### 6.3 数据质量（实测）

| 检查项 | 结果 |
|---|---|
| 时间覆盖 | 2026-08-01 ~ 2026-10-31（74 个有数据日）|
| 板块分布 | 深市 68,604 / 沪市 57,603 / 北交所 7,750 |
| 标题为空 | **0** |
| PDF 链接缺失 | **0** |
| 导入失败 | **0** |

---

## 7. 日志系统

### 7.1 设计

| 特性 | 实现 |
|---|---|
| 双输出 | 控制台带颜色 + 文件完整记录 |
| 按天切分 | `app.log` 保留 30 天 |
| 错误分离 | `error.log` 单文件 10 MB，保留 10 份 |
| 任务上下文 | `with log_context(task="xxx")` |
| 第三方降噪 | urllib3/requests/pymilvus 降到 WARNING |

### 7.2 格式

**控制台**（紧凑）：
```
21:13:26 INF __main__    批次  47  已处理  9,400/118,468  插入 200  6.9s  29.0 条/秒
21:13:18 WRN embedding   向量化限流 429，等待 5s (1/3)
21:12:19 ERR embedding   向量化 HTTP 400: Free quota exhausted
```

**文件**（完整，便于排查）：
```
2026-10-31 21:13:26 | INFO | scripts.embed_announcements:245 | run | 批次 47 ...
```

### 7.3 级别控制

```powershell
$env:LOG_LEVEL='DEBUG'    # 打印每次 HTTP 请求
$env:LOG_LEVEL='INFO'     # 默认
$env:LOG_LEVEL='WARNING'  # 只看警告和错误
```

---

## 8. 数据库设计

### 8.1 MySQL 表

| 表 | 行数 | 用途 |
|---|---|---|
| `announcements` | 133,957 | 公告主表 |
| `companies` | 5,619 | 公司基本信息 |
| `quotes` | 5 | 行情快照 |
| `fx_rates` | 25 | 汇率中间价 |
| `gold_prices` | 2,392 | 黄金行情 |
| `etl_jobs` | 0 | 采集日志 |

核心字段：

```sql
-- announcements
announcement_id VARCHAR(32) PRIMARY KEY   -- 巨潮 ID，可核对
sec_code        VARCHAR(16)               -- 股票代码
title           VARCHAR(1024)
pub_date        DATE
plate           VARCHAR(8)                -- sz/sh/bj
level           CHAR(2)                   -- L1/L2/L3
is_embedded     TINYINT                   -- 是否已向量化
```

### 8.2 Milvus 集合

```python
{
  "id": INT64 (auto),
  "announcement_id": VARCHAR(32),
  "sec_code": VARCHAR(16),
  "sec_name": VARCHAR(128),
  "title": VARCHAR(1024),
  "level": VARCHAR(4),
  "pub_date": VARCHAR(16),
  "embedding": FLOAT_VECTOR(1024),   # HNSW + COSINE
}
```

**索引**：`embedding` 用 HNSW（M=16, efConstruction=200），标量字段用 INVERTED。

---

## 9. 评估体系

### 9.1 指标

| 指标 | 说明 |
|---|---|
| **TSR** | 任务成功率：答案是否含预期关键信息 |
| **Intent Accuracy** | 意图识别准确率（选中的 agent 是否与预期一致）|
| **Keyword Coverage** | 关键信息覆盖度 |
| **Forbidden Pass** | 无幻觉/禁词比例 |
| **Latency** | P50/P95/max |
| **Error Rate** | 错误率 |

### 9.2 实测结果

**2026-11-02 21:53 全量评估（18 条用例，向量化全量 118,468 条后）**

| 指标 | 结果 |
|---|---|
| 用例总数 | 18 |
| 成功 | **18** |
| **TSR** | **100.0%** |
| **意图准确率** | **100.0%** |
| 关键词覆盖 | 100.0% |
| 无幻觉/禁词 | 100.0% |
| 错误率 | 0.0% |
| 延迟 avg / P50 / P95 / max | **9.73s / 6.79s / 48.91s / 48.91s** |
| 总耗时 | 175.2s |

**按类别**：

| 类别 | 通过率 | 说明 |
|---|---|---|
| `single` 单意图 | **9/9 100%** | 公告检索、公司查询、市场数据各 3 条 |
| `multi` 多意图 | **4/4 100%** | 双意图并发、公司过滤 |
| `boundary` 边界 | **3/3 100%** | 指代不明、超范围、不存在的公司 |
| `safety` 安全 | **2/2 100%** | 无推诿话术、答案可溯源 |

**延迟说明**：P95 的 48.91s 来自 `multi_announcement_market`（公告+市场双 agent
并发 + LLM 汇总），单 agent 查询通常在 2.6-8 秒。

**关于意图识别的「宁少勿多」原则**：早期版本的 prompt 只写了"复合问题选多个"，
导致 LLM 过度拆分——"贵州燃气发了多少公告"被拆成 `company + announcement`，
"688728 最近有什么公告"被拆成 `announcement + company`，两条用例因此失败。

修复方式是在路由 prompt 里加入显式的判断方法（**能否拆成两个各自独立、单独问也
成立的句子**）和一组反例，明确「某个 agent 能独立完成整个诉求时只选它」。
修复后 18/18 通过，平均延迟从 11.42s 降到 9.73s（少调一个 agent + 少一次汇总）。

### 9.3 运行

```powershell
# 全量评估（18 条用例）
.\.venv\Scripts\python.exe scripts\evaluate.py

# 只跑单意图
.\.venv\Scripts\python.exe scripts\evaluate.py --category single

# 跑单条
.\.venv\Scripts\python.exe scripts\evaluate.py --case company_by_code
```

结果写入 `tests/eval_results.json`，**按用例 ID 累加合并**，
分次跑不同类别不会互相覆盖。

### 9.4 用例设计原则

**可验证**：每条用例的期望值都能从数据库直接核对。

例如 `company_announcement_count`：
- 查询："贵州燃气发了多少公告"
- 期望关键词：`公告`
- 可核对：`SELECT COUNT(*) FROM announcements WHERE sec_code='600903'` → 108

**四类覆盖**：

| 类别 | 数量 | 测什么 |
|---|---|---|
| `single` | 9 | 单 agent 调用是否准确 |
| `multi` | 4 | 多 agent 并发调度、公司过滤是否生效 |
| `boundary` | 3 | 信息缺失时是否追问、超范围是否拒答 |
| `safety` | 2 | 有无推诿话术、答案是否带公告 ID |

---

## 10. 实测踩到的坑

### A2A 协议层

**坑 1：方法名是 `SendMessage`，不是 `message/send`**

```python
# v0.3 写法（会返回 Method not found）
{"method": "message/send", ...}

# v1.0 正确写法
{"method": "SendMessage", ...}
```

**坑 2：必须带 `A2A-Version: 1.0` 头**

```python
# 不带会报 "A2A version '0.3' is not supported"
requests.post(url, json=payload,
              headers={"A2A-Version": "1.0"})
```

**坑 3：同步响应不能直接用 TaskUpdater**

```
错误：await updater.start_work()
     → "Agent should enqueue Task before TaskStatusUpdateEvent event"

正确：直接 enqueue 一个 Message 对象
     await event_queue.enqueue_event(Message(role=ROLE_AGENT, parts=[...]))
```

**坑 4：`DefaultRequestHandler` 需要 `agent_card` 参数**

```python
DefaultRequestHandler(
    agent_executor=executor,
    task_store=InMemoryTaskStore(),
    agent_card=card,     # 必须传
)
```

### 数据采集层

**坑 5：巨潮分页第 150 页回绕到第 1 页** ⚠️ 最重要

单日实际有 5,469 条公告，直接翻页最多只能拿 3,000 条（漏 45%）。

**解法**：按 `plate` 板块拆分（`sz`/`sh`/`bj`），每个板块单独翻页。

**坑 6：pageSize 上限固定 30**

传 100/300/500 都只返回 30 条。

**坑 7：上金所必须带 `instid` 参数**

```python
# 错误：https://www.sge.com.cn/graph/Dailyhq
# 返回 2,392 条记录，但所有数值都是 0

# 正确：https://www.sge.com.cn/graph/Dailyhq?instid=Au99.99
```

**坑 8：东方财富对本机 IP 断连**

补全浏览器头后仍失败，改用新浪财经（注意 **GBK 编码**）。

### 数据存储层

**坑 9：跨模型向量不兼容**

```
同一文本在两个模型下的向量余弦相似度:  0.078
同模型内正确匹配的相似度:            0.886
```

**混用模型会导致检索完全失效**，必须全库用同一个模型。

**坑 10：`Decimal` / `date` 不能直接 JSON 序列化**

MySQL 的 DECIMAL 返回 `Decimal`，DATE 返回 `date`。A2A 返回结果前必须转换。

**坑 11：LIKE 的 `%` 与参数化冲突**

```python
# 错误：pymysql 会把 %美 当成格式符
"WHERE pair LIKE '%美元%' AND x = %s"

# 正确：字面量 % 要写成 %%
"WHERE pair LIKE '%%美元%%' AND x = %s"
```

### 文本处理层

**坑 12：贪婪正则切不出公司名**

```python
# 错误："浦发银行最近有什么公告" → 整体匹配成 "浦发银行最近有什"
re.findall(r"[\u4e00-\u9fff]{2,8}", query)

# 正确：连续中文段做滑窗子串
segments = re.findall(r"[\u4e00-\u9fff]+", query)
candidates = {seg[i:i+L] for L in range(8,1,-1) for i in range(...)}
```

### API 额度层

**坑 13：额度耗尽会连续失败刷日志**

任务跑了 490 批，每批失败都重试后才跳过，产生 **20 万字节**错误日志。

**修复**：识别 `Free quota exhausted` 后抛专用异常，立即停止整个任务。

### Web 集成层

**坑 14：FastAPI 里不能用 `asyncio.run()`**

```
RuntimeError: asyncio.run() cannot be called from a running event loop
RuntimeWarning: coroutine 'Orchestrator.dispatch_parallel' was never awaited
```

**原因**：Web 请求处理函数本身已在事件循环中，而 `orchestrator.run()`
内部用 `asyncio.run()` 驱动异步调度，两者冲突。

**修复**：拆成两个入口

```python
# 异步版本（Web 用）
async def arun(self, query): ...
    results = await self.dispatch_parallel(intents, query)

# 同步版本（CLI 用）
def run(self, query):
    return asyncio.run(self.arun(query))
```

Web 层改为 `await _orch.arun(query)`。

---

## 11. 成本

### 11.1 向量化（一次性）

| 项目 | 数值 |
|---|---|
| 待向量化 | 118,468 条 |
| 实际完成 | 118,468 条（失败 0 条）|
| 实测耗时 | 57.4 分钟（1,663 条/分钟）|
| 总字符数 | 2,994,629 |
| Token 估算（1 token ≈ 1.5 字符）| **约 200 万 tokens** |
| flash 单价 | 0.000125 元/千token |
| **flash 成本** | **0.25 元** |
| 标准版成本 | 1.00 元 |

免费额度：每个模型 100 万 tokens（90 天）。

### 11.2 日常运行

| 操作 | 成本 |
|---|---|
| 一次查询（意图识别 + 汇总）| 约 0.01-0.03 元 |
| 数据采集 | **0 元**（全部免费公开接口）|

**整个项目 API 成本不到 5 元。**

---

## 12. 常见问题

### Q1: `ModuleNotFoundError: No module named 'pip'`

`uv venv` 创建的虚拟环境不含 pip：

```powershell
.\.venv\Scripts\python.exe -m ensurepip --upgrade
```

### Q2: Milvus 连接失败

```powershell
cd docker
docker compose ps        # 检查三个容器是否 healthy
docker compose up -d     # 重新启动
```

内存不足时 Milvus 会启动失败，需约 8 GB 可用内存。

### Q3: Agent Card 能访问但查询失败

检查 agent 日志：

```powershell
Get-Content logs\agent_5005.err.log -Tail 30
```

常见原因：
- Embedding 额度耗尽
- Milvus 集合为空（还没向量化）

### Q4: 查询返回"未识别出公司"

```powershell
.\.venv\Scripts\python.exe -c "import sys; sys.path.insert(0,'.'); from a2a_server.company_agent import CompanyExecutor; print(CompanyExecutor()._find_company('贵州燃气'))"
```

如果返回 None，说明公司名不在 `companies` 表中。

### Q5: 如何完全重来

```powershell
# 1. 删数据
Remove-Item data\raw\* -Force
Remove-Item data\checkpoint\* -Force

# 2. 重建数据库
.\.venv\Scripts\python.exe utils\db.py drop
.\.venv\Scripts\python.exe utils\db.py init

# 3. 重建 Milvus
.\.venv\Scripts\python.exe utils\milvus_client.py drop
.\.venv\Scripts\python.exe utils\milvus_client.py init

# 4. 重新采集
.\.venv\Scripts\python.exe scripts\fetch_announcements.py
.\.venv\Scripts\python.exe scripts\load_to_mysql.py all
.\.venv\Scripts\python.exe scripts\embed_announcements.py
```

### Q6: 换 embedding 模型后检索变差

**必须清空重建**（跨模型向量不兼容）：

```powershell
.\.venv\Scripts\python.exe -c "import sys; sys.path.insert(0,'.'); from utils.db import connect; c=connect(); cur=c.cursor(); cur.execute('UPDATE announcements SET is_embedded=0'); c.commit()"
.\.venv\Scripts\python.exe utils\milvus_client.py init --rebuild
.\.venv\Scripts\python.exe scripts\embed_announcements.py
```

---

## 13. 路线图

| 阶段 | 内容 | 状态 |
|---|---|---|
| **W1** | 数据层：5 数据源采集 + MySQL 入库 + 日志 | ✅ 完成 |
| **W2** | 向量化：Milvus + 两阶段检索（embedding + rerank）| ✅ 完成 |
| **W3** | A2A 层：三个 agent（Agent Card + SendMessage）| ✅ 完成 |
| **W4** | 编排层：意图识别 + 并发调度 + 结果汇总 | ✅ 完成 |
| **W5** | 质量关：评估集 + Web 界面 + 文档 | ✅ 完成 |

**当前状态：五个阶段全部完成。** 全量 118,468 条向量入库，
18 条评估用例 18/18 通过，TSR 100%，Web 界面与 CLI 均已实测可用。

### 后续可扩展

- [ ] 接入 A2A TCK 做协议合规性测试
- [ ] 增加「研报分析 agent」（演示动态发现能力）
- [ ] 公告 PDF 全文解析（现在的向量只有标题）
- [ ] 对话历史（多轮上下文）

---

## 项目结构

```
market-agent/
├── README.md
├── a2a_server/                  # A2A 服务层
│   ├── base.py                  # AgentCard + BaseExecutor + 服务启动
│   ├── announcement_agent.py    # 公告检索 agent (:5005)
│   ├── company_agent.py         # 公司画像 agent (:5006)
│   ├── market_agent.py          # 市场数据 agent (:5007)
│   ├── orchestrator.py          # 编排层（动态发现 + 意图 + 调度 + 汇总）
│   └── web.py                   # Web 界面 (:8085)
├── scripts/
│   ├── fetch_announcements.py   # 采集公告（板块拆分 + 断点续传）
│   ├── fetch_companies.py       # 采集公司信息
│   ├── fetch_fx_gold.py         # 采集汇率 + 黄金
│   ├── fetch_quotes.py          # 采集行情
│   ├── load_to_mysql.py         # 导入 MySQL
│   ├── embed_announcements.py   # 向量化（断点续传 + 额度保护）
│   ├── search_test.py           # 检索验证
│   ├── compare_embedding.py     # 向量化方案对比实验
│   ├── evaluate.py              # 评估框架
│   ├── test_a2a_client.py       # A2A 客户端测试
│   ├── run_agents.py            # 启动所有 agent
│   └── start_all.py             # 一键启动（环境检查 + 启动）
├── utils/
│   ├── config.py                # 全局配置
│   ├── db.py                    # MySQL 连接与建表
│   ├── logger.py                # 日志系统
│   ├── llm.py                   # qwen3.8-27b 客户端
│   ├── embedding.py             # 向量化（qwen / BGE-M3 双后端）
│   ├── milvus_client.py         # Milvus 集合管理与检索
│   └── rerank.py                # qwen3.7-text-rerank 精排
├── sql/schema.sql               # 6 张表 + 2 个视图
├── tests/
│   ├── eval_dataset.json        # 18 条评估用例
│   └── eval_results.json        # 评估结果
├── docker/docker-compose.yml    # Milvus 部署
├── data/                        # 原始数据 + 断点
└── logs/                        # 日志
```

**代码量**：约 7,000 行（Python + SQL）

---

## 许可证

MIT
