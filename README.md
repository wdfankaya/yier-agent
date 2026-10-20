<div align="center">

# 一二智能客服

**连接对话、知识与业务工具的电商客服 Agent**

商品咨询 · 订单查询 · 物流跟踪 · 退换货 · 投诉处理

![Python](https://img.shields.io/badge/Python-3.11%2B-3776AB?style=flat-square&logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-SSE-009688?style=flat-square&logo=fastapi&logoColor=white)
![PostgreSQL](https://img.shields.io/badge/PostgreSQL-pgvector-4169E1?style=flat-square&logo=postgresql&logoColor=white)
![Redis](https://img.shields.io/badge/Redis-Cache%20%26%20Rate%20Limit-DC382D?style=flat-square&logo=redis&logoColor=white)

[功能概览](#功能概览) · [关键设计](#关键设计) · [快速开始](#快速开始) · [HTTP 服务](#http-服务) · [测试与评估](#测试与评估)

</div>

---

一二智能客服基于 ReAct 组织多轮对话，通过业务工具查询订单和物流，通过 RAG 检索政策与 FAQ，并使用长短期记忆保留对话信息。项目提供命令行入口和 FastAPI 异步服务，支持 SSE 事件输出、会话持久化、缓存回源、限流重试及退款操作确认。

新增可选 **LangGraph 可恢复工作流、中文 BM25 + RRF 混合检索、可查询请求追踪和固定100题对照实验**。使用 `python scripts/serve.py` 启动新模式；配置、验收结果和边界见 [增强实现与验收](docs/agent-upgrade-20261019.md)。原模式及历史实验记录仍保留。

## 功能概览

| 能力 | 实现方式 |
| :--- | :--- |
| **对话编排** | ReAct 工具调用循环；可选售前、售后、投诉意图路由，分别配置提示词与工具白名单 |
| **工具与技能** | 统一接入本地 Function Calling 与 MCP 工具；按需加载退货、订单跟踪、商品推荐 Skills |
| **知识检索** | numpy / Chroma / pgvector 后端；向量与全文混合召回，可选 BGE 重排及来源返回 |
| **上下文与记忆** | 调用前预算、完整轮次压缩、可核验摘要、带来源/有效期的事实、相关性召回及并发增量合并 |
| **异步服务** | FastAPI + Asyncio + SSE，输出路由、工具调用、确认请求及最终回复等事件 |
| **状态与稳定性** | PostgreSQL 持久化、Redis 冷读缓存、Lua 令牌桶限流，以及模型和工具的超时控制 |
| **操作确认** | 工具执行层拦截退款请求，经用户批准后执行；拒绝后阻止该会话对同一订单再次退款 |
| **效果验证** | 隔离评估环境、调用轨迹、规则指标与 LLM-as-Judge；保留对照实验及压测记录 |

## 关键设计

| 设计 | 说明 |
| :--- | :--- |
| 入口复用 | CLI 与 HTTP 共用 Agent 实现，HTTP 通过异步调用与事件回调输出 SSE |
| 状态分层 | 活跃 Agent 保存在进程内；数据库模式下 PostgreSQL 持久化，Redis 加速冷读 |
| 调用隔离 | Memory 与 Skill 按 Agent 注入，通过 ContextVar 绑定工具调用上下文 |
| 检索融合 | 全文检索结果优先，向量结果去重补充；重排失败时保留召回顺序 |
| 异常处理 | Redis 故障时回源数据库或回退内存限流；模型调用超时重试，工具执行设置超时 |

## 快速开始

需要 **Python 3.11+**，以及可用的对话模型和 Embedding 接口。基础模式使用 JSON 持久化与 numpy 检索，无需先部署 PostgreSQL、Redis。

### 1. 获取项目并安装依赖

```bash
git clone https://github.com/wdfankaya/yier-agent.git
cd yier-agent
```

<details open>
<summary><strong>Windows · PowerShell</strong></summary>

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
Copy-Item .env.example .env
```

后续命令中的 `python` 使用 `.\.venv\Scripts\python.exe`，无需激活虚拟环境。

</details>

<details>
<summary><strong>macOS / Linux · Bash</strong></summary>

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
cp .env.example .env
```

</details>

### 2. 配置模型接口

编辑 `.env`，填写与你的模型服务匹配的配置：

```dotenv
OPENAI_API_KEY=your_api_key
OPENAI_BASE_URL=https://api.openai.com/v1
MODEL_NAME=gpt-4o-mini
EMBEDDING_MODEL=text-embedding-3-small
RAG_BACKEND=numpy
DB_ENABLED=false
REDIS_ENABLED=false
RERANK_ENABLED=false
```

以上为基础配置示例。若对话服务不支持 Embedding，另行填写 `EMBEDDING_API_KEY` 和 `EMBEDDING_BASE_URL`。重排需要支持相应接口的服务，配置好后再开启。

### 3. 构建索引并开始对话

```bash
python -m app.scripts.build_kb_index
python main.py
```

可以尝试这些问题：

> “订单 ORD-20240115-001 的物流到哪里了？”
>
> “七天无理由退货需要满足什么条件？”
>
> “我想取消订单 ORD-20240120-002，申请退款。”

CLI 支持 `skills`、`memory`、`reset`、`quit` 命令。退款演示使用示例订单数据，执行前由确认机制拦截。

## HTTP 服务

完成基础配置和索引构建后，为 HTTP 接口配置访问令牌。先生成一个随机令牌：

```bash
python -c "import secrets; print(secrets.token_urlsafe(32))"
```

在本地 `.env` 中填入生成结果（不要提交真实令牌）：

```dotenv
API_TOKENS={"u1":"替换成生成的随机令牌"}
HITL_TOKEN_TTL=300
```

每个用户使用不同的令牌，至少 32 字符；空配置会拒绝 API 请求。身份由令牌决定，请求中的 `user_id` 可省略；提供时必须与令牌身份一致。对外部署需通过 HTTPS 传输令牌。随后启动：

```bash
python -m app.server.main
```

打开 **[本地对话调试页](http://127.0.0.1:8000/demo)**，输入访问令牌后查看 SSE 事件和退款确认交互。令牌不会写入浏览器本地存储。

| 方法 | 路径 | 用途 |
| :--- | :--- | :--- |
| `POST` | `/api/chat` | 发起对话，接收 SSE 事件 |
| `POST` | `/api/confirm` | 批准或拒绝待确认操作 |
| `GET` | `/api/sessions/{session_id}` | 读取会话历史 |
| `DELETE` | `/api/sessions/{session_id}` | 删除会话 |
| `GET` | `/health` | 查询服务状态 |
| `GET` | `/demo` | 打开对话调试页 |

<details>
<summary><strong>请求示例 · Bash / curl</strong></summary>

```bash
# 先将本地令牌赋给 YIER_API_TOKEN 环境变量
curl -N -X POST http://127.0.0.1:8000/api/chat \
  -H 'Content-Type: application/json' \
  -H "Authorization: Bearer $YIER_API_TOKEN" \
  -d '{"session_id":"demo","message":"我的订单还没发货，怎么回事？"}'
```

所有 `/api/*` 请求都需携带 `Authorization: Bearer <令牌>`。ID 仅允许 1–64 位英文字母、数字、下划线和连字符，禁止系统保留名。

同一用户的同一会话正在处理时，聊天、确认、读取和删除请求返回 `409`。客户端断开 SSE 后，已接受的轮次继续完成并保存；请等待完成后读取历史，避免立即重发。

确认令牌绑定工具和完整参数，默认 5 分钟过期、只能使用一次；参数变化会使旧令牌失效。同参数执行结果在当前驻留会话中复用。服务重启或会话回收后，内存中的确认与去重状态不保留。确认接口返回 `execution_status`：`succeeded`、`failed`、`unknown` 或 `denied`，超时不会被显示成成功。

事件类型包括 `session`、`route`（多 Agent 模式）、`thought`、`tool_call`、`tool_result`、`confirm_required`、`final` 和 `error`。SSE 按处理事件输出，并非逐 token 输出。

</details>

### 本地工具参数校验

`@tool` 从函数类型注解和 Pydantic 字段约束构建入参模型，同一模型用于生成工具 JSON Schema 和执行前校验。所有本地工具开启严格类型检查，禁止多余字段；非法参数在创建确认令牌和执行业务函数之前返回 `invalid_tool_arguments`，包含字段位置、错误类型和修正提示，不回显原始输入。

- 订单号、商品关键词、退款原因和知识检索问题不能为空，并分别限制长度。
- `search_knowledge.top_k` 默认 3，仅接受 1～5 的整数；字符串、布尔值和越界值会被拒绝，不再静默转换或截断。
- `recall_user_memory.query` 可省略或为空；无参数工具仅接受空对象。
- 模型返回损坏 JSON、非对象或重复字段时，单 Agent 和多 Agent 都会收到工具错误反馈，可以在后续 ReAct 步骤修正参数。

当前统一 Pydantic 校验覆盖本地注册工具；远端 MCP 工具由其服务端负责字段级校验，客户端只统一检查参数 JSON 对象格式。参数校验不能替代订单归属或业务权限检查。

### 运行与性能

- **会话回收**：默认空闲 30 分钟回收、最多驻留 200 个会话，每 30 秒检查一次。容量不足时优先回收最久未使用的空闲会话；正在处理、后台维护中或有有效待确认操作的会话不会被淘汰。保存失败也会保留会话。回收不删除持久历史，下次请求可恢复。
- **冷启动**：按用户和会话分别控制初始化，不同会话可以并行创建；同一会话只构造一次。
- **响应顺序**：先保存完整答案，再发送 `final`。短期记忆提取、历史压缩移到后台，每阶段默认 5 秒预算；下一轮对话、确认、读取及删除会等待该会话的维护结束。维护失败会记录日志，已保存的答案仍可读取。CLI 会等待维护完成后返回。
- **长期记忆**：JSON 使用用户级进程内锁、操作系统文件锁和原子替换；PostgreSQL 使用用户状态行锁事务合并。记录来源、有效期、版本与重置水位，按查询召回，防止跨会话覆盖或旧偏好复活。具体配置、数据库迁移与验证边界见 [记忆修复说明](docs/memory-context-20261020.md)。
- **诊断**：服务输出 JSON 日志，包含请求编号、阶段耗时、异常类型和堆栈位置；新增日志不包含提示词、访问令牌或原始异常消息。聊天响应头提供 `X-Request-ID`，SSE 错误也包含对应编号。
- **超时与关闭**：聊天和确认共用工具超时控制。超时只代表停止等待，无法强制终止已启动的同步线程或撤销远端操作；结果未知时应核实状态，不能盲目重试。应用关闭共用默认 10 秒等待预算，并取消超预算的异步任务；同步工具自身仍必须配置网络/数据库超时，这个预算不是操作系统级强杀保证。

可在 `.env` 调整 `SESSION_IDLE_TTL`、`SESSION_MAX_RESIDENT`、`SESSION_REAP_INTERVAL`、`SHUTDOWN_TIMEOUT` 和 `MEMORY_MAINTENANCE_TIMEOUT`。

<details>
<summary><strong>启用 PostgreSQL、pgvector 与 Redis</strong></summary>

先准备 PostgreSQL 16、pgvector 扩展和目标数据库，再修改 `.env`：

```dotenv
DB_ENABLED=true
DATABASE_URL=postgresql+asyncpg://postgres:your_password@127.0.0.1:5432/yier_agent
RAG_BACKEND=pgvector
REDIS_ENABLED=true
REDIS_URL=redis://127.0.0.1:6379/0
```

当前数据库迁移中的向量列为 **1024 维**。需配置匹配的 Embedding 模型及接口，例如返回 1024 维向量的 `BAAI/bge-m3`；不要直接沿用基础示例中维度不同的模型。

```bash
python -m alembic upgrade head
python -m app.scripts.build_kb_index --backend pgvector
docker run -d --name yier-redis -p 6379:6379 redis:7
python -m app.server.main
```

修改知识库文档或切换 Embedding 模型后，需要重建对应索引。

</details>

<details>
<summary><strong>可选能力与配置</strong></summary>

| 配置 | 用途 |
| :--- | :--- |
| `MULTI_AGENT_ENABLED=true` | 启用售前、售后、投诉意图路由 |
| `MCP_ENABLED=true` | 接入 MCP 工具服务，另行启动 `python mcp_server/server.py` |
| `RERANK_ENABLED=true` | 启用重排；按需填写 `RERANK_API_KEY`、`RERANK_BASE_URL` |
| `MEMORY_ENABLED=true` | 启用长短期记忆 |
| `SKILLS_ENABLED=true` | 启用按需加载的业务技能 |
| `HITL_ENABLED=true` | 在工具执行前请求用户确认退款 |

完整配置见 [`.env.example`](.env.example)。

</details>

## 测试与评估

| 验证项 | 入口 / 记录 | 条件 |
| :--- | :--- | :--- |
| 异步并发 | [异步调用测试](tests/test_async_concurrency.py) | 模拟模型响应，检查多会话异步调用 |
| 会话隔离 | [并发隔离测试](tests/test_concurrency.py) | 模拟模型下 10 用户各 3 轮 SSE |
| 缓存与回源 | [缓存测试](tests/test_session_cache.py) | 含模拟依赖及可选真实数据库验证 |
| 限流与重试 | [稳定性测试](tests/test_stability.py) | 检查限流、模型重试和工具超时 |
| 退款确认 | [操作确认测试](tests/test_hitl.py) | 检查拦截、批准、拒绝和确认接口 |
| 对话评估 | [评估报告](eval_records/README.md) | 10 条用例，每种配置复跑两次 |
| 重排对比 | [检索报告](eval_records/rerank.md) | 6 条带金标文档的问句 |
| 服务压测 | [服务验证记录](eval_records/service_validation.md) | 模拟模型条件下的服务层测试 |

```bash
python tests/test_async_concurrency.py
python tests/test_session_cache.py
python tests/test_stability.py
python tests/test_hitl.py
python -m app.scripts.run_eval
```

部分测试需要模型接口、PostgreSQL 或 Redis。模拟模型下的并发与压测结果不代表真实模型吞吐量，小规模评估结果也仅适用于对应测试条件。

## 项目结构

```text
yier-agent/
├── main.py                 # 命令行入口
├── app/
│   ├── agent/              # ReAct、工具、Skills、记忆与 RAG
│   ├── config/             # 配置管理
│   ├── evaluation/         # 评估沙箱、轨迹与指标
│   ├── mcp_client/         # MCP 客户端
│   ├── multi_agent/        # 意图路由与子 Agent
│   ├── prompts/            # 对话、记忆及评估提示词
│   ├── schemas/            # 结构化响应
│   ├── scripts/            # 索引构建、评估及压测入口
│   └── server/             # HTTP 服务、数据库、缓存及限流
├── mcp_server/             # MCP 工具服务
├── migrations/             # 数据库迁移
├── tests/                  # 测试脚本
├── eval_records/           # 评估与压测记录
├── web/demo.html           # 对话调试页
└── locustfile.py           # Locust 压测场景
```

## 实现边界

- **部署与并发**：当前使用单 worker 和进程内会话注册表；同一用户的同一会话由服务端互斥控制，冲突返回 409。多 worker 部署需要重新设计状态加载与并发控制。
- **持久化与缓存**：`DB_ENABLED=false` 时使用 JSON；数据库模式下 PostgreSQL 持久化、Redis 加速冷读。内存限流回退仅适用于单进程。
- **词项检索**：PostgreSQL 使用原生全文检索与 `ts_rank`，不是 BM25；`simple` 配置无法对连续中文有效分词，中文查询主要依赖向量召回。
- **Agent 路由**：每轮交给一个子 Agent 处理，未实现子 Agent 间讨论或接力。
- **业务数据**：订单和退款工具使用示例数据，未接入真实支付系统。确认与执行结果去重状态仅在内存中；接入真实支付前，还需实现持久化业务幂等键、订单归属校验和结果对账，不能将本会话内去重当作支付级幂等保证。

## 许可

版权和使用范围见 [LICENSE](LICENSE) 与 [授权条款](授权条款.md)。

HTTP 安全与中断恢复回归测试（不调用真实模型或数据库）：

```bash
python -m unittest tests.test_tool_validation tests.test_security_regressions tests.test_optimizations -v
```


## 记忆验证与环境复现

已在全新 Python 3.13.14 环境按 `requirements-tested.lock.txt` 复跑原 87 项测试，扩展后的离线套件为 98 项通过；隔离真实 PostgreSQL 16.15 的 8 项并发/回滚测试通过。另有实际模型预算验证和可选 BGE 语义记忆召回。复现命令、报告与单 worker 等边界见 [验证与复现记录](docs/memory-validation-20261020.md)。
