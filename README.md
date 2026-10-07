# 一二智能客服

面向商品咨询、订单查询、物流跟踪、退换货和投诉处理的电商客服 Agent。
支持 ReAct 工具调用、意图路由、RAG 知识检索、长短期记忆，以及 FastAPI / SSE 异步对话服务。

## 功能

- 对话编排：按意图分发至售前、售后或投诉 Agent，使用独立提示词与工具白名单。
- 工具与技能：统一接入本地工具及 MCP 工具，按需加载退货、订单跟踪、商品推荐等 Skills。
- 知识检索：支持 numpy、Chroma 和 pgvector 后端；pgvector 结合 HNSW 向量检索与 PostgreSQL 全文检索，可选 BGE 重排，重排失败时保留召回顺序。
- 会话与记忆：保留近期消息与历史摘要，提取会话事实和用户偏好；启用数据库后持久化会话及长期记忆。
- 在线服务：SSE 输出处理事件，提供会话查询、退款确认和健康检查接口；Redis 用于冷读缓存及令牌桶限流。
- 操作确认：退款工具在执行前检查确认状态，向用户发送 `confirm_required` 事件，批准后执行，拒绝后阻止该会话再次退款。
- 测试与评估：覆盖并发会话、缓存回源、超时重试和敏感操作；结合规则指标与 LLM-as-Judge 记录执行过程和回答质量。

## 技术栈

Python · FastAPI · Asyncio · SSE · PostgreSQL · pgvector · Redis · SQLAlchemy · Alembic · OpenAI SDK · Pydantic · MCP · Locust

## 服务设计

| 模块 | 实现 |
| --- | --- |
| 服务入口 | CLI 与 HTTP 共用 Agent 实现；HTTP 使用异步调用与 SSE 事件流 |
| 状态管理 | 进程内保留活跃 Agent；PostgreSQL 持久化；Redis 加速冷读并在故障时回源数据库 |
| 限流与重试 | Redis Lua 原子更新令牌桶，故障时回退进程内限流；模型调用超时后按指数退避重试 |
| 工具执行 | 工具超时控制与退款确认，确认状态按会话管理 |
| 检索 | 全文检索结果优先，向量结果去重补充；可选 cross-encoder 重排 |

## 项目结构

```text
yier-agent/
├── main.py                 # CLI 入口
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
├── web/demo.html           # 对话调试页面
└── locustfile.py           # Locust 压测场景
```

## 快速开始

需要 Python 3.11 或更高版本。以下命令使用 Bash：

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# 在 .env 中填写 OPENAI_API_KEY，按需配置 OPENAI_BASE_URL / MODEL_NAME
python -m app.scripts.build_kb_index
python main.py
```

CLI 支持 `skills`、`memory`、`reset`、`quit` 命令，回复包含意图、置信度和转人工标识。

可选配置：

| 配置 | 说明 |
| --- | --- |
| `MULTI_AGENT_ENABLED=true` | 启用售前、售后与投诉意图路由 |
| `MCP_ENABLED=true` | 接入 MCP 工具服务，需另行启动 `python mcp_server/server.py` |
| `RAG_BACKEND=pgvector` | 使用 PostgreSQL 混合检索，需启用数据库、完成迁移并构建相应索引 |
| `RERANK_ENABLED=true` | 对召回候选进行重排 |

### HTTP 服务

先准备 PostgreSQL 16 与 pgvector 扩展，在 `.env` 中设置 `DB_ENABLED=true` 和 `DATABASE_URL`。
Redis 为可选依赖，启用时设置 `REDIS_ENABLED=true` 及连接参数。

```bash
alembic upgrade head
# 使用 pgvector 检索时构建索引
python -m app.scripts.build_kb_index --backend pgvector
# 可选：启动 Redis
docker run -d --name yier-redis -p 6379:6379 redis:7
# 单 worker 启动
python -m app.server.main
```

```bash
curl -N -X POST http://127.0.0.1:8000/api/chat \
  -H 'Content-Type: application/json' \
  -d '{"session_id":"demo","user_id":"u1","message":"我的订单还没发货，怎么回事？"}'
```

调试页面：`http://127.0.0.1:8000/demo`。健康检查：`GET /health`。
SSE 事件包括 `session`、`thought`、`tool_call`、`tool_result`、`confirm_required` 和 `final`。

## 实现边界

- 当前使用单 worker 和进程内会话注册表。多 worker 部署需要重新设计状态加载与并发控制。
- Redis 仅加速会话冷读，热态仍在进程内；Redis 不可用时会话读取回源 PostgreSQL。内存限流回退仅适用于单进程。
- PostgreSQL 词项检索使用原生全文检索与 `ts_rank`，不是 BM25。`simple` 配置无法对连续中文有效分词，中文查询主要依赖向量召回。
- 多 Agent 采用意图路由：每轮交给一个子 Agent 处理，未实现子 Agent 间讨论或接力。
- 订单和退款工具使用示例业务数据；退款确认机制用于验证操作流程，未接入真实支付系统。
- `DB_ENABLED=false` 时使用 JSON 持久化；评估数据和压测条件详见下方记录。

## 测试与评估

```bash
pytest
python tests/test_concurrency.py
python tests/test_stability.py
python tests/test_hitl.py
python -m app.scripts.run_eval
```

部分测试需要模型接口、PostgreSQL 或 Redis。并发与服务层压测使用模拟模型，不代表真实模型吞吐量。

- [对话评估](eval_records/README.md)：10 条用例、每种配置复跑两次，保留过程分、结果分和 token 记录。
- [重排对比](eval_records/rerank.md)：6 条带金标文档的问句，说明样本规模和检索配置。
- [服务验证](eval_records/service_validation.md)：并发、缓存、确认流程及服务层压测条件。

## 许可

版权和使用范围见 [LICENSE](LICENSE) 与 [授权条款](授权条款.md)。
