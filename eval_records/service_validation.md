# 服务验证记录

本记录汇总单 worker 服务的缓存、异步调用、限流、敏感操作确认与并发验证。

## 验证范围

| 模块 | 验证内容 | 文件 |
| --- | --- | --- |
| 缓存 | Redis 冷读、进程内热态及数据库回源 | `tests/test_session_cache.py` |
| 异步调用 | `achat`、AsyncOpenAI 与 5 路并发重叠 | `tests/test_async_concurrency.py` |
| 稳定性 | Redis Lua / 内存令牌桶、模型超时重试、工具 10 秒超时 | `tests/test_stability.py` |
| 操作确认 | 退款拦截、SSE 确认事件与 `/api/confirm` | `tests/test_hitl.py` |
| 重排 | top-10 召回经 BGE 重排后取 top-3，保持 embedding 不变 | `rerank.md` |
| 会话隔离 | 模拟模型下 10 用户各 3 轮 SSE；记忆与技能调用上下文隔离 | `tests/test_concurrency.py` |
| 压测 | 模拟模型下的服务层请求处理 | `locust_baseline.json` |

## 服务层压测

条件：10 虚拟用户、15 秒，SSE 对话与历史查询比例约 3:1，关闭限流，模型以约 20 毫秒延迟的模拟响应替代。

| 指标 | 结果 |
| --- | --- |
| 成功 / 失败 | 2640 / 1 |
| RPS | 176 |
| p50 / p95 / p99 | 54ms / 74ms / 90ms |

这些数据仅反映该测试环境下 FastAPI、SSE 和会话注册表的处理表现，不代表真实模型吞吐量或服务性能上限。
真实模型调用另受 60 秒超时和默认每用户每分钟 10 次令牌桶限流约束。
真实服务的 Locust 入口为 `locust -f locustfile.py --host http://127.0.0.1:8000`。

## 实现限制

- 会话热态保存在进程内，PostgreSQL 持久化，Redis 加速冷读。当前仅支持单 worker。
- 退款确认在工具执行层拦截，用户批准后放行；业务数据为示例数据。
- 重排与 embedding 独立，当前 embedding 为 bge-m3 / 1024 维。6 条问句的 Hit@1 从 0.83 变为 1.00，样本较小，详见重排报告。
- Memory 与 Skill 通过 ContextVar 绑定到调用上下文，并在上述并发测试中验证隔离。
- `DB_ENABLED=false` 时保留 JSON 持久化，用于 CLI 和离线评估。

## 后续工作

1. 使用 Docker Compose 管理 API、PostgreSQL 和 Redis。
2. 增加真实模型压测，记录 SSE 连接数及数据库建连开销。
3. 设计多 worker 的状态加载与并发控制。
4. 评估中文分词或其他词项检索方案；当前 PostgreSQL `simple` 配置不提供中文分词。
