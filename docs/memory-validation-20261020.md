# 记忆系统验证与复现（2026-10-20）

## 验证结果

| 项目 | 结果与证据 |
| --- | --- |
| 原 87 项测试独立复跑 | 原环境通过后，另建虚拟环境，按 154 个精确依赖版本重新安装，87 项再次全部通过；历史记录 reports/repro-87-tests.json。 |
| 新增修复后的完整离线套件 | 独立环境 98 项通过，无跳过；reports/upgrade-tests.json 同时记录 Python、解释器路径、依赖核对结果、锁文件与源码 SHA-256。 |
| 真实 PostgreSQL | PostgreSQL 16.15 隔离容器中 8 项通过，无跳过；reports/memory-postgres.json。不是 SQLite 或 mock 数据库。 |
| 实际模型预算 | deepseek-v4-flash：长中文、大工具结果、超限本地拒绝均通过；reports/context-live.json。 |
| 表达差异召回 | 8 条合成用例：关键词 6/8，BGE 和实际接入路径均为 8/8；其中 6 条正例由 4/6 改善为 6/6，2 条无关查询均未误召回；reports/memory-recall.json。 |

所有数量仅指对应报告的运行范围。召回用例规模很小，用于发现具体漏召回并验证修复，不代表真实用户准确率；没有做吞吐量、长期稳定性或线上故障注入评估。

## 固定环境与复现

实际验证环境：Windows 11 x64，Python 3.13.14。`.python-version` 固定解释器版本，`requirements-tested.lock.txt` 固定已测 154 个包版本。基础解释器所在目录名字包含 3.13.12，但 sys.version 实际为 3.13.14，以报告记录为准。锁文件包含 Windows 专属依赖，不声称跨平台可原样复现。

从项目根目录执行，先确保 `python --version` 为 3.13.14：

```powershell
python -m venv app/sessions/repro-env
app/sessions/repro-env/Scripts/python.exe -m pip install -r requirements-tested.lock.txt
app/sessions/repro-env/Scripts/python.exe -m pip check
app/sessions/repro-env/Scripts/python.exe -X utf8 scripts/verify_upgrades.py
```

本机已经实际执行以上创建、安装、依赖核对和回归步骤。离线脚本主动关闭语义接口调用；模型使用测试替身，不需要请求真实模型。依赖或 Python 版本不符会使运行状态失败，报告会列出差异。`requirements.txt` 保留开发用依赖范围；交付复现使用锁文件。

## PostgreSQL 验证范围

启动 Docker Desktop 后执行：

```powershell
app/sessions/repro-env/Scripts/python.exe -X utf8 scripts/verify_memory_postgres.py
```

脚本创建名称随机、仅绑定 127.0.0.1 随机端口的临时容器，用随机密码和 tmpfs 数据目录，不使用项目应用数据库。结束时停止并自动删除该容器。镜像标签为 postgres:16-alpine，报告记录实际镜像 ID 与 PostgreSQL 版本，便于审计和按 digest 固定后续运行。

实际验证：

1. 同一用户 12 个线程同时合并独立事实，完整保留 12 条，其他用户为空。
2. 4 个独立进程写入同一用户，合并不丢事实。
3. reset 先持有数据库行锁，旧写入实际阻塞，reset 提交后旧写入不能恢复记忆。
4. 旧写入先持有行锁，reset 实际阻塞，最终 reset 清空全部旧事实。
5. 旧偏好晚保存不能覆盖观察时间更新的偏好。
6. 数据库触发器主动拒绝 UPDATE，实际 SQL 失败回滚，旧数据不变，后续正常写入仍可获得锁。
7. 新用户首次事务在 INSERT 后抛异常，确认插入也被回滚。
8. 旧表事实保持无来源状态；reset 后不能再次从旧表恢复。

竞争测试读取 pg_stat_activity 确认发生真实锁等待，而非仅靠线程启动顺序推断。测试使用实际 ORM 表结构创建隔离测试表；完整应用迁移链另有离线 SQL 检查，不等于对生产迁移、pgvector/RAG、Redis 或整个服务做了数据库集成验收。

## 实际模型上下文预算

使用项目已配置的接口，只有合成输入：

```powershell
app/sessions/repro-env/Scripts/python.exe -X utf8 scripts/verify_context_live.py
```

- 长中文：本地消息序列化估算 21595 字节，供应商返回输入 3952 tokens，模型正确返回末尾核验码。
- 长工具结果：从约 180575 字节压缩到 745 字节，供应商返回输入 179 tokens，订单号 ORD-9081 和金额 129.5 正确保留并用于回答。工具调用先由真实模型生成，随后使用合成大结果。
- 超限用户输入：本地明确拒绝，模型接口调用次数为 0。

报告中的默认输出预留为 2048；这两个实际调用各显式请求 256 输出 tokens，生成工具调用的准备步骤请求 1024。测试采用配置窗口 32768，验证该安全预算内的行为，不宣称测出了模型供应商最大窗口，也不将字节估算当作精确 tokenizer。

实际测试还发现当前模型的工具调用要求回传 reasoning_content。ReAct、MultiAgent 和 Graph 的工具消息现保留这一不透明协议字段；PG 通过既有消息元数据通道持久化，调用前恢复给供应商，用户消息接口与来源查看接口不展示该字段。没有将其作为用户偏好或摘要内容。

## 语义召回的决策与实现

先测得“深色”无法关键词匹配“我喜欢黑色”、“预算有限”无法匹配“我喜欢便宜一点的”。在回答对照中，缺失记忆时模型回答不知道；提供相关原始偏好后能回答黑色。因此才加入语义补充，而不是直接增加向量库复杂度。

实现复用已配置的 BAAI/bge-m3 embeddings：每个记忆管理器有容量 256 的向量缓存，关键词重合度与语义相似度共同排序，语义补充最低阈值默认 0.55。来源可信性、有效期、会话范围、品类范围和动态事实过滤仍在返回前执行。相似表达只用于召回原始证据，不会把“喜欢黑色”改写成“喜欢所有深色”。embedding 请求期间发生 reset 时，返回前刷新存储状态，不能靠旧向量重新召回已清空事实。

配置：

```dotenv
MEMORY_SEMANTIC_ENABLED=true
MEMORY_SEMANTIC_MIN_SCORE=0.55
MEMORY_SEMANTIC_TIMEOUT=3
```

本地 `.env` 已开启；模板和代码默认关闭，其他部署可按需开启。向量缓存属于单个 manager，不跨用户共享；提示词组装只读已准备的结果，不在事件循环里直接同步调用 embedding API。超时或接口失败记录事件并回退关键词。该实现不需要额外向量数据库，不承诺在当前小样本外达到同样召回率。

运行实际召回对比：

```powershell
app/sessions/repro-env/Scripts/python.exe -X utf8 scripts/evaluate_memory_recall.py
```

## 保留的边界

服务仍明确按单 worker 部署。这次 PostgreSQL 多线程/多进程记忆测试通过，不代表会话归属、注册表、工作流和整个服务已支持多 worker。

事实提取仍以明确表达为主。新增否定、改口、鞋/衣服/裤子范围、今日午夜、会话临时偏好和不确定表达测试；支持的明确范围被保留，不支持或不确定的表达不升级为确认事实，不承诺理解任意复杂表达、反讽或隐含偏好。
