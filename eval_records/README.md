# Evaluation 改造前后对比留档

> rerank 对比见 [rerank.md](rerank.md)；服务验证见 [service_validation.md](service_validation.md)、[locust_baseline.json](locust_baseline.json)。

**评估摘要**：把同一套黄金测试集跑在「改造前」（`fa3816f`：JSON 存储 + numpy RAG）与「服务化全开」
（评估时版本：PostgreSQL 落库 + pgvector 混合检索）上，各跑 2 次 —— **通过面完全一致（均 1/10）**，
过程/结果分差在模型采样噪音带内，唯一可复现的系统差异是现状跑平均 token 高约 +20%（逐用例高方差，
集中于少数长对话用例，详见「局限」）。**没有证据表明服务化改造损伤对话效果。**

## 测试配置与方法

| 项 | 基线（改造前） | 现状（服务化全开） |
|---|---|---|
| 代码 | `git worktree` 检出 `fa3816f`（服务化改造前最后一个 commit） | `db6847b`（评估时版本） |
| 持久化 | JSON（fa3816f 时代无 PG） | PostgreSQL（`DB_ENABLED=true`，落库+重建） |
| RAG 后端 | numpy（JSON 索引） | pgvector（HNSW 向量 + 全文混合检索） |
| 评估 | `run_eval --mode single --judge`，10 用例 | 同左 |
| 复跑 | 2 次（n=2） | 2 次（n=2） |

控制变量：
- **同一份测试集** `app/evaluation/cases.json`（两边 checkout 的该文件内容一致）。
- **同一份 KB 语料**：曾在 FAQ 加过一条 Q11，基线 worktree 的 `常见问题FAQ.md` 已与本仓库同步成一致
  内容（避免 KB 内容差异污染对比），基线在其上重建 numpy 索引（19 chunks / bge-m3 / 1024 维）。
- **同一模型**：agent 与 LLM-judge 都用 `deepseek-v4-flash`，temperature 同设置。
- 沙箱一律关 memory/MCP；每次 `run_eval` 用一次性 `eval-<hex>` 用户隔离 PG 会话行，
  两跑之间不串上下文、不污染开发库 `default` 用户（现状两跑各落 `eval-*` 10 会话 / ~64 messages）。

## 结果（每个指标 = 该配置 2 跑的均值；通过率 = 2 跑通过次数）

| 用例 | 通过 基/现 | 过程分 基→现 | 结果分 基→现 | 平均 token 基→现 |
|---|---|---|---|---|
| order_query_basic | 0/2 · 0/2 | 0.64 → 0.71 | 1.00 → 1.00 | 10349 → 10283 |
| logistics_track | 0/2 · 0/2 | 0.71 → 0.78 | 1.00 → 1.00 | 7656 → 7076 |
| logistics_no_tracking | 0/2 · 0/2 | 0.78 → 0.78 | 1.00 → 1.00 | 8541 → 10098 |
| product_out_of_stock | 0/2 · 0/2 | 0.67 → 0.65 | 1.00 → 1.00 | 9682 → 12531 |
| return_request | 0/2 · 0/2 | 0.40 → 0.60 | 0.90 → 0.90 | 7171 → 10166 |
| knowledge_policy | 0/2 · 0/2 | 1.00 → 1.00 | 1.00 → 1.00 | 6848 → 9151 |
| complaint_to_human | 0/2 · 0/2 | 0.70 → 1.00 | 0.60 → 0.47 | 7075 → 11332 |
| greeting_no_tool | 2/2 · 2/2 | 1.00 → 1.00 | 1.00 → 1.00 | 2935 → 2886 |
| multi_turn_followup | 0/2 · 0/2 | 0.68 → 0.65 | 1.00 → 0.90 | 18473 → 22062 |
| list_orders | 0/2 · 0/2 | 0.92 → 0.92 | 1.00 → 1.00 | 7024 → 7252 |

| 汇总 | 通过率 | 平均过程分 | 平均结果分 | 平均 token/整跑 |
|---|---|---|---|---|
| 基线 ×2 | 1/10 ×2 | 0.75 | 0.95 | 85,756 |
| 现状 ×2 | 1/10 ×2 | 0.81 | 0.93 | 102,838 |

## 结论

1. **通过面完全一致**：两配置各 2 跑都是 1/10，且唯一通过的用例（greeting_no_tool）两边都稳定通过。
2. **质量指标持平**：过程分 +0.06、结果分 −0.02，量级都在「同配置 2 跑也会出现的波动」内
   （例如 `complaint_to_human` 结果分在基线自己 2 跑就是 0.75/0.50 波动，现状 0.47/0.60）。
3. **RAG 用例对比**：`knowledge_policy`（唯一命中 `search_knowledge` 的用例）在
   现状 pgvector 混合检索下与基线 numpy 同为 过程/结果 双 1.00 —— 换后端没有伤检索质量。
4. **绝对通过率低（1/10）是数据集校准问题、改造前后一致**：`max_tokens` 预算按「较精简的回复模型」
   设定（4k–9k），`deepseek-v4-flash` 回复冗长导致多数用例超预算被判 token 不达标；
   `return_request`/`complaint_to_human` 偶发 intent 标签不匹配（枚举里有这些值，是结构化判定与
   数据集期望不一致）。以上在 fa3816f 上同样存在，**不是服务化引入的回归**。

## 局限

- **n=2/配置**：LLM 采样有噪音，表内过程/结果分用均值呈现，单跑原始数据见下方 JSON；个别维度的
  ±0.1 级差异不宜过度解读。
- **token 现状略高（+~20%）是唯一可复现观测，但逐用例高方差**：高的用例（product_out_of_stock、
  return_request、knowledge_policy、complaint_to_human、multi_turn_followup）一次能差 3–5k，
  低的用例（order_query、logistics_track、greeting）几乎持平或略低 → 更可能是该模型在部分长对话上
  回复更长（judge 输入随之变大）的采样效应，而非某个点上的代码缺陷；要定论需加大 n。
- **只复测了 single 模式**：multi 路由分发本次未纳入。
- **judge 与 agent 同模型**：没有第二家模型做交叉裁判。
- 现状跑用的是「服务化全开」配置（PG + pgvector）；运行配置默认仍是 numpy RAG、DB 开关关闭时
  回退 JSON —— 两边跑的是同一套代码的两档配置与两个代码状态，口径如上表。

## 文件

- `baseline_fa3816f.json` / `baseline_fa3816f_rep2.json` —— 改造前 2 次完整报告（含 trace + judge 理由）
- `current_head.json` / `current_head_rep2.json` —— 现状 2 次完整报告
- 复现方法：`python -m app.scripts.run_eval --mode single --judge --output eval_records/<name>.json`
  （现状跑加环境变量 `DB_ENABLED=true RAG_BACKEND=pgvector`；基线在 fa3816f worktree 内用 numpy）

注：文中的提交标识用于记录原始评估版本，当前仓库未保留这些历史提交；原始结果见本目录 JSON 报告。
