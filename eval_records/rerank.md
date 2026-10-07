# rerank 前后检索对比

**测试方法**：不跑整表 `run_eval`（`knowledge_policy` 已经过程/结果双 1.00，
整表分数会被 `max_tokens` 预算淹没）。改对 **6 条带金标文档的问句** 比
「只召回 top-3」vs「召回 top-10 → `BAAI/bge-reranker-v2-m3` 重排 → top-3」。

| 项 | 值 |
|---|---|
| embedding | `BAAI/bge-m3` / 1024 维 |
| 召回 | numpy 余弦（与 CLI 默认 `RAG_BACKEND=numpy` 一致） |
| reranker | SiliconFlow `POST /v1/rerank`，模型 `BAAI/bge-reranker-v2-m3` |
| 重排配置 | 不换 embedding、不重建索引；reranker 与向量模型独立 |

## 结果

| 问句 | 金标文档 | 无 rerank 名次 | 有 rerank 名次 | 无 rerank top1 | 有 rerank top1 |
|---|---|---|---|---|---|
| 七天无理由退货可以吗 | 退换货政策 | 1 | 1 | 退换货政策 | 退换货政策 |
| 钻石会员有哪些权益 | 会员权益 | 1 | 1 | 会员权益 | 会员权益 |
| 偏远地区还包邮吗 | 配送说明 | 1 | 1 | 配送说明 | 配送说明 |
| 忘记密码了怎么办 | 常见问题FAQ | 1 | 1 | 常见问题FAQ | 常见问题FAQ |
| 退款多久到账 | 常见问题FAQ | **3** | **1** | 退换货政策 | 常见问题FAQ |
| 你们支持七天无理由退货吗？ | 退换货政策 | 1 | 1 | 退换货政策 | 退换货政策 |

| | Hit@1 | Hit@3 | MRR |
|---|---|---|---|
| 无 rerank | 0.83 | 1.00 | 0.89 |
| 有 rerank | **1.00** | 1.00 | **1.00** |

## 结果分析

1. **标题级问句已经饱和**：四条「文档名就在问句里」的 query，向量 top-1 已是金标，
   rerank 保持不变。知识库约含 19 个 chunk，结果仅反映当前小规模样本。
2. **真正拉开差距的是 FAQ 原题**：「退款多久到账」在政策文档里也反复出现（1-3 个工作日），
   余弦把「退换货政策 / 退款流程」排第一；cross-encoder 把 FAQ Q9（标题就是这句）顶到第一。
   金标按 FAQ 计，名次 3→1。
3. **分数不可比**：召回分是余弦 ~0.6–0.8；rerank 分是 0–1 相关度。注入 prompt 时只取顺序，
   不要把两种分加在一起。
4. **失败策略**：`/rerank` 挂了 fail-open，退回召回原序（见 `tests/test_rerank.py`）。

复现：`python -m app.scripts.compare_rerank --output eval_records/rerank_compare.json`

原始逐条 top-3 见 `rerank_compare.json`。
