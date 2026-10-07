"""加 / 不加 rerank 的检索对比（检索类用例，不跑整表 LLM judge）。

对同一组带金标文档的问句：
- off：混合/向量召回直接取 top-3
- on ：召回 top-10 → bge-reranker-v2-m3 重排 → top-3

指标：Hit@1 / Hit@3 / MRR。这比整表 Evaluation 更能单独说明 rerank 的贡献
（knowledge_policy 在 已经双 1.00，整表分数会被 token 预算淹没）。

用法：
  python -m app.scripts.compare_rerank
  python -m app.scripts.compare_rerank --output eval_records/rerank_compare.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from app.config.settings import settings  # noqa: E402
from app.agent.rag.backends import create_backend  # noqa: E402
from app.agent.rag.embedder import Embedder  # noqa: E402
from app.agent.rag.retriever import KnowledgeRetriever  # noqa: E402
from app.agent.tools import knowledge as knowledge_tool  # noqa: E402

CASES = [
    {"query": "七天无理由退货可以吗", "doc": "退换货政策"},
    {"query": "钻石会员有哪些权益", "doc": "会员权益"},
    {"query": "偏远地区还包邮吗", "doc": "配送说明"},
    {"query": "忘记密码了怎么办", "doc": "常见问题FAQ"},
    {"query": "退款多久到账", "doc": "常见问题FAQ"},
    {"query": "你们支持七天无理由退货吗？", "doc": "退换货政策"},  # knowledge_policy
]


def _rank_of(hits, doc: str) -> int | None:
    for i, h in enumerate(hits, start=1):
        if h.chunk.doc == doc:
            return i
    return None


def _metrics(rows: list[dict]) -> dict:
    n = len(rows) or 1
    hit1 = sum(1 for r in rows if r["rank"] == 1) / n
    hit3 = sum(1 for r in rows if r["rank"] is not None) / n
    mrr = sum((1 / r["rank"]) if r["rank"] else 0.0 for r in rows) / n
    return {
        "hit_at_1": round(hit1, 4),
        "hit_at_3": round(hit3, 4),
        "mrr": round(mrr, 4),
    }


def _run(retriever: KnowledgeRetriever, rerank: bool) -> list[dict]:
    settings.rerank_enabled = rerank
    retriever._reranker_ready = False
    retriever.last_reranked = False
    rows = []
    for case in CASES:
        hits = retriever.search(case["query"], top_k=3)
        rank = _rank_of(hits, case["doc"])
        rows.append({
            "query": case["query"],
            "gold_doc": case["doc"],
            "rank": rank,
            "top": [
                {"doc": h.chunk.doc, "section": h.chunk.section, "score": round(h.score, 4)}
                for h in hits
            ],
            "reranked": bool(retriever.last_reranked),
        })
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="eval_records/rerank_compare.json")
    args = parser.parse_args()

    old = (settings.rerank_enabled, settings.rag_backend)
    try:
        embedder = Embedder.from_settings(settings)
        name = settings.rag_backend.lower()
        if name == "pgvector" and settings.db_enabled:
            backend = create_backend("pgvector")
        else:
            backend = create_backend("numpy", index_path=ROOT / settings.kb_index_path)
            name = "numpy"
        retriever = KnowledgeRetriever(embedder=embedder, backend=backend)
        retriever.load()

        off_rows = _run(retriever, False)
        on_rows = _run(retriever, True)
        report = {
            "backend": name,
            "embedding_model": settings.embedding_model,
            "rerank_model": settings.rerank_model,
            "recall_k": settings.rerank_recall_k,
            "off": {"cases": off_rows, "metrics": _metrics(off_rows)},
            "on": {"cases": on_rows, "metrics": _metrics(on_rows)},
        }
        out = ROOT / args.output
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

        print("=" * 64)
        print(f"  rerank 对比  backend={name}  embedding={settings.embedding_model}")
        print("=" * 64)
        print(f"  {'问句':<28}{'off@':>6}{'on@':>6}  off top1 → on top1")
        for a, b in zip(off_rows, on_rows):
            print(
                f"  {a['query'][:26]:<28}"
                f"{str(a['rank']):>6}{str(b['rank']):>6}  "
                f"{a['top'][0]['doc'] if a['top'] else '-'} → "
                f"{b['top'][0]['doc'] if b['top'] else '-'}"
            )
        print()
        print(f"  off  Hit@1={report['off']['metrics']['hit_at_1']}  "
              f"Hit@3={report['off']['metrics']['hit_at_3']}  "
              f"MRR={report['off']['metrics']['mrr']}")
        print(f"  on   Hit@1={report['on']['metrics']['hit_at_1']}  "
              f"Hit@3={report['on']['metrics']['hit_at_3']}  "
              f"MRR={report['on']['metrics']['mrr']}")
        print(f"  已写入 {out}")
        return 0
    except Exception as e:  # noqa: BLE001
        print(f"对比失败: {type(e).__name__}: {e}")
        return 1
    finally:
        settings.rerank_enabled, settings.rag_backend = old
        knowledge_tool.reset_retriever()


if __name__ == "__main__":
    raise SystemExit(main())
