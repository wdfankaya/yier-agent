"""知识库检索工具：通过向量/混合检索回答 FAQ、政策类问题。

与查订单/查物流这类「结构化数据查询」工具不同，
search_knowledge 面向非结构化文本（退换货政策、配送说明、FAQ 等），
返回 Top-K 命中片段及其来源，由 LLM 引用回答。

检索后端由 settings.rag_backend 切换（.env RAG_BACKEND，默认 numpy）：
- numpy   ：手写余弦相似度，无额外向量数据库依赖，便于调试（离线/默认）
- chroma  ：嵌入式向量数据库，HNSW 索引，支持本地持久化
- pgvector：PG 里向量 lane（HNSW）+ 全文检索 lane 混合检索（需 DB_ENABLED=true）
- 召回 top-10 后可选 cross-encoder 重排取 top-3（`RERANK_ENABLED`，失败回落原序）

为避免每次进程启动都重建索引，单例缓存 Retriever。
"""

from pathlib import Path
from typing import Optional

from app.agent.tools.decorator import tool
from app.config.settings import settings
from app.agent.rag.backends import create_backend
from app.agent.rag.embedder import Embedder
from app.agent.rag.retriever import KnowledgeRetriever

_retriever: Optional[KnowledgeRetriever] = None


def _create_backend_from_settings():
    """根据 settings.rag_backend 创建对应后端实例。"""
    name = settings.rag_backend.lower()
    if name == "numpy":
        return create_backend("numpy", index_path=Path(settings.kb_index_path))
    if name == "chroma":
        return create_backend(
            "chroma",
            persist_dir=Path(settings.chroma_persist_dir),
            collection_name=settings.chroma_collection,
        )
    if name == "pgvector":
        # 需要 DB_ENABLED=true（repository sync 层连 PG；离线 CLI/eval 无 DB 时别选）
        if not settings.db_enabled:
            raise RuntimeError(
                "pgvector 后端需要 db_enabled=true（.env DB_ENABLED=true）；"
                "离线/无 PG 场景请用 numpy。"
            )
        return create_backend("pgvector")
    raise ValueError(
        f"未知的 RAG 后端: {settings.rag_backend}（可选: numpy / chroma / pgvector）"
    )


def _get_retriever() -> KnowledgeRetriever:
    global _retriever
    if _retriever is None:
        embedder = Embedder.from_settings(settings)
        backend = _create_backend_from_settings()
        _retriever = KnowledgeRetriever(embedder=embedder, backend=backend)
        _retriever.load()
    return _retriever


def reset_retriever() -> None:
    """清空单例缓存（测试或切换后端时使用）。"""
    global _retriever
    _retriever = None


@tool(
    desc=(
        "检索一二商城的政策与帮助文档（退换货政策、配送说明、会员权益、常见问题 FAQ）。"
        "当顾客询问规则、流程、时效、是否支持等政策类问题时使用，"
        "比如「能退货吗」「多久到账」「钻石会员有什么权益」「偏远地区包邮吗」。"
        "返回 Top-K 命中片段及来源文档，请基于检索结果回答，不要编造政策"
    ),
    params={
        "query": "用顾客的原问题或一句简洁中文描述要查的政策点",
        "top_k": "返回片段数，默认 3，最大 5",
    },
)
def search_knowledge(query: str, top_k: int = 3) -> dict:
    """检索退换货政策、配送说明、会员权益、FAQ 等知识库内容。

    Returns:
        {
          "success": bool,
          "backend": "numpy" | "chroma" | "pgvector",
          "query": str,
          "results": [
            {"doc": "...", "section": "...", "score": 0.83, "text": "..."},
            ...
          ],
          "error": "..."  # 仅失败时存在
        }
    """
    if not query or not query.strip():
        return {"success": False, "error": "query 不能为空", "query": query, "results": []}

    try:
        retriever = _get_retriever()
    except FileNotFoundError as e:
        return {
            "success": False,
            "error": str(e),
            "backend": settings.rag_backend,
            "query": query,
            "results": [],
        }
    except Exception as e:
        return {
            "success": False,
            "error": f"知识库初始化失败: {e}",
            "backend": settings.rag_backend,
            "query": query,
            "results": [],
        }

    top_k = max(1, min(int(top_k or 3), 5))
    hits = retriever.search(query, top_k=top_k)

    return {
        "success": True,
        "backend": settings.rag_backend,
        "query": query,
        "reranked": bool(getattr(retriever, "last_reranked", False)),
        "results": [
            {
                "doc": h.chunk.doc,
                "section": h.chunk.section,
                "score": round(h.score, 4),
                "text": h.chunk.text,
            }
            for h in hits
        ],
    }
