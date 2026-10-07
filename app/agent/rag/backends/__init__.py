"""向量后端：抽象接口 + 多套实现（Numpy 手写 / Chroma / PgVector 混合检索）。

设计目标：同一份知识库可在多种后端间切换，便于直观对比
- Numpy 手写余弦（便于调试，无额外向量数据库依赖，适合 < 1k chunks）
- Chroma 向量数据库（生产标配，HNSW/IVF 索引，支持元数据过滤 + 并发写入）
- PgVector 混合检索（PG 里向量 lane + 全文检索 lane，见 pgvector_backend 模块）

通过 settings.rag_backend 选择，对上层 KnowledgeRetriever / search_knowledge 透明。
"""

from app.agent.rag.backends.base import RetrievedChunk, VectorBackend
from app.agent.rag.backends.numpy_backend import NumpyBackend

__all__ = [
    "RetrievedChunk",
    "VectorBackend",
    "NumpyBackend",
    "create_backend",
]


def create_backend(name: str, **kwargs) -> VectorBackend:
    """工厂方法：根据名称创建对应后端。

    name: "numpy" | "chroma" | "pgvector"
    kwargs: 后端特定参数（index_path / persist_dir / collection 等）
    """
    name = (name or "numpy").lower()
    if name == "numpy":
        return NumpyBackend(index_path=kwargs["index_path"])
    if name == "chroma":
        from app.agent.rag.backends.chroma_backend import ChromaBackend

        return ChromaBackend(
            persist_dir=kwargs["persist_dir"],
            collection_name=kwargs.get("collection_name", "yier_kb"),
        )
    if name == "pgvector":
        from app.agent.rag.backends.pgvector_backend import PgVectorBackend

        return PgVectorBackend()
    raise ValueError(f"未知的 RAG 后端: {name}（可选: numpy / chroma / pgvector）")
