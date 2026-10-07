"""向量后端抽象接口。

设计要点：
- upsert：构建索引时调用，传入 chunks + 向量 + embedding_model 标识。
  实现需把 embedding_model 持久化，下次加载时校验一致性（防止跨模型混用）。
- search：在线检索时调用，输入 query 向量，返回 Top-K 命中。
- size：当前已索引的 chunk 数量；首次访问可触发懒加载。
- load：从持久化路径加载索引；NumpyBackend 是读 JSON，ChromaBackend 是连 client。

不在接口里抽象 query 文本→向量这一步，是因为 embedder 由 KnowledgeRetriever 持有，
让后端只关心"向量怎么存、怎么搜"这一件事，职责更纯粹。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from app.agent.rag.chunker import Chunk


@dataclass
class RetrievedChunk:
    chunk: Chunk
    score: float


class VectorBackend(ABC):
    """向量索引后端的统一接口。

    search / search_hybrid 的分工：
    - search：单路向量召回，所有后端都必须能跑（也是无全文能力的后端唯一路径）。
    - search_hybrid：混合召回 = 向量 lane + 词项 lane（全文检索）。基类默认实现退化成
      纯向量 search，即「没有全文索引的后端，混合就是纯向量」；
      带全文能力的后端（PgVectorBackend）覆盖为真正的融合。
      这样 KnowledgeRetriever 只调 search_hybrid，numpy/chroma 行为不变。
    """

    # 是否有真正的词项检索 lane（决定 search_hybrid 是真混合还是纯向量）
    supports_hybrid: bool = False

    @abstractmethod
    def upsert(
        self,
        chunks: list[Chunk],
        vectors: list[list[float]],
        embedding_model: str,
    ) -> None:
        """全量重建索引（覆盖式）。

        每次调用都会清空既有数据再写入，避免脏数据。增量更新不在范围。
        """

    @abstractmethod
    def search(self, query_vector: list[float], top_k: int) -> list[RetrievedChunk]:
        """按余弦相似度（或等价度量）返回 Top-K 命中。"""

    def search_hybrid(
        self, query_text: str, query_vector: list[float], top_k: int
    ) -> list[RetrievedChunk]:
        """混合召回；默认实现 = 纯向量（无词项 lane 的后端）。"""
        return self.search(query_vector, top_k=top_k)

    @abstractmethod
    def size(self) -> int:
        """当前已索引的 chunk 数量。"""

    @abstractmethod
    def load(self) -> None:
        """从持久化存储加载索引；不存在时抛 FileNotFoundError。"""

    @abstractmethod
    def expected_embedding_model(self) -> str:
        """已持久化索引使用的 embedding 模型名（用于校验）。"""
