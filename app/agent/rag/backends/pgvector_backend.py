"""pgvector 后端：向量（HNSW）+ 全文检索（PG 原生全文）双 lane 混合检索。

与 numpy / chroma 的关系：
- 同样的 VectorBackend 接口：upsert / search / size / load / expected_embedding_model。
- 存储真相源从 JSON / chroma 目录换成 PostgreSQL 的 kb_chunks 表
  （向量列 vector(1024) + HNSW 索引 idx_kb_chunks_vec + GIN 索引 idx_kb_chunks_fts）。
- 新增能力：supports_hybrid=True。search_hybrid = 向量 lane + 全文检索 lane 融合，
  numpy/chroma 没有词项索引，search_hybrid 退化成纯向量（基类默认）。

词项检索优先的设计依据：
- 对「ORD-20240115-001」这种无语义串，1024 维余弦对所有 chunk 都是 ~0.36 的
  漂移噪音（实测），向量 lane 给不出定位；而全文检索的 exact token 命中是唯一真信号。
- 'simple' 分词按非字母数字切，中文连续文本切不成词，三类实测行为：
  纯中文问句 → 全文检索无召回 → 回落纯向量；纯单号 → 精确命中含该标识的 chunk；
  单号与中文连写（如「ORD-… 的售后进度怎么查」）→ 连续中文被并成一个整串 token，
  plainto_tsquery AND 上它后整体失配 → 仍回落纯向量。故全文检索只承接"干净"的
  英数标识查询，这是明示的边界，不装 zhparser（README 已写明）。
- 因此采用「词项 lane 优先 + 向量 lane 去重填充」的融合，而非对两条不可比分数
  （余弦相似度 vs ts_rank）做加权平均，避免直接混合不同评分尺度。
- 限制：若 query 同时含普通英文词，全文检索可能把含该词的 chunk 顶到
  最前；本知识库英数 token 基本都是强标识（单号/EMS/顺丰等），风险可控。

DB 访问走 app.server.repository 的 sync_kb_*（worker 线程无事件循环 →
repository 内部用 asyncio.run 临时 loop + NullPool，遵循存储层约定）。
"""

from __future__ import annotations

from app.agent.rag.backends.base import RetrievedChunk, VectorBackend
from app.agent.rag.chunker import Chunk

__all__ = ["PgVectorBackend"]


class PgVectorBackend(VectorBackend):
    """PostgreSQL pgvector + 全文检索混合后端。"""

    supports_hybrid = True

    def __init__(self) -> None:
        self._embedding_model: str = ""
        self._size_cache: int | None = None

    # ---- repository sync 层惰性接线（避免离线无 asyncpg 时 import 报错） ----

    def _repo(self):
        from app.server.repository import (  # 延迟到真正用 DB 时才 import
            sync_kb_bm25_search,
            sync_kb_meta_value,
            sync_kb_replace_all,
            sync_kb_size,
            sync_kb_vector_search,
        )
        return (
            sync_kb_replace_all,
            sync_kb_vector_search,
            sync_kb_bm25_search,
            sync_kb_size,
            sync_kb_meta_value,
        )

    # ---- VectorBackend 接口 ----

    def upsert(
        self,
        chunks: list[Chunk],
        vectors: list[list[float]],
        embedding_model: str,
    ) -> None:
        if len(chunks) != len(vectors):
            raise ValueError(
                f"chunks 与 vectors 长度不一致: {len(chunks)} vs {len(vectors)}"
            )
        replace, *_ = self._repo()
        rows = [c.to_dict() for c in chunks]  # chunk_id/doc/section/text
        replace(rows, vectors, embedding_model)
        self._embedding_model = embedding_model
        self._size_cache = len(chunks)

    def search(self, query_vector: list[float], top_k: int) -> list[RetrievedChunk]:
        """单路向量召回（HNSW 余弦最近邻），score = 余弦相似度。"""
        _, vector_search, *_ = self._repo()
        hits = vector_search(query_vector, top_k)
        return [self._to_retrieved(h) for h in hits]

    def search_hybrid(
        self, query_text: str, query_vector: list[float], top_k: int
    ) -> list[RetrievedChunk]:
        """混合召回：全文检索 lane 优先，向量 lane 去重填充。

        融合去重规则：
        1. 先执行全文检索（ts_rank 排序）。命中即"query 里出现了知识库文本中的精确
           token"，置信度高 → 放前面（score 归一化到 lane 内 [0,1]，top1=1.0）。
        2. 再用向量 lane 补位：跳过 全文检索已命中的 chunk_id，其余按余弦相似度降序。
        3. 全文检索无召回（典型中文问句）→ 退化为纯向量结果。
        """
        _, vector_search, bm25_search, *_ = self._repo()
        bm_hits = bm25_search(query_text, top_k)
        vec_hits = vector_search(query_vector, top_k)

        if not bm_hits:
            return [self._to_retrieved(h) for h in vec_hits]

        # 全文检索 lane 分数做 lane 内归一化：top1 视为 1.0，便于和向量相似度同一量纲展示
        bm_top = bm_hits[0]["score"] or 1.0
        merged: list[RetrievedChunk] = []
        seen: set[str] = set()
        for h in bm_hits:
            seen.add(h["chunk_id"])
            merged.append(self._to_retrieved(h, score=h["score"] / bm_top))
        for h in vec_hits:
            if h["chunk_id"] in seen:
                continue
            seen.add(h["chunk_id"])
            merged.append(self._to_retrieved(h))
        return merged[:top_k]

    def size(self) -> int:
        *_, kb_size, _meta = self._repo()
        if self._size_cache is None:
            self._size_cache = kb_size()
        return self._size_cache

    def load(self) -> None:
        """确认 kb_chunks 已构建；空表 = 还没 build，抛 FileNotFoundError。"""
        *_, kb_size, kb_meta = self._repo()
        n = kb_size()
        if n == 0:
            raise FileNotFoundError(
                "PostgreSQL 知识库索引为空。请先运行 "
                "`python app/scripts/build_kb_index.py --backend pgvector` 构建索引。"
            )
        self._size_cache = n
        self._embedding_model = kb_meta("embedding_model")

    def expected_embedding_model(self) -> str:
        if not self._embedding_model:
            try:
                self.load()
            except FileNotFoundError:
                return ""
        return self._embedding_model

    # ---- 内部工具 ----

    @staticmethod
    def _to_retrieved(h: dict, score: float | None = None) -> RetrievedChunk:
        chunk = Chunk(
            chunk_id=h["chunk_id"],
            doc=h["doc"],
            section=h["section"],
            text=h["text"],
        )
        return RetrievedChunk(chunk=chunk, score=h["score"] if score is None else score)
