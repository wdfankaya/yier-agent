"""知识库检索器：query → 向量化 → 混合召回 → 可选 rerank。

search 先按 recall_k（默认 10）召回，再交给 cross-encoder 重排取 top_k（默认 3）。
rerank 关掉或失败时，直接切召回原序的 top_k（fail-open）。
"""

from __future__ import annotations

from typing import Optional

from app.agent.rag.backends.base import RetrievedChunk, VectorBackend
from app.agent.rag.embedder import Embedder
from app.config.settings import settings

__all__ = ["KnowledgeRetriever", "RetrievedChunk"]


class KnowledgeRetriever:
    """对上层暴露统一接口，对下委托给具体 backend。"""

    def __init__(
        self,
        embedder: Embedder,
        backend: VectorBackend,
        reranker=None,
    ):
        self._embedder = embedder
        self._backend = backend
        self._loaded = False
        # None = 按 settings 懒构造；显式传入（含测试桩）则用传入的
        self._reranker = reranker
        self._reranker_ready = reranker is not None
        self.last_reranked = False

    @property
    def backend(self) -> VectorBackend:
        return self._backend

    @property
    def size(self) -> int:
        return self._backend.size()

    def load(self) -> None:
        if self._loaded:
            return
        self._backend.load()

        expected = self._backend.expected_embedding_model()
        if expected and expected != self._embedder.model:
            raise ValueError(
                f"索引模型({expected}) 与当前 Embedder 模型"
                f"({self._embedder.model}) 不一致，请重建索引。"
            )
        self._loaded = True

    def _ensure_reranker(self):
        if self._reranker_ready:
            return self._reranker
        self._reranker_ready = True
        if not settings.rerank_enabled:
            self._reranker = None
            return None
        from app.agent.rag.reranker import HttpReranker
        self._reranker = HttpReranker.from_settings()
        return self._reranker

    def search(self, query: str, top_k: int = 3) -> list[RetrievedChunk]:
        if not self._loaded:
            self.load()
        q_vec = self._embedder.encode_one(query)
        top_k = max(1, top_k)
        recall_k = top_k
        if settings.rerank_enabled:
            recall_k = max(top_k, settings.rerank_recall_k)
        hits = self._backend.search_hybrid(query, q_vec, top_k=recall_k)
        return self._maybe_rerank(query, hits, top_k)

    def _maybe_rerank(
        self, query: str, hits: list[RetrievedChunk], top_k: int,
    ) -> list[RetrievedChunk]:
        self.last_reranked = False
        if not hits:
            return []
        if not settings.rerank_enabled or len(hits) <= 1:
            return hits[:top_k]
        reranker = self._ensure_reranker()
        if reranker is None:
            return hits[:top_k]
        try:
            texts = [h.chunk.text for h in hits]
            ranked = reranker.rerank(query, texts, top_n=top_k)
        except Exception as e:  # noqa: BLE001 —— fail-open
            from app.agent.rag.reranker import warn_fail_open
            warn_fail_open(f"{type(e).__name__}: {e}")
            return hits[:top_k]
        if not ranked:
            return hits[:top_k]
        out: list[RetrievedChunk] = []
        seen: set[int] = set()
        for idx, score in ranked:
            if idx in seen or idx < 0 or idx >= len(hits):
                continue
            seen.add(idx)
            h = hits[idx]
            out.append(RetrievedChunk(chunk=h.chunk, score=float(score)))
            if len(out) >= top_k:
                break
        self.last_reranked = bool(out)
        return out or hits[:top_k]
