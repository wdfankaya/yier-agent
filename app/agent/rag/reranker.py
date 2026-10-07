"""Cross-encoder rerank。

召回（向量 / 混合）只要「候选里有金子」；把金子排到注入 prompt 的 top-3
是 reranker 的事。模型用 `BAAI/bge-reranker-v2-m3`（cross-encoder），
与 embedding 是否同厂无关——reranker 与向量模型独立。

本仓库 embedding 实际是 bge-m3 / 1024（SiliconFlow），不是计划最初写的
OpenAI 1536。因此走方案①的「改动小」分支：不换 embedding、不重建索引，
rerank 打同一家的 `/v1/rerank`（也兼容 Cohere 风格 results[].index）。

失败策略：fail-open。API 挂了 / 超时 / 未配置 key → 退回召回原序，检索不中断。
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Optional, Protocol

from app.config.settings import settings

_warned = False


class Reranker(Protocol):
    def rerank(
        self, query: str, texts: list[str], top_n: int,
    ) -> list[tuple[int, float]]:
        """返回 (原下标, 分数) 按分数降序，长度 ≤ top_n。失败应抛异常。"""


class HttpReranker:
    """OpenAI 兼容网关的 /rerank（SiliconFlow / 同类）。"""

    def __init__(self, api_key: str, base_url: str, model: str, timeout: float = 8.0):
        self.api_key = api_key
        self.model = model
        self.timeout = timeout
        self.url = base_url.rstrip("/") + "/rerank"

    @classmethod
    def from_settings(cls) -> Optional["HttpReranker"]:
        key = (
            settings.rerank_api_key
            or settings.embedding_api_key
            or settings.openai_api_key
        )
        base = (
            settings.rerank_base_url
            or settings.embedding_base_url
            or settings.openai_base_url
        )
        if not key or not base:
            return None
        return cls(api_key=key, base_url=base, model=settings.rerank_model)

    def rerank(
        self, query: str, texts: list[str], top_n: int,
    ) -> list[tuple[int, float]]:
        if not texts:
            return []
        payload = {
            "model": self.model,
            "query": query,
            "documents": texts,
            "top_n": max(1, min(top_n, len(texts))),
            "return_documents": False,
        }
        body = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            self.url,
            data=body,
            method="POST",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        rows = data.get("results") or data.get("data") or []
        out: list[tuple[int, float]] = []
        for row in rows:
            if not isinstance(row, dict):
                continue
            idx = row.get("index")
            score = row.get("relevance_score", row.get("score", 0.0))
            if idx is None:
                continue
            out.append((int(idx), float(score)))
        out.sort(key=lambda x: x[1], reverse=True)
        return out[: max(1, top_n)]


def warn_fail_open(reason: str) -> None:
    global _warned
    if _warned:
        return
    _warned = True
    print(f"⚠️  rerank 失败，回落召回原序: {reason}")
