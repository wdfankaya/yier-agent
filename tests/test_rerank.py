"""rerank：召回原序被 cross-encoder 重排；失败 fail-open。

用法：python tests/test_rerank.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from app.agent.rag.backends.base import RetrievedChunk  # noqa: E402
from app.agent.rag.chunker import Chunk  # noqa: E402
from app.agent.rag.retriever import KnowledgeRetriever  # noqa: E402
from app.agent.rag.reranker import HttpReranker  # noqa: E402
from app.config.settings import settings  # noqa: E402


def _ok(msg: str):
    print(f"  ✅ {msg}")


def _fail(msg: str):
    print(f"  ❌ {msg}")
    sys.exit(1)


def _chunks(n: int = 5) -> list[RetrievedChunk]:
    out = []
    for i in range(n):
        c = Chunk(
            chunk_id=f"c{i}",
            doc=f"doc-{i}",
            section=f"s{i}",
            text=f"正文{i} " * 8,
        )
        out.append(RetrievedChunk(chunk=c, score=1.0 - i * 0.1))
    return out


class _FakeEmbedder:
    model = "fake-emb"

    def encode_one(self, text: str) -> list[float]:
        return [0.1, 0.2]


class _FakeBackend:
    supports_hybrid = False

    def __init__(self, hits: list[RetrievedChunk]):
        self._hits = hits

    def load(self) -> None:
        return None

    def expected_embedding_model(self) -> str:
        return "fake-emb"

    def size(self) -> int:
        return len(self._hits)

    def search_hybrid(self, query, q_vec, top_k: int):
        return list(self._hits[:top_k])

    def search(self, query_vector, top_k: int):
        return list(self._hits[:top_k])

    def upsert(self, *args, **kwargs):
        raise NotImplementedError


class _ReverseReranker:
    def rerank(self, query, texts, top_n: int):
        order = list(range(len(texts)))[::-1]
        return [(i, float(len(texts) - k)) for k, i in enumerate(order[:top_n])]


class _BoomReranker:
    def rerank(self, query, texts, top_n: int):
        raise RuntimeError("rerank down")


def test_rerank_reorders():
    print("\n[1/5] 重排改变 top-3 顺序")
    old = (settings.rerank_enabled, settings.rerank_recall_k)
    settings.rerank_enabled = True
    settings.rerank_recall_k = 5
    try:
        hits = _chunks(5)
        r = KnowledgeRetriever(
            embedder=_FakeEmbedder(),
            backend=_FakeBackend(hits),
            reranker=_ReverseReranker(),
        )
        out = r.search("q", top_k=3)
        names = [h.chunk.doc for h in out]
        if names != ["doc-4", "doc-3", "doc-2"]:
            _fail(f"期望倒序 top3，实际 {names}")
        if not r.last_reranked:
            _fail("last_reranked 应为 True")
        _ok("召回 doc-0..4，重排后 top3=doc-4,3,2")
    finally:
        settings.rerank_enabled, settings.rerank_recall_k = old


def test_disabled_keeps_order():
    print("\n[2/5] 关闭 rerank 保持召回原序")
    old = settings.rerank_enabled
    settings.rerank_enabled = False
    try:
        r = KnowledgeRetriever(
            embedder=_FakeEmbedder(),
            backend=_FakeBackend(_chunks(5)),
            reranker=_ReverseReranker(),
        )
        names = [h.chunk.doc for h in r.search("q", top_k=3)]
        if names != ["doc-0", "doc-1", "doc-2"]:
            _fail(f"关闭后应保持原序，实际 {names}")
        if r.last_reranked:
            _fail("关闭时不应标记 reranked")
        _ok("RERANK_ENABLED=false → 原序 top-3")
    finally:
        settings.rerank_enabled = old


def test_fail_open():
    print("\n[3/5] rerank 抛错 fail-open")
    old = (settings.rerank_enabled, settings.rerank_recall_k)
    settings.rerank_enabled = True
    settings.rerank_recall_k = 5
    try:
        r = KnowledgeRetriever(
            embedder=_FakeEmbedder(),
            backend=_FakeBackend(_chunks(5)),
            reranker=_BoomReranker(),
        )
        names = [h.chunk.doc for h in r.search("q", top_k=3)]
        if names != ["doc-0", "doc-1", "doc-2"]:
            _fail(f"失败应回落原序，实际 {names}")
        if r.last_reranked:
            _fail("失败不应标记 reranked")
        _ok("异常时回落召回原序，检索不中断")
    finally:
        settings.rerank_enabled, settings.rerank_recall_k = old


def test_http_parse():
    print("\n[4/5] HttpReranker 解析 SiliconFlow 风格 JSON")
    rr = HttpReranker(api_key="k", base_url="https://example.com/v1", model="m")

    class _Resp:
        def read(self):
            return json.dumps({
                "results": [
                    {"index": 2, "relevance_score": 0.9},
                    {"index": 0, "relevance_score": 0.2},
                ]
            }).encode()

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    import app.agent.rag.reranker as mod

    def _urlopen(req, timeout=0):
        if "/rerank" not in req.full_url:
            _fail(f"URL 应为 .../rerank，实际 {req.full_url}")
        return _Resp()

    old = mod.urllib.request.urlopen
    mod.urllib.request.urlopen = _urlopen
    try:
        ranked = rr.rerank("q", ["a", "b", "c"], top_n=2)
        if ranked != [(2, 0.9), (0, 0.2)]:
            _fail(f"解析不对: {ranked}")
        _ok("results[].index / relevance_score 解析正确")
    finally:
        mod.urllib.request.urlopen = old


def test_tool_flag():
    print("\n[5/5] search_knowledge 带 reranked 标记")
    from app.agent.tools import knowledge as knowledge_tool

    old = (
        settings.rerank_enabled,
        settings.rerank_recall_k,
        settings.rag_backend,
    )
    settings.rerank_enabled = True
    settings.rerank_recall_k = 5
    hits = _chunks(5)
    retriever = KnowledgeRetriever(
        embedder=_FakeEmbedder(),
        backend=_FakeBackend(hits),
        reranker=_ReverseReranker(),
    )
    knowledge_tool._retriever = retriever
    try:
        data = knowledge_tool.search_knowledge("七天无理由", top_k=3)
        if not data.get("success") or not data.get("reranked"):
            _fail(f"期望 reranked=true，实际 {data}")
        if data["results"][0]["doc"] != "doc-4":
            _fail(f"工具结果未走重排: {data['results']}")
        _ok("工具结果 reranked=true 且顺序已重排")
    finally:
        knowledge_tool.reset_retriever()
        settings.rerank_enabled, settings.rerank_recall_k, settings.rag_backend = old


def main():
    print("=" * 60)
    print("rerank")
    print("=" * 60)
    test_rerank_reorders()
    test_disabled_keeps_order()
    test_fail_open()
    test_http_parse()
    test_tool_flag()
    print("\n全部通过。")


if __name__ == "__main__":
    main()
