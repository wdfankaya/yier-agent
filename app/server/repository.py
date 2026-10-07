"""PostgreSQL 数据访问层，基于 SQLAlchemy 异步接口。

提供会话、消息、记忆事实与知识库的读写操作。
异步函数可直接 await；同步入口通过 asyncio.run 在独立事件循环中调用。
同步入口只能在线程内没有运行中事件循环时使用。

引擎使用 NullPool，避免跨事件循环复用 asyncpg 连接。
每次操作都会建立并关闭连接；连接开销是当前实现的性能限制。
"""

from __future__ import annotations

import asyncio
import uuid
from dataclasses import dataclass, field
from typing import Optional

from sqlalchemy import delete, func, select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from sqlalchemy.pool import NullPool

from app.config.settings import settings
from app.server.models import Base, KbChunk, KbMeta, MemoryFact, Message, Session

_engine = None
_session_factory = None


def _get_engine():
    """懒创建：NullPool + 单例。跨 asyncio.run 复用安全（无持久连接）。"""
    global _engine, _session_factory
    if _engine is None:
        _engine = create_async_engine(settings.database_url, poolclass=NullPool)
        _session_factory = async_sessionmaker(_engine, expire_on_commit=False)
    return _engine


def _get_session_factory():
    """确保 engine/factory 已初始化并返回 factory（懒初始化入口）。"""
    _get_engine()
    return _session_factory


async def aclose_engine() -> None:
    """异步释放引擎（lifespan shutdown 在事件循环线程里 await 这个）。

    不能把旧的 sync dispose（内部 asyncio.run）放进 lifespan：asyncio.run 在
    【已有运行中事件循环】的线程里调用会直接报错。故拆成两面。
    """
    global _engine, _session_factory
    if _engine is not None:
        await _engine.dispose()
        _engine = None
        _session_factory = None


def dispose_engine() -> None:
    """同步便捷层：在 worker 线程 / 无事件循环处释放引擎。"""
    if _engine is not None:
        _run(aclose_engine())


# ---------------------------------------------------------------------------
# 领域值对象
# ---------------------------------------------------------------------------

@dataclass
class SessionState:
    """一个会话在 DB 里的完整状态（load 后可直接喂给 agent）。"""

    session_id: uuid.UUID
    session_key: str
    user_id: str
    summary: Optional[str] = None
    stm: Optional[dict] = None
    messages: list[dict] = field(default_factory=list)


# ---------------------------------------------------------------------------
# async 核心函数（真正做 SQL 的部分）
# ---------------------------------------------------------------------------

async def get_or_create_session_id(user_id: str, session_key: str) -> uuid.UUID:
    """按 (user_id, session_key) 找会话；没有则新建，返回内部 UUID。"""
    async with _get_session_factory()() as db:
        row = await db.execute(
            select(Session.id).where(
                Session.user_id == user_id, Session.session_key == session_key
            )
        )
        sid = row.scalar_one_or_none()
        if sid is not None:
            return sid
        s = Session(user_id=user_id, session_key=session_key)
        db.add(s)
        await db.commit()
        return s.id


async def replace_session_state(
    session_id: uuid.UUID, messages: list[dict],
    summary: Optional[str], stm: Optional[dict],
) -> None:
    """整窗重写：消息全删全插 + 更新 summary/stm（一个事务）。

    语义与旧 JSON「save_session 整包覆盖」等价，因此能正确处理 _compress_history
    把旧消息压缩成 summary 的情况（被压缩掉的消息从表中消失，历史 = 当前窗口 + summary）。
    消息逐行落库 → 崩溃不丢。
    """
    async with _get_session_factory()() as db:
        await db.execute(delete(Message).where(Message.session_id == session_id))
        for m in messages:
            db.add(Message(
                session_id=session_id,
                role=m.get("role", ""),
                content=m.get("content"),
                tool_call_id=m.get("tool_call_id"),
                tool_calls=m.get("tool_calls"),
            ))
        await db.execute(
            Session.__table__.update().where(Session.id == session_id).values(
                summary=summary,
                stm=stm,
                updated_at=func.now(),
            )
        )
        await db.commit()


async def load_session_state(user_id: str, session_key: str) -> Optional[SessionState]:
    """取会话行 + 按插入序取全部消息。会话不存在返回 None。"""
    async with _get_session_factory()() as db:
        s = await db.execute(
            select(Session).where(Session.user_id == user_id, Session.session_key == session_key)
        )
        session = s.scalar_one_or_none()
        if session is None:
            return None
        rows = await db.execute(
            select(Message).where(Message.session_id == session.id).order_by(Message.id)
        )
        messages = [_message_to_dict(m) for m in rows.scalars()]
        return SessionState(
            session_id=session.id,
            session_key=session.session_key,
            user_id=session.user_id,
            summary=session.summary,
            stm=session.stm,
            messages=messages,
        )


async def find_session_owner(session_key: str) -> Optional[str]:
    """GET/DELETE 接口只有 session_id、没有 user_id：跨用户找最近活跃的归属用户。

    session_key 跨用户不唯一，这里取最近更新的一条 —— 服务端进程重启后靠它恢复会话。
    """
    async with _get_session_factory()() as db:
        row = await db.execute(
            select(Session.user_id)
            .where(Session.session_key == session_key)
            .order_by(Session.updated_at.desc())
            .limit(1)
        )
        return row.scalar_one_or_none()


async def delete_session_state(user_id: str, session_key: str) -> bool:
    """删会话行（messages 靠外键级联清空）。返回是否存在。"""
    async with _get_session_factory()() as db:
        s = await db.execute(
            select(Session).where(Session.user_id == user_id, Session.session_key == session_key)
        )
        session = s.scalar_one_or_none()
        if session is None:
            return False
        await db.delete(session)
        await db.commit()
        return True


def _message_to_dict(m: Message) -> dict:
    d: dict = {"role": m.role, "content": m.content}
    if m.tool_call_id:
        d["tool_call_id"] = m.tool_call_id
    if m.tool_calls is not None:
        d["tool_calls"] = m.tool_calls
    return d


# ---------------------------------------------------------------------------
# memory_facts（kind 区分 'fact' / 'session_summary'；UNIQUE 约束去重）
# ---------------------------------------------------------------------------

async def upsert_facts(
    user_id: str, kind: str,
    items: list[dict],  # [{category, content}]
    keep_max: Optional[int] = None,
) -> None:
    """插入新事实；重复的（user_id, kind, content）被 UNIQUE 约束吞掉，不报错。"""
    if not items:
        return
    async with _get_session_factory()() as db:
        stmt = pg_insert(MemoryFact).values([
            {"user_id": user_id, "kind": kind, "category": it.get("category", "other"),
             "content": it["content"]}
            for it in items
        ])
        stmt = stmt.on_conflict_do_nothing(
            constraint="uq_memory_facts_user_kind_content"
        )
        await db.execute(stmt)
        if keep_max is not None:
            # 只保留最新的 keep_max 条（对齐旧版 max_facts 裁剪）
            await _prune_facts(db, user_id, kind, keep_max)
        await db.commit()


async def _prune_facts(db, user_id: str, kind: str, keep_max: int) -> None:
    subq = (
        select(MemoryFact.id)
        .where(MemoryFact.user_id == user_id, MemoryFact.kind == kind)
        .order_by(MemoryFact.id.desc())
        .offset(keep_max)
    )
    await db.execute(delete(MemoryFact).where(MemoryFact.id.in_(subq)))


async def load_facts(user_id: str) -> tuple[list[dict], list[dict]]:
    """返回 (facts, session_summaries)。每个元素 [{category, content, created_at}]。"""
    async with _get_session_factory()() as db:
        rows = await db.execute(
            select(MemoryFact).where(MemoryFact.user_id == user_id).order_by(MemoryFact.id)
        )
        facts, summaries = [], []
        for f in rows.scalars():
            item = {"category": f.category, "content": f.content,
                    "created_at": f.created_at.isoformat() if f.created_at else ""}
            (summaries if f.kind == "session_summary" else facts).append(item)
        return facts, summaries


async def delete_user_memory(user_id: str) -> None:
    """清空某用户全部长期记忆（reset 用）。"""
    async with _get_session_factory()() as db:
        await db.execute(
            delete(MemoryFact).where(MemoryFact.user_id == user_id)
        )
        await db.commit()


# ---------------------------------------------------------------------------
# kb_chunks（pgvector + 全文检索混合检索）
# 向量 + 全文两条 lane 各自独立查询；"如何融合" 是 PgVectorBackend 的职责，
# 这里只保证返回纯数据（dict），不 import agent 层的 Chunk / RetrievedChunk。
# ---------------------------------------------------------------------------

async def kb_replace_all(
    rows: list[dict],  # [{chunk_id, doc, section, text}, ...]
    vectors: list[list[float]],
    embedding_model: str,
) -> int:
    """整表重建 kb_chunks（覆盖式），并把 embedding_model 写进 kb_meta。

    等价 numpy JSON 顶层 / chroma collection.metadata 的语义：换 embedding 后
    旧的向量行必须整体清掉，否则 retriever.load() 的模型校验会挡下一次搜索。
    """
    if len(rows) != len(vectors):
        raise ValueError(f"rows 与 vectors 长度不一致: {len(rows)} vs {len(vectors)}")

    async with _get_session_factory()() as db:
        await db.execute(delete(KbChunk))
        for row, vec in zip(rows, vectors):
            db.add(KbChunk(
                chunk_id=row["chunk_id"],
                doc=row["doc"],
                section=row["section"],
                text=row["text"],
                embedding=vec,
            ))
        await db.execute(
            pg_insert(KbMeta)
            .values(key="embedding_model", value=embedding_model)
            .on_conflict_do_update(
                index_elements=[KbMeta.key],
                set_={"value": pg_insert(KbMeta).excluded.value},
            )
        )
        await db.commit()
    return len(rows)


async def kb_vector_search(
    query_vector: list[float], top_k: int
) -> list[dict]:
    """向量 lane：HNSW 余弦最近邻。score = 1 - cosine_distance（余弦相似度）。"""
    async with _get_session_factory()() as db:
        distance = KbChunk.embedding.cosine_distance(query_vector)
        rows = await db.execute(
            select(KbChunk, (1 - distance).label("score"))
            .order_by(distance.asc())
            .limit(top_k)
        )
        out = []
        for chunk, score in rows.all():
            out.append({
                "chunk_id": chunk.chunk_id,
                "doc": chunk.doc,
                "section": chunk.section,
                "text": chunk.text,
                "score": float(score),
            })
        return out


async def kb_bm25_search(query_text: str, top_k: int) -> list[dict]:
    """全文检索 lane：PG 原生全文检索，ts_rank 排序。

    'simple' 分词按非字母数字切分：英文/数字/连字符 token 精确匹配有效，
    中文连续文本会被切成一个 token → 语义检索几乎无信号（边界见 README）。
    """
    tsvector = func.to_tsvector("simple", KbChunk.text)
    tsquery = func.plainto_tsquery("simple", query_text)
    async with _get_session_factory()() as db:
        rows = await db.execute(
            select(KbChunk, func.ts_rank(tsvector, tsquery).label("rank"))
            .where(tsvector.op("@@")(tsquery))
            .order_by(func.ts_rank(tsvector, tsquery).desc())
            .limit(top_k)
        )
        out = []
        for chunk, rank in rows.all():
            out.append({
                "chunk_id": chunk.chunk_id,
                "doc": chunk.doc,
                "section": chunk.section,
                "text": chunk.text,
                "score": float(rank),
            })
        return out


async def kb_size() -> int:
    """当前已索引的 chunk 数量（0 表示还没 build 过）。"""
    async with _get_session_factory()() as db:
        return (await db.execute(select(func.count()).select_from(KbChunk))).scalar_one()


async def kb_meta_value(key: str) -> str:
    """读 kb_meta；缺 key 返回空串。"""
    async with _get_session_factory()() as db:
        row = await db.execute(select(KbMeta.value).where(KbMeta.key == key))
        val = row.scalar_one_or_none()
        return val or ""


# ---------------------------------------------------------------------------
# sync 便捷层：供 worker 线程（agent / storage / LTM / RAG backend）调用
# ---------------------------------------------------------------------------

def _run(coro):
    """在【没有运行中事件循环】的线程里跑一个 async 协程。"""
    return asyncio.run(coro)


def sync_get_or_create_session(user_id: str, session_key: str) -> uuid.UUID:
    return _run(get_or_create_session_id(user_id, session_key))


def sync_replace_session_state(
    session_id: uuid.UUID, messages: list[dict], summary: Optional[str], stm: Optional[dict],
) -> None:
    _run(replace_session_state(session_id, messages, summary, stm))


def sync_load_session_state(user_id: str, session_key: str) -> Optional[SessionState]:
    return _run(load_session_state(user_id, session_key))


def sync_delete_session_state(user_id: str, session_key: str) -> bool:
    return _run(delete_session_state(user_id, session_key))


def sync_find_session_owner(session_key: str) -> Optional[str]:
    return _run(find_session_owner(session_key))


def sync_upsert_facts(user_id, kind, items, keep_max=None) -> None:
    _run(upsert_facts(user_id, kind, items, keep_max))


def sync_load_facts(user_id: str) -> tuple[list[dict], list[dict]]:
    return _run(load_facts(user_id))


def sync_delete_user_memory(user_id: str) -> None:
    _run(delete_user_memory(user_id))


def sync_kb_replace_all(rows, vectors, embedding_model) -> int:
    return _run(kb_replace_all(rows, vectors, embedding_model))


def sync_kb_vector_search(query_vector, top_k) -> list[dict]:
    return _run(kb_vector_search(query_vector, top_k))


def sync_kb_bm25_search(query_text, top_k) -> list[dict]:
    return _run(kb_bm25_search(query_text, top_k))


def sync_kb_size() -> int:
    return _run(kb_size())


def sync_kb_meta_value(key: str) -> str:
    return _run(kb_meta_value(key))
