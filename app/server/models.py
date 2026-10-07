"""SQLAlchemy ORM 模型（三张核心表 + pgvector 知识库表）。

设计要点：
- sessions.id 用 UUID 主键；客户端传来的会话字符串存 session_key，
  UNIQUE(user_id, session_key) 保证「同一用户下会话 id 不重复」
- messages.session_id 外键 ON DELETE CASCADE：清会话 = 删一行 session，消息自动级联清空
- memory_facts 的 UNIQUE(user_id, kind, content) 是去重约束（kind 区分
  'fact' 长期事实 / 'session_summary' 交互摘要，替代原来 JSON 的字符串小写比对）

设计要点（pgvector + 全文混合检索）：
- kb_chunks：知识库 chunk 的向量表，embedding 维度 = bge-m3 的 1024（与索引和迁移定义一致；更换不同维度的模型时需重建此列）。
  向量用 HNSW（vector_cosine_ops）加速余弦最近邻；全文检索用 GIN(to_tsvector('simple', text))。
- kb_meta：键值表，存索引级元数据（embedding_model），等价 NumpyBackend JSON 顶层 /
  Chroma collection.metadata 里持久化的模型标识——加载时校验，防止跨模型混用旧索引。
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Optional

from sqlalchemy import (
    BigInteger,
    DateTime,
    ForeignKey,
    Index,
    String,
    Text,
    UniqueConstraint,
    func,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
from pgvector.sqlalchemy import Vector

# embedding 维度与 DDL 写死的一致（bge-m3 → 1024 维）。
# 换了 embedding 模型要重建 kb_chunks 列 / 索引（需同步重建知识库索引）。
KB_EMBEDDING_DIM = 1024


class Base(DeclarativeBase):
    pass


class Session(Base):
    __tablename__ = "sessions"
    __table_args__ = (UniqueConstraint("user_id", "session_key", name="uq_sessions_user_key"),)

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, server_default=text("gen_random_uuid()")
    )
    session_key: Mapped[str] = mapped_column(String(128), nullable=False)
    user_id: Mapped[str] = mapped_column(String(64), nullable=False, server_default="default")
    summary: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    stm: Mapped[Optional[dict]] = mapped_column(JSONB, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now(), onupdate=func.now()
    )


class Message(Base):
    __tablename__ = "messages"
    __table_args__ = (Index("idx_messages_session", "session_id", "id"),)

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    session_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("sessions.id", ondelete="CASCADE"),
        nullable=False,
    )
    role: Mapped[str] = mapped_column(String(16), nullable=False)
    content: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    tool_call_id: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    tool_calls: Mapped[Optional[dict]] = mapped_column(JSONB, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )


class MemoryFact(Base):
    __tablename__ = "memory_facts"
    __table_args__ = (
        UniqueConstraint("user_id", "kind", "content", name="uq_memory_facts_user_kind_content"),
    )

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    user_id: Mapped[str] = mapped_column(String(64), nullable=False)
    kind: Mapped[str] = mapped_column(String(16), nullable=False, server_default="fact")
    category: Mapped[str] = mapped_column(String(32), nullable=False, server_default="other")
    content: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )


class KbChunk(Base):
    """知识库 chunk 向量表（pgvector）。chunk_id 稳定，便于增量更新。"""

    __tablename__ = "kb_chunks"
    __table_args__ = (
        # 余弦最近邻的近似索引（HNSW）
        Index(
            "idx_kb_chunks_vec",
            "embedding",
            postgresql_using="hnsw",
            postgresql_ops={"embedding": "vector_cosine_ops"},
        ),
        # 全文检索索引：对整段 text 建 tsvector 的 GIN（PG 原生 全文检索 载体）
        Index(
            "idx_kb_chunks_fts",
            text("to_tsvector('simple', text)"),
            postgresql_using="gin",
        ),
    )

    chunk_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    doc: Mapped[str] = mapped_column(String(128), nullable=False)
    section: Mapped[str] = mapped_column(String(256), nullable=False)
    text: Mapped[str] = mapped_column(Text, nullable=False)
    embedding: Mapped[list] = mapped_column(Vector(KB_EMBEDDING_DIM), nullable=False)


class KbMeta(Base):
    """知识库索引级元数据：等价 numpy JSON 顶层 / chroma collection.metadata。

    当前只存一条 embedding_model，供加载时校验「换 embedding 没重建索引」。
    """

    __tablename__ = "kb_meta"

    key: Mapped[str] = mapped_column(String(64), primary_key=True)
    value: Mapped[str] = mapped_column(Text, nullable=False)
