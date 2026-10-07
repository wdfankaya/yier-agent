"""kb_chunks + kb_meta（pgvector + 全文检索混合检索）

Revision ID: d6a1b2c3d4e5
Revises: 0066c96ff26d
Create Date: 2026-09-07

前提：DB 里已 `CREATE EXTENSION vector`。官方 postgres:16 镜像不含 pgvector，
需先安装 pgvector 并启用 vector 扩展，也可使用预装 pgvector 的 PostgreSQL 镜像。
embedding 维度 1024 = bge-m3（基线经 SiliconFlow 使用，kb_index.json 即 1024 维）。
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from pgvector.sqlalchemy import Vector

# revision identifiers, used by Alembic.
revision: str = 'd6a1b2c3d4e5'
down_revision: Union[str, Sequence[str], None] = '0066c96ff26d'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.create_table(
        'kb_chunks',
        sa.Column('chunk_id', sa.String(length=64), nullable=False),
        sa.Column('doc', sa.String(length=128), nullable=False),
        sa.Column('section', sa.String(length=256), nullable=False),
        sa.Column('text', sa.Text(), nullable=False),
        sa.Column('embedding', Vector(1024), nullable=False),
        sa.PrimaryKeyConstraint('chunk_id'),
    )
    op.create_table(
        'kb_meta',
        sa.Column('key', sa.String(length=64), nullable=False),
        sa.Column('value', sa.Text(), nullable=False),
        sa.PrimaryKeyConstraint('key'),
    )
    # 余弦最近邻近似索引（HNSW）。与 models.KbChunk.__table_args__ 保持一致。
    op.execute(
        "CREATE INDEX idx_kb_chunks_vec ON kb_chunks "
        "USING hnsw (embedding vector_cosine_ops)"
    )
    # 全文检索 载体：整段 text 的 tsvector('simple') GIN 索引。
    # 与 models.KbChunk.__table_args__ 保持一致。
    op.execute(
        "CREATE INDEX idx_kb_chunks_fts ON kb_chunks "
        "USING gin (to_tsvector('simple', text))"
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_index('idx_kb_chunks_fts', table_name='kb_chunks')
    op.drop_index('idx_kb_chunks_vec', table_name='kb_chunks')
    op.drop_table('kb_meta')
    op.drop_table('kb_chunks')
