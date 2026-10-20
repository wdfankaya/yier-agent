"""Versioned user memory with provenance and transactional merge.

Revision ID: e9b1c2d3e4f5
Revises: d6a1b2c3d4e5
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB

revision = "e9b1c2d3e4f5"
down_revision = "d6a1b2c3d4e5"
branch_labels = None
depends_on = None

def upgrade():
    op.create_table("memory_states_v2",
                    sa.Column("user_id", sa.String(64), primary_key=True),
                    sa.Column("payload", JSONB, nullable=False))

def downgrade():
    op.drop_table("memory_states_v2")
