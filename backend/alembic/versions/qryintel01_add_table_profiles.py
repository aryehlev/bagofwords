"""add table_profiles (query-intelligence data profiles)

Stores the learned data profile per (connection, table): value dictionaries for
low-cardinality columns, per-column null/distinct estimates, inferred join keys
and effective uniqueness, and a rolling runtime-cost summary fed back from real
query timings. Advisory only — consumed by the coder prompt and the SQL
optimizer; never required for execution. See docs/design/query-intelligence.md.

Revision ID: qryintel01
Revises: ff8803de3eb2
Create Date: 2026-06-25
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = "qryintel01"
down_revision: Union[str, None] = "ff8803de3eb2"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "table_profiles",
        sa.Column("connection_id", sa.String(length=36), nullable=False),
        sa.Column("connection_table_id", sa.String(length=36), nullable=True),
        sa.Column("table_fqn", sa.String(), nullable=False),
        sa.Column("row_count_estimate", sa.Integer(), nullable=True),
        sa.Column("sample_rows", sa.Integer(), nullable=False),
        sa.Column("column_profiles", sa.JSON(), nullable=False),
        sa.Column("value_dictionaries", sa.JSON(), nullable=False),
        sa.Column("unique_columns", sa.JSON(), nullable=False),
        sa.Column("learned_join_keys", sa.JSON(), nullable=False),
        sa.Column("cost_summary", sa.JSON(), nullable=True),
        sa.Column("profile_version", sa.Integer(), nullable=False),
        sa.Column("status", sa.String(), nullable=False),
        sa.Column("profiled_at", sa.DateTime(), nullable=True),
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=True),
        sa.Column("updated_at", sa.DateTime(), nullable=True),
        sa.Column("deleted_at", sa.DateTime(), nullable=True),
        sa.ForeignKeyConstraint(["connection_id"], ["connections.id"], ),
        sa.ForeignKeyConstraint(["connection_table_id"], ["connection_tables.id"], ),
        sa.PrimaryKeyConstraint("id"),
    )
    with op.batch_alter_table("table_profiles", schema=None) as batch_op:
        batch_op.create_index(batch_op.f("ix_table_profiles_id"), ["id"], unique=True)
        batch_op.create_index(batch_op.f("ix_table_profiles_connection_id"), ["connection_id"], unique=False)
        batch_op.create_index(batch_op.f("ix_table_profiles_connection_table_id"), ["connection_table_id"], unique=False)
        batch_op.create_index("ix_table_profiles_conn_fqn", ["connection_id", "table_fqn"], unique=True)


def downgrade() -> None:
    with op.batch_alter_table("table_profiles", schema=None) as batch_op:
        batch_op.drop_index("ix_table_profiles_conn_fqn")
        batch_op.drop_index(batch_op.f("ix_table_profiles_connection_table_id"))
        batch_op.drop_index(batch_op.f("ix_table_profiles_connection_id"))
        batch_op.drop_index(batch_op.f("ix_table_profiles_id"))
    op.drop_table("table_profiles")
