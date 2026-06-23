"""add embeddings table (pgvector / libSQL F32_BLOB)

Revision ID: e4m2b3d5t6p7
Revises: e3m1b2d4t5p6
Create Date: 2026-06-23 00:00:01.000000

Polymorphic embedding storage for semantic retrieval. The vector column type is
dialect-specific, so this migration branches on the bind dialect:

- Postgres: ``CREATE EXTENSION vector``; ``embedding vector(DIM)`` + an hnsw
  cosine index.
- SQLite/Turso: ``embedding F32_BLOB(DIM)`` (BLOB affinity under stock SQLite;
  libSQL interprets it natively). No ANN index initially — full-scan
  ``vector_distance_cos`` is fine at this scale and avoids needing libSQL at
  migration time.

DIM defaults to 384 (local fastembed bge-small) and is overridable at migration
time via ``BOW_EMBEDDING_DIM`` for deployments standardizing on a larger model
(e.g. 1536 for OpenAI text-embedding-3-small). Switching the active model later
requires a re-embed/backfill.
"""
import os
from typing import Sequence, Union

from alembic import op


revision: str = 'e4m2b3d5t6p7'
down_revision: Union[str, None] = 'e3m1b2d4t5p6'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

DIM = int(os.getenv("BOW_EMBEDDING_DIM", "384"))


def upgrade() -> None:
    bind = op.get_bind()
    dialect = bind.dialect.name

    if dialect == 'postgresql':
        op.execute("CREATE EXTENSION IF NOT EXISTS vector")
        op.execute(
            f"""
            CREATE TABLE embeddings (
                id VARCHAR(36) PRIMARY KEY,
                organization_id VARCHAR NOT NULL,
                owner_type VARCHAR NOT NULL,
                owner_id VARCHAR NOT NULL,
                content_hash VARCHAR NOT NULL,
                model_id VARCHAR NOT NULL,
                dim INTEGER NOT NULL,
                embedding vector({DIM}) NOT NULL,
                created_at TIMESTAMP,
                updated_at TIMESTAMP,
                deleted_at TIMESTAMP,
                CONSTRAINT uq_embeddings_owner_model
                    UNIQUE (owner_type, owner_id, model_id)
            )
            """
        )
        op.execute(
            "CREATE INDEX ix_embeddings_organization_id "
            "ON embeddings (organization_id)"
        )
        op.execute(
            "CREATE INDEX ix_embeddings_lookup "
            "ON embeddings (organization_id, owner_type, model_id)"
        )
        op.execute(
            "CREATE INDEX embeddings_embedding_hnsw "
            "ON embeddings USING hnsw (embedding vector_cosine_ops)"
        )
    else:
        # SQLite / libSQL. Stock SQLite parses F32_BLOB(N) as BLOB affinity;
        # libSQL stores/queries it as a native vector.
        op.execute(
            f"""
            CREATE TABLE embeddings (
                id VARCHAR(36) PRIMARY KEY,
                organization_id VARCHAR NOT NULL,
                owner_type VARCHAR NOT NULL,
                owner_id VARCHAR NOT NULL,
                content_hash VARCHAR NOT NULL,
                model_id VARCHAR NOT NULL,
                dim INTEGER NOT NULL,
                embedding F32_BLOB({DIM}) NOT NULL,
                created_at DATETIME,
                updated_at DATETIME,
                deleted_at DATETIME,
                CONSTRAINT uq_embeddings_owner_model
                    UNIQUE (owner_type, owner_id, model_id)
            )
            """
        )
        op.execute(
            "CREATE INDEX ix_embeddings_organization_id "
            "ON embeddings (organization_id)"
        )
        op.execute(
            "CREATE INDEX ix_embeddings_lookup "
            "ON embeddings (organization_id, owner_type, model_id)"
        )


def downgrade() -> None:
    op.execute("DROP TABLE IF EXISTS embeddings")
