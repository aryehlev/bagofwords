"""Polymorphic embedding storage for semantic retrieval.

One row per (owner, active embedding model). Keeping embeddings in a dedicated
polymorphic table (rather than columns on ``instructions``/``steps``) isolates
the dialect-specific vector column type to one place and keeps re-embedding and
multi-owner support clean.

Physical column types are created by the migration, which branches on dialect:
  - Postgres : ``embedding`` is pgvector ``vector(dim)`` with a cosine ANN index.
  - SQLite   : ``embedding`` is ``F32_BLOB(dim)`` (BLOB affinity); the vector
               functions (``vector_distance_cos``/``vector32``) are provided by
               the libSQL engine used in :mod:`app.ai.context.vector_store`.

The ORM column type below is dialect-aware so the model stays self-consistent,
but actual vector reads/writes go through raw SQL in ``VectorStore`` (the
distance operators are not expressible through the ORM portably).
"""

from __future__ import annotations

from sqlalchemy import Column, String, Integer, Index, UniqueConstraint, LargeBinary
from sqlalchemy.types import TypeDecorator

from app.models.base import BaseSchema

# Default vector width — matches the local fastembed model (bge-small, 384-dim).
# The migration makes the column dimension configurable, defaulting here.
DEFAULT_EMBEDDING_DIM = 384

OWNER_INSTRUCTION = "instruction"
OWNER_STEP = "step"


class Vector(TypeDecorator):
    """Dialect-aware vector column: pgvector on Postgres, BLOB elsewhere."""

    impl = LargeBinary
    cache_ok = True

    def __init__(self, dim: int = DEFAULT_EMBEDDING_DIM):
        self.dim = dim
        super().__init__()

    def load_dialect_impl(self, dialect):
        if dialect.name == "postgresql":
            from pgvector.sqlalchemy import Vector as PGVector

            return dialect.type_descriptor(PGVector(self.dim))
        return dialect.type_descriptor(LargeBinary())


class Embedding(BaseSchema):
    __tablename__ = "embeddings"

    organization_id = Column(String, nullable=False, index=True)
    owner_type = Column(String, nullable=False)  # 'instruction' | 'step'
    owner_id = Column(String, nullable=False)    # instructions.id / steps.id
    content_hash = Column(String, nullable=False)  # sha256 of embedded text
    model_id = Column(String, nullable=False)      # provenance
    dim = Column(Integer, nullable=False)          # dimension guard
    embedding = Column(Vector(DEFAULT_EMBEDDING_DIM), nullable=False)

    __table_args__ = (
        UniqueConstraint(
            "owner_type", "owner_id", "model_id", name="uq_embeddings_owner_model"
        ),
        Index("ix_embeddings_lookup", "organization_id", "owner_type", "model_id"),
    )
