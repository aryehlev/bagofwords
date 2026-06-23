"""Dialect-aware vector storage + similarity search for the `embeddings` table.

Two backends behind one interface:

  - ``PgVectorStore``    — Postgres/pgvector. Runs on the request's AsyncSession;
                           cosine distance via the ``<=>`` operator.
  - ``LibsqlVectorStore``— SQLite/Turso. Uses the dedicated *sync* libSQL engine
                           (``create_libsql_vector_engine``) inside
                           ``run_in_executor`` so it never blocks the loop;
                           cosine distance via ``vector_distance_cos``.

Both use raw SQL — the distance operators aren't portably expressible through
the ORM. Callers get ``None`` from query/upsert when the backend is unavailable
(e.g. libSQL driver missing) so semantic search degrades to literal ranking.

Scores are similarity in ``[0, 1]`` (``1 - cosine_distance``), higher = closer.
"""

from __future__ import annotations

import asyncio
import json
import uuid
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Sequence, Tuple

from sqlalchemy import bindparam, text
from sqlalchemy.ext.asyncio import AsyncSession

from app.settings.logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class EmbeddingRow:
    organization_id: str
    owner_type: str
    owner_id: str
    content_hash: str
    model_id: str
    dim: int
    vector: List[float]


def _vec_to_json(vec: Sequence[float]) -> str:
    """Serialize a vector to the ``[a,b,c]`` JSON form both engines accept."""
    return "[" + ",".join(repr(float(x)) for x in vec) + "]"


class VectorStore:
    async def upsert(self, rows: List[EmbeddingRow]) -> None:
        raise NotImplementedError

    async def existing_hashes(
        self, organization_id: str, owner_type: str, owner_ids: List[str], model_id: str
    ) -> Dict[str, str]:
        raise NotImplementedError

    async def query(
        self,
        *,
        organization_id: str,
        owner_type: str,
        query_vector: List[float],
        model_id: str,
        dim: int,
        top_k: int,
        owner_ids: Optional[List[str]] = None,
    ) -> Optional[List[Tuple[str, float]]]:
        raise NotImplementedError


class PgVectorStore(VectorStore):
    """pgvector-backed store running on the request's async session."""

    def __init__(self, db: AsyncSession):
        self.db = db

    async def upsert(self, rows: List[EmbeddingRow]) -> None:
        if not rows:
            return
        now = datetime.utcnow()
        stmt = text(
            """
            INSERT INTO embeddings
                (id, organization_id, owner_type, owner_id, content_hash,
                 model_id, dim, embedding, created_at, updated_at)
            VALUES
                (:id, :org, :owner_type, :owner_id, :content_hash,
                 :model_id, :dim, CAST(:vec AS vector), :now, :now)
            ON CONFLICT (owner_type, owner_id, model_id) DO UPDATE SET
                content_hash = EXCLUDED.content_hash,
                dim = EXCLUDED.dim,
                embedding = EXCLUDED.embedding,
                updated_at = EXCLUDED.updated_at
            """
        )
        for r in rows:
            await self.db.execute(
                stmt,
                {
                    "id": str(uuid.uuid4()),
                    "org": r.organization_id,
                    "owner_type": r.owner_type,
                    "owner_id": r.owner_id,
                    "content_hash": r.content_hash,
                    "model_id": r.model_id,
                    "dim": r.dim,
                    "vec": _vec_to_json(r.vector),
                    "now": now,
                },
            )
        await self.db.commit()

    async def existing_hashes(
        self, organization_id: str, owner_type: str, owner_ids: List[str], model_id: str
    ) -> Dict[str, str]:
        if not owner_ids:
            return {}
        stmt = text(
            """
            SELECT owner_id, content_hash FROM embeddings
            WHERE organization_id = :org AND owner_type = :owner_type
              AND model_id = :model_id AND owner_id IN :owner_ids
            """
        ).bindparams(bindparam("owner_ids", expanding=True))
        rows = (
            await self.db.execute(
                stmt,
                {
                    "org": organization_id,
                    "owner_type": owner_type,
                    "model_id": model_id,
                    "owner_ids": owner_ids,
                },
            )
        ).all()
        return {oid: h for oid, h in rows}

    async def query(
        self,
        *,
        organization_id: str,
        owner_type: str,
        query_vector: List[float],
        model_id: str,
        dim: int,
        top_k: int,
        owner_ids: Optional[List[str]] = None,
    ) -> Optional[List[Tuple[str, float]]]:
        params = {
            "org": organization_id,
            "owner_type": owner_type,
            "model_id": model_id,
            "dim": dim,
            "vec": _vec_to_json(query_vector),
            "k": top_k,
        }
        owner_clause = ""
        bind_extra = []
        if owner_ids is not None:
            if not owner_ids:
                return []
            owner_clause = " AND owner_id IN :owner_ids"
            params["owner_ids"] = owner_ids
            bind_extra.append(bindparam("owner_ids", expanding=True))
        stmt = text(
            f"""
            SELECT owner_id,
                   1 - (embedding <=> CAST(:vec AS vector)) AS score
            FROM embeddings
            WHERE organization_id = :org AND owner_type = :owner_type
              AND model_id = :model_id AND dim = :dim{owner_clause}
            ORDER BY embedding <=> CAST(:vec AS vector)
            LIMIT :k
            """
        )
        if bind_extra:
            stmt = stmt.bindparams(*bind_extra)
        rows = (await self.db.execute(stmt, params)).all()
        return [(oid, float(score)) for oid, score in rows]


class LibsqlVectorStore(VectorStore):
    """libSQL-backed store; all SQL runs on a sync engine off the event loop."""

    def __init__(self, engine):
        self.engine = engine

    async def _run(self, fn):
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, fn)

    async def upsert(self, rows: List[EmbeddingRow]) -> None:
        if not rows:
            return

        def _do():
            now = datetime.utcnow()
            stmt = text(
                """
                INSERT INTO embeddings
                    (id, organization_id, owner_type, owner_id, content_hash,
                     model_id, dim, embedding, created_at, updated_at)
                VALUES
                    (:id, :org, :owner_type, :owner_id, :content_hash,
                     :model_id, :dim, vector32(:vec), :now, :now)
                ON CONFLICT (owner_type, owner_id, model_id) DO UPDATE SET
                    content_hash = excluded.content_hash,
                    dim = excluded.dim,
                    embedding = excluded.embedding,
                    updated_at = excluded.updated_at
                """
            )
            with self.engine.begin() as conn:
                for r in rows:
                    conn.execute(
                        stmt,
                        {
                            "id": str(uuid.uuid4()),
                            "org": r.organization_id,
                            "owner_type": r.owner_type,
                            "owner_id": r.owner_id,
                            "content_hash": r.content_hash,
                            "model_id": r.model_id,
                            "dim": r.dim,
                            "vec": _vec_to_json(r.vector),
                            "now": now,
                        },
                    )

        await self._run(_do)

    async def existing_hashes(
        self, organization_id: str, owner_type: str, owner_ids: List[str], model_id: str
    ) -> Dict[str, str]:
        if not owner_ids:
            return {}

        def _do():
            placeholders = ",".join(f":id{i}" for i in range(len(owner_ids)))
            params = {f"id{i}": oid for i, oid in enumerate(owner_ids)}
            params.update({"org": organization_id, "owner_type": owner_type, "model_id": model_id})
            with self.engine.connect() as conn:
                rows = conn.execute(
                    text(
                        f"""
                        SELECT owner_id, content_hash FROM embeddings
                        WHERE organization_id = :org AND owner_type = :owner_type
                          AND model_id = :model_id AND owner_id IN ({placeholders})
                        """
                    ),
                    params,
                ).all()
            return {oid: h for oid, h in rows}

        return await self._run(_do)

    async def query(
        self,
        *,
        organization_id: str,
        owner_type: str,
        query_vector: List[float],
        model_id: str,
        dim: int,
        top_k: int,
        owner_ids: Optional[List[str]] = None,
    ) -> Optional[List[Tuple[str, float]]]:
        if owner_ids is not None and not owner_ids:
            return []

        def _do():
            params = {
                "org": organization_id,
                "owner_type": owner_type,
                "model_id": model_id,
                "dim": dim,
                "vec": _vec_to_json(query_vector),
                "k": top_k,
            }
            owner_clause = ""
            if owner_ids is not None:
                placeholders = ",".join(f":id{i}" for i in range(len(owner_ids)))
                owner_clause = f" AND owner_id IN ({placeholders})"
                params.update({f"id{i}": oid for i, oid in enumerate(owner_ids)})
            with self.engine.connect() as conn:
                rows = conn.execute(
                    text(
                        f"""
                        SELECT owner_id,
                               1 - vector_distance_cos(embedding, vector32(:vec)) AS score
                        FROM embeddings
                        WHERE organization_id = :org AND owner_type = :owner_type
                          AND model_id = :model_id AND dim = :dim{owner_clause}
                        ORDER BY vector_distance_cos(embedding, vector32(:vec)) ASC
                        LIMIT :k
                        """
                    ),
                    params,
                ).all()
            return [(oid, float(score)) for oid, score in rows]

        try:
            return await self._run(_do)
        except Exception as exc:
            logger.warning("libSQL vector query failed; falling back: %s", exc)
            return None


def get_vector_store(db: AsyncSession) -> Optional[VectorStore]:
    """Pick the vector store for the active dialect, or None if unavailable.

    Postgres → pgvector on the request session. SQLite → the dedicated libSQL
    engine (None if its driver isn't installed → caller falls back to Jaccard).
    """
    try:
        dialect = db.get_bind().dialect.name
    except Exception:
        dialect = ""
    if dialect == "postgresql":
        return PgVectorStore(db)
    # SQLite / libSQL path.
    from app.settings.database import create_libsql_vector_engine

    engine = create_libsql_vector_engine()
    if engine is None:
        return None
    return LibsqlVectorStore(engine)
