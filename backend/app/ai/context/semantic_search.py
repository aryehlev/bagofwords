"""High-level semantic search + indexing helper.

Ties :class:`EmbeddingService` (query/text → vector) to :class:`VectorStore`
(vector → owners by cosine similarity). Designed to **degrade gracefully**:
``rank`` returns ``None`` whenever semantic search can't run (no vector engine,
empty index, or any error) so callers keep their existing Jaccard ranking.

    ss = SemanticSearch(db, organization, organization_settings)
    scores = await ss.rank("revenue by region", owner_type="instruction", top_k=10)
    if scores is None:
        ...  # fall back to literal ranking
"""

from __future__ import annotations

import asyncio
import hashlib
from typing import Dict, List, Optional, Sequence, Tuple

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.ai.context.vector_store import EmbeddingRow, get_vector_store
from app.models.organization import Organization
from app.services.embedding_service import build_embedding_service
from app.settings.logging_config import get_logger

logger = get_logger(__name__)

# Cached session maker for off-request keep-fresh indexing, plus strong refs to
# in-flight tasks so they aren't GC'd mid-run (see asyncio.create_task caveat).
_index_session_maker = None
_PENDING_INDEX_TASKS: set = set()


def _get_index_session_maker():
    global _index_session_maker
    if _index_session_maker is None:
        from app.settings.database import create_async_database_engine_for_indexing

        engine = create_async_database_engine_for_indexing()
        _index_session_maker = async_sessionmaker(
            engine, expire_on_commit=False, class_=AsyncSession
        )
    return _index_session_maker


def schedule_index(
    organization_id: str, owner_type: str, items: Sequence[Tuple[str, str]]
) -> None:
    """Fire-and-forget embedding refresh on a fresh background session.

    Keeps request latency unaffected: the embed + upsert run off the response
    path on the dedicated indexing engine. Best-effort — all failures are logged
    and swallowed.
    """
    items = [(oid, txt) for oid, txt in items if oid and (txt or "").strip()]
    if not items:
        return
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return

    async def _run():
        try:
            maker = _get_index_session_maker()
            async with maker() as session:
                org = await session.get(Organization, str(organization_id))
                if org is None:
                    return
                await SemanticSearch(session, org).index_texts(owner_type, items)
        except Exception as exc:
            logger.debug("Background index failed (owner_type=%s): %s", owner_type, exc)

    task = loop.create_task(_run())
    _PENDING_INDEX_TASKS.add(task)
    task.add_done_callback(_PENDING_INDEX_TASKS.discard)


def content_hash(text: str) -> str:
    """Stable hash of embedded text → skip re-embedding when unchanged."""
    return hashlib.sha256((text or "").encode("utf-8")).hexdigest()


class SemanticSearch:
    def __init__(self, db: AsyncSession, organization: Organization, organization_settings=None):
        self.db = db
        self.organization = organization
        self.organization_settings = organization_settings

    async def rank(
        self,
        query: str,
        *,
        owner_type: str,
        top_k: int,
        candidate_ids: Optional[List[str]] = None,
    ) -> Optional[Dict[str, float]]:
        """Return ``{owner_id: similarity}`` for the closest owners, or None.

        ``None`` (fall back) is returned when the vector engine is unavailable,
        the index is empty, or anything errors. An optional ``candidate_ids``
        restricts the search to a known candidate set.
        """
        if not query or not query.strip():
            return None
        store = get_vector_store(self.db)
        if store is None:
            return None
        try:
            svc = await build_embedding_service(
                self.db, self.organization, self.organization_settings
            )
            qvec = await svc.embed_query(query)
            if not qvec:
                return None
            results = await store.query(
                organization_id=str(self.organization.id),
                owner_type=owner_type,
                query_vector=qvec,
                model_id=svc.model_id,
                dim=svc.dim,
                top_k=top_k,
                owner_ids=candidate_ids,
            )
        except Exception as exc:
            logger.warning("Semantic rank failed (owner_type=%s); falling back: %s",
                           owner_type, exc)
            return None
        if not results:
            return None
        return {owner_id: score for owner_id, score in results}

    async def index_texts(
        self,
        owner_type: str,
        items: Sequence[Tuple[str, str]],
    ) -> int:
        """Embed + upsert ``(owner_id, text)`` items, skipping unchanged ones.

        Returns the number of rows (re)embedded. Best-effort: returns 0 and logs
        on any failure so callers (write hooks / backfill) never break.
        """
        items = [(oid, txt) for oid, txt in items if oid and (txt or "").strip()]
        if not items:
            return 0
        store = get_vector_store(self.db)
        if store is None:
            return 0
        try:
            svc = await build_embedding_service(
                self.db, self.organization, self.organization_settings
            )
            owner_ids = [oid for oid, _ in items]
            existing = await store.existing_hashes(
                str(self.organization.id), owner_type, owner_ids, svc.model_id
            )
            hashes = {oid: content_hash(txt) for oid, txt in items}
            changed = [(oid, txt) for oid, txt in items if existing.get(oid) != hashes[oid]]
            if not changed:
                return 0
            vectors = await svc.embed_texts([txt for _, txt in changed])
            if len(vectors) != len(changed):
                raise RuntimeError(
                    f"Embedding count mismatch for owner_type={owner_type}: "
                    f"expected {len(changed)}, got {len(vectors)}"
                )
            rows = [
                EmbeddingRow(
                    organization_id=str(self.organization.id),
                    owner_type=owner_type,
                    owner_id=oid,
                    content_hash=hashes[oid],
                    model_id=svc.model_id,
                    dim=svc.dim,
                    vector=vec,
                )
                for (oid, _), vec in zip(changed, vectors, strict=True)
            ]
            await store.upsert(rows)
            return len(rows)
        except Exception as exc:
            logger.warning("Semantic index_texts failed (owner_type=%s): %s",
                           owner_type, exc)
            return 0
