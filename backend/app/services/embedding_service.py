"""Single entry point for producing embeddings.

Resolves the active backend once and exposes a uniform async API:

    svc = await build_embedding_service(db, organization, organization_settings)
    vectors = await svc.embed_texts(["revenue by region", ...])
    svc.model_id, svc.dim   # provenance + dimension guard

Selection order:
  1. org-configured API embedding model (an ``LLMModel`` with
     ``model_type == "embedding"``), unless the deployment forces local-only;
  2. else the local fastembed model (default, no API key, offline).

The local model is CPU-bound, so it runs in a thread executor and never blocks
the event loop. ``embed_texts`` always returns vectors — local fallback means
it effectively never returns ``None``.
"""

from __future__ import annotations

import asyncio
from typing import List, Optional

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.ai.embeddings import local_embedder
from app.ai.llm.llm import LLM
from app.models.llm_model import LLMModel
from app.models.organization import Organization
from app.settings.logging_config import get_logger

logger = get_logger(__name__)

# organization_settings.config keys (stored as plain values via set_config).
SETTING_DEFAULT_EMBEDDING_MODEL_ID = "default_embedding_model_id"
SETTING_EMBEDDINGS_LOCAL_ONLY = "embeddings_local_only"


class EmbeddingService:
    """Embeds text via either an API model (``llm``) or the local model."""

    def __init__(self, *, llm: Optional[LLM] = None, dim: Optional[int] = None):
        self._llm = llm
        if llm is not None:
            self.model_id = llm.model_id
            self.dim = int(dim) if dim else 0
            self._local = None
        else:
            self._local = local_embedder.LocalEmbedder.instance()
            self.model_id = self._local.model_id
            self.dim = self._local.dim

    @property
    def is_local(self) -> bool:
        return self._llm is None

    async def embed_texts(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        if self._llm is not None:
            return await self._llm.embed(texts)
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self._local.embed, texts)

    async def embed_query(self, text: str) -> List[float]:
        vecs = await self.embed_texts([text])
        return vecs[0] if vecs else []


def _get_setting(organization_settings, key, default=None):
    if organization_settings is None:
        return default
    try:
        value = organization_settings.get_config(key, default)
    except Exception:
        return default
    return value if value is not None else default


async def _resolve_embedding_model(
    db: AsyncSession, organization: Organization, organization_settings
) -> Optional[LLMModel]:
    """Find the org's active API embedding model, or None for local."""
    if bool(_get_setting(organization_settings, SETTING_EMBEDDINGS_LOCAL_ONLY, False)):
        return None

    configured_id = _get_setting(organization_settings, SETTING_DEFAULT_EMBEDDING_MODEL_ID)
    if configured_id:
        row = await db.execute(
            select(LLMModel).where(
                LLMModel.id == configured_id,
                LLMModel.organization_id == organization.id,
                LLMModel.is_enabled == True,  # noqa: E712
            )
        )
        model = row.scalar_one_or_none()
        if model is not None:
            return model

    # Fall back to the first enabled embedding model for the org.
    row = await db.execute(
        select(LLMModel)
        .where(
            LLMModel.organization_id == organization.id,
            LLMModel.model_type == "embedding",
            LLMModel.is_enabled == True,  # noqa: E712
        )
        .limit(1)
    )
    return row.scalar_one_or_none()


async def build_embedding_service(
    db: AsyncSession,
    organization: Organization,
    organization_settings=None,
) -> EmbeddingService:
    """Build an ``EmbeddingService`` bound to the org's active backend.

    Never raises for "no model configured" — that's the local default. Only an
    explicit failure to construct a configured API client falls back to local
    (with a warning), so retrieval keeps working.
    """
    try:
        model = await _resolve_embedding_model(db, organization, organization_settings)
    except Exception as exc:
        logger.warning("Embedding model resolution failed; using local: %s", exc)
        model = None

    if model is None:
        return EmbeddingService()

    try:
        llm = LLM(model)
        dim = model.get_embedding_dim()
        if not dim:
            logger.warning(
                "Embedding model %s has no known dimension; using local instead",
                model.model_id,
            )
            return EmbeddingService()
        return EmbeddingService(llm=llm, dim=dim)
    except Exception as exc:
        logger.warning(
            "Failed to build API embedding client (%s); using local: %s",
            getattr(model, "model_id", "?"), exc,
        )
        return EmbeddingService()
