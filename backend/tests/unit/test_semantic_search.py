"""Unit tests for semantic retrieval plumbing.

These don't touch the DB or any embedding provider — they mock the vector store
and embedding service to verify routing, blending, hashing, and the
graceful-fallback contract (rank → None keeps callers on Jaccard).
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.ai.context import semantic_search as ss_mod
from app.ai.context.semantic_search import PreparedQuery, SemanticSearch, content_hash
from app.ai.context.vector_store import (
    EmbeddingRow,
    LibsqlVectorStore,
    PgVectorStore,
    _vec_to_json,
)
from app.ai.embeddings import local_embedder
from app.services.embedding_service import EmbeddingService


# --- pure helpers -----------------------------------------------------------

def test_content_hash_stable_and_sensitive():
    assert content_hash("hello") == content_hash("hello")
    assert content_hash("hello") != content_hash("world")


def test_vec_to_json_format():
    assert _vec_to_json([1.0, 2.5, -3.0]) == "[1.0,2.5,-3.0]"


# --- EmbeddingService backend selection -------------------------------------

@pytest.mark.asyncio
async def test_embedding_service_api_path_routes_to_llm():
    llm = MagicMock()
    llm.model_id = "text-embedding-3-small"
    llm.embed = AsyncMock(return_value=[[0.1, 0.2]])
    svc = EmbeddingService(llm=llm, dim=1536)
    assert svc.is_local is False
    assert (svc.model_id, svc.dim) == ("text-embedding-3-small", 1536)
    assert await svc.embed_texts(["x"]) == [[0.1, 0.2]]
    llm.embed.assert_awaited_once()


@pytest.mark.asyncio
async def test_embedding_service_local_path(monkeypatch):
    fake = SimpleNamespace(
        model_id="bge-small-en-v1.5",
        dim=384,
        embed=lambda texts: [[0.0] * 384 for _ in texts],
    )
    monkeypatch.setattr(
        local_embedder.LocalEmbedder, "instance", classmethod(lambda cls: fake)
    )
    svc = EmbeddingService()
    assert svc.is_local is True
    assert svc.dim == 384
    vecs = await svc.embed_texts(["a", "b"])
    assert len(vecs) == 2 and len(vecs[0]) == 384


# --- SemanticSearch.rank fallback contract ----------------------------------

@pytest.mark.asyncio
async def test_rank_none_when_store_unavailable(monkeypatch):
    monkeypatch.setattr(ss_mod, "get_vector_store", lambda db: None)
    ss = SemanticSearch(MagicMock(), SimpleNamespace(id="o1"))
    assert await ss.rank("revenue", owner_type="instruction", top_k=5) is None


@pytest.mark.asyncio
async def test_rank_none_on_empty_query(monkeypatch):
    monkeypatch.setattr(ss_mod, "get_vector_store", lambda db: MagicMock())
    ss = SemanticSearch(MagicMock(), SimpleNamespace(id="o1"))
    assert await ss.rank("   ", owner_type="instruction", top_k=5) is None


@pytest.mark.asyncio
async def test_rank_none_when_index_empty(monkeypatch):
    store = MagicMock()
    store.query = AsyncMock(return_value=[])
    monkeypatch.setattr(ss_mod, "get_vector_store", lambda db: store)
    svc = SimpleNamespace(model_id="m", dim=384, embed_query=AsyncMock(return_value=[0.1] * 384))
    monkeypatch.setattr(ss_mod, "build_embedding_service", AsyncMock(return_value=svc))
    ss = SemanticSearch(MagicMock(), SimpleNamespace(id="o1"))
    assert await ss.rank("revenue", owner_type="step", top_k=5) is None


@pytest.mark.asyncio
async def test_rank_returns_scores(monkeypatch):
    store = MagicMock()
    store.query = AsyncMock(return_value=[("a", 0.9), ("b", 0.5)])
    monkeypatch.setattr(ss_mod, "get_vector_store", lambda db: store)
    svc = SimpleNamespace(model_id="m", dim=384, embed_query=AsyncMock(return_value=[0.1] * 384))
    monkeypatch.setattr(ss_mod, "build_embedding_service", AsyncMock(return_value=svc))
    ss = SemanticSearch(MagicMock(), SimpleNamespace(id="o1"))
    out = await ss.rank("revenue", owner_type="instruction", top_k=5, candidate_ids=["a", "b"])
    assert out == {"a": 0.9, "b": 0.5}


# --- SemanticSearch.index_texts (keep-fresh) --------------------------------

@pytest.mark.asyncio
async def test_index_texts_skips_unchanged(monkeypatch):
    store = MagicMock()
    store.existing_hashes = AsyncMock(return_value={"a": content_hash("same")})
    store.upsert = AsyncMock()
    monkeypatch.setattr(ss_mod, "get_vector_store", lambda db: store)
    svc = SimpleNamespace(model_id="m", dim=384, embed_texts=AsyncMock(return_value=[[0.2] * 384]))
    monkeypatch.setattr(ss_mod, "build_embedding_service", AsyncMock(return_value=svc))
    ss = SemanticSearch(MagicMock(), SimpleNamespace(id="o1"))

    n = await ss.index_texts("instruction", [("a", "same"), ("b", "new")])
    assert n == 1  # only the changed/new one
    svc.embed_texts.assert_awaited_once_with(["new"])
    store.upsert.assert_awaited_once()


@pytest.mark.asyncio
async def test_index_texts_purges_stale_model_rows(monkeypatch):
    store = MagicMock()
    store.existing_hashes = AsyncMock(return_value={})
    store.upsert = AsyncMock()
    store.delete_stale_models = AsyncMock()
    monkeypatch.setattr(ss_mod, "get_vector_store", lambda db: store)
    svc = SimpleNamespace(model_id="m2", dim=384, embed_texts=AsyncMock(return_value=[[0.2] * 384]))
    monkeypatch.setattr(ss_mod, "build_embedding_service", AsyncMock(return_value=svc))
    ss = SemanticSearch(MagicMock(), SimpleNamespace(id="o1"))

    n = await ss.index_texts("instruction", [("a", "text")])
    assert n == 1
    # Old-model rows are purged opportunistically after a successful upsert.
    store.delete_stale_models.assert_awaited_once_with("o1", "instruction", "m2")


@pytest.mark.asyncio
async def test_index_texts_survives_purge_failure(monkeypatch):
    store = MagicMock()
    store.existing_hashes = AsyncMock(return_value={})
    store.upsert = AsyncMock()
    store.delete_stale_models = AsyncMock(side_effect=RuntimeError("boom"))
    monkeypatch.setattr(ss_mod, "get_vector_store", lambda db: store)
    svc = SimpleNamespace(model_id="m2", dim=384, embed_texts=AsyncMock(return_value=[[0.2] * 384]))
    monkeypatch.setattr(ss_mod, "build_embedding_service", AsyncMock(return_value=svc))
    ss = SemanticSearch(MagicMock(), SimpleNamespace(id="o1"))

    # Purge failure is best-effort — the indexed count is still reported.
    assert await ss.index_texts("instruction", [("a", "text")]) == 1


# --- prepared-query reuse (one embed per agent turn) -------------------------

@pytest.mark.asyncio
async def test_prepare_embeds_once_and_rank_reuses(monkeypatch):
    store = MagicMock()
    store.query = AsyncMock(return_value=[("a", 0.9)])
    monkeypatch.setattr(ss_mod, "get_vector_store", lambda db: store)
    svc = SimpleNamespace(model_id="m", dim=384, embed_query=AsyncMock(return_value=[0.1] * 384))
    build = AsyncMock(return_value=svc)
    monkeypatch.setattr(ss_mod, "build_embedding_service", build)
    ss = SemanticSearch(MagicMock(), SimpleNamespace(id="o1"))

    prepared = await ss.prepare("revenue")
    assert isinstance(prepared, PreparedQuery)
    assert (prepared.model_id, prepared.dim) == ("m", 384)

    out1 = await ss.rank("revenue", owner_type="step", top_k=5, prepared=prepared)
    out2 = await ss.rank("revenue", owner_type="step", top_k=5, prepared=prepared)
    assert out1 == out2 == {"a": 0.9}
    # The model was resolved and the query embedded exactly once (in prepare).
    build.assert_awaited_once()
    svc.embed_query.assert_awaited_once()
    assert store.query.await_args.kwargs["query_vector"] == prepared.vector


@pytest.mark.asyncio
async def test_prepare_none_when_store_unavailable(monkeypatch):
    monkeypatch.setattr(ss_mod, "get_vector_store", lambda db: None)
    ss = SemanticSearch(MagicMock(), SimpleNamespace(id="o1"))
    assert await ss.prepare("revenue") is None


@pytest.mark.asyncio
async def test_code_builder_skips_when_prepared_unavailable(monkeypatch):
    """A caller-supplied prepared=None (semantic unavailable) short-circuits
    instead of re-attempting the embedding per ranking pass."""
    from app.ai.context.builders import code_context_builder as ccb_mod
    from app.ai.context.builders.code_context_builder import CodeContextBuilder

    def _boom(*args, **kwargs):
        raise AssertionError("SemanticSearch should not be constructed")

    monkeypatch.setattr(ccb_mod, "SemanticSearch", _boom)
    cb = CodeContextBuilder.__new__(CodeContextBuilder)
    cb.db, cb.organization = MagicMock(), SimpleNamespace(id="o1")
    out = await cb._semantic_step_scores({"title": "Revenue"}, ["s1"], prepared=None)
    assert out is None


# --- store delete + batch upsert plumbing ------------------------------------

def _row(owner_id: str) -> EmbeddingRow:
    return EmbeddingRow(
        organization_id="o1", owner_type="instruction", owner_id=owner_id,
        content_hash=content_hash(owner_id), model_id="m", dim=3, vector=[0.1, 0.2, 0.3],
    )


@pytest.mark.asyncio
async def test_pg_upsert_batches_rows_into_one_executemany():
    db = MagicMock()
    db.execute = AsyncMock()
    db.commit = AsyncMock()
    store = PgVectorStore(db)

    await store.upsert([_row("a"), _row("b"), _row("c")])
    db.execute.assert_awaited_once()
    params = db.execute.await_args.args[1]
    assert isinstance(params, list) and len(params) == 3
    assert [p["owner_id"] for p in params] == ["a", "b", "c"]
    db.commit.assert_awaited_once()


@pytest.mark.asyncio
async def test_libsql_upsert_batches_rows_into_one_executemany():
    engine = MagicMock()
    conn = engine.begin.return_value.__enter__.return_value
    store = LibsqlVectorStore(engine)

    await store.upsert([_row("a"), _row("b")])
    conn.execute.assert_called_once()
    params = conn.execute.call_args.args[1]
    assert isinstance(params, list) and len(params) == 2
    assert [p["owner_id"] for p in params] == ["a", "b"]


@pytest.mark.asyncio
async def test_pg_delete_owners():
    db = MagicMock()
    db.execute = AsyncMock()
    db.commit = AsyncMock()
    store = PgVectorStore(db)

    await store.delete_owners("o1", "instruction", ["a", "b"])
    db.execute.assert_awaited_once()
    params = db.execute.await_args.args[1]
    assert params == {"org": "o1", "owner_type": "instruction", "owner_ids": ["a", "b"]}
    db.commit.assert_awaited_once()

    # Empty owner list is a no-op (no extra round-trip).
    db.execute.reset_mock()
    await store.delete_owners("o1", "instruction", [])
    db.execute.assert_not_awaited()


@pytest.mark.asyncio
async def test_libsql_delete_owners_and_stale_models():
    engine = MagicMock()
    conn = engine.begin.return_value.__enter__.return_value
    store = LibsqlVectorStore(engine)

    await store.delete_owners("o1", "instruction", ["a"])
    params = conn.execute.call_args.args[1]
    assert params == {"id0": "a", "org": "o1", "owner_type": "instruction"}

    conn.execute.reset_mock()
    await store.delete_stale_models("o1", "step", "m-new")
    params = conn.execute.call_args.args[1]
    assert params == {"org": "o1", "owner_type": "step", "model_id": "m-new"}


@pytest.mark.asyncio
async def test_pg_delete_stale_models():
    db = MagicMock()
    db.execute = AsyncMock()
    db.commit = AsyncMock()
    store = PgVectorStore(db)

    await store.delete_stale_models("o1", "step", "m-new")
    params = db.execute.await_args.args[1]
    assert params == {"org": "o1", "owner_type": "step", "model_id": "m-new"}
    db.commit.assert_awaited_once()


# --- step keep-fresh hook -----------------------------------------------------

def test_schedule_index_step_delegates(monkeypatch):
    from app.services import embedding_service as es_mod

    calls = []
    monkeypatch.setattr(
        ss_mod, "schedule_index", lambda org, owner_type, items: calls.append((org, owner_type, items))
    )
    es_mod.schedule_index_step("o1", "s1", "revenue by region")
    assert calls == [("o1", "step", [("s1", "revenue by region")])]


def test_schedule_index_step_never_raises(monkeypatch):
    from app.services import embedding_service as es_mod

    def _boom(*args, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(ss_mod, "schedule_index", _boom)
    es_mod.schedule_index_step("o1", "s1", "text")  # must not raise


# --- blending helpers -------------------------------------------------------

def test_instruction_blend_semantic():
    from app.ai.context.builders.instruction_context_builder import InstructionContextBuilder

    assert InstructionContextBuilder._blend_semantic(0.4, None) == 0.4
    assert InstructionContextBuilder._blend_semantic(0.4, 0.8) == pytest.approx(0.6)


def test_rank_related_instructions_blend(monkeypatch):
    from app.services.instruction_service import InstructionService

    svc = InstructionService.__new__(InstructionService)  # skip heavy __init__
    items = [
        SimpleNamespace(id="a", text="quarterly sales by region"),
        SimpleNamespace(id="b", text="unrelated note about cats"),
    ]
    tokens = {"revenue"}  # no literal overlap with either item

    # Without semantic, both score 0 (order preserved, stable).
    plain = svc._rank_related_instructions(tokens, items)
    assert {i.id for i in plain} == {"a", "b"}

    # Semantic surfaces 'a' (meaning match) above 'b'.
    ranked = svc._rank_related_instructions(tokens, items, {"a": 0.9, "b": 0.0})
    assert ranked[0].id == "a"


def test_code_data_model_text():
    from app.ai.context.builders.code_context_builder import CodeContextBuilder

    cb = CodeContextBuilder.__new__(CodeContextBuilder)
    text = cb._data_model_text(
        {"title": "Revenue", "columns": [{"generated_column_name": "total_sales"}]}
    )
    assert "Revenue" in text and "total_sales" in text
