"""Persistence + orchestration for the query-intelligence subsystem.

Glue between the pure compute modules (profiler / cost_recorder /
profile_formatter) and the database:

  * ``profile_data_source`` — the "learn about the data" entry point: samples
    every active table of a data source and upserts a ``TableProfile`` per
    (connection, table).
  * ``get_profiles_for_tables`` — fetch profiles for the coder prompt.
  * ``record_costs`` — fold real query timings back onto the profiles.

All methods are defensive: a failure on one table/connection is logged and
skipped, never propagated, so profiling can never break the request that
triggered it.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from typing import Any, Iterable, Optional

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.ai.query_intelligence.config import QIConfig, get_qi_config
from app.ai.query_intelligence.cost_recorder import attribute_costs_to_tables, merge_cost_summary
from app.ai.query_intelligence.profiler import DataProfiler, infer_join_keys
from app.ai.query_intelligence.sql_optimizer import dialect_for
from app.models.connection_table import ConnectionTable
from app.models.table_profile import TableProfile

logger = logging.getLogger(__name__)


def _norm(name: Optional[str]) -> str:
    """Normalize a table name for matching (unqualified, lowercased)."""
    if not name:
        return ""
    return str(name).split(".")[-1].strip().strip('"').strip("`").lower()


async def _connection_ids_for_data_sources(
    db: AsyncSession, data_sources: Iterable[Any]
) -> list[str]:
    ids: set[str] = set()
    for ds in data_sources or []:
        try:
            for conn in getattr(ds, "connections", []) or []:
                ids.add(str(conn.id))
        except Exception:
            continue
    return list(ids)


async def get_profiles_for_tables(
    db: AsyncSession,
    *,
    data_sources: Iterable[Any],
    table_names: Optional[Iterable[str]] = None,
) -> list[TableProfile]:
    """Return profiles for the given data sources, optionally filtered to the
    tables actually referenced this turn."""
    conn_ids = await _connection_ids_for_data_sources(db, data_sources)
    if not conn_ids:
        return []
    stmt = select(TableProfile).where(TableProfile.connection_id.in_(conn_ids))
    rows = (await db.execute(stmt)).scalars().all()
    if table_names:
        wanted = {_norm(t) for t in table_names if t}
        rows = [r for r in rows if _norm(r.table_fqn) in wanted]
    return rows


async def profile_data_source(
    db: AsyncSession,
    data_source: Any,
    *,
    config: Optional[QIConfig] = None,
    table_limit: Optional[int] = None,
) -> dict:
    """Sample and profile all active tables of a data source.

    Returns a summary dict ``{profiled, skipped, errors}``. Issues read-only
    queries only; honors the per-query timeout via the client's own controls.
    """
    cfg = config or get_qi_config()
    profiler = DataProfiler(cfg)
    summary = {"profiled": 0, "skipped": 0, "errors": 0}

    connections = list(getattr(data_source, "connections", []) or [])
    for connection in connections:
        dialect = dialect_for(getattr(connection, "type", None))
        try:
            client = await asyncio.to_thread(connection.get_client)
        except Exception as e:
            logger.warning("query-intel: could not build client for connection %s: %s", connection.id, e)
            summary["errors"] += 1
            continue

        # Discovered tables for this connection.
        ct_rows = (
            await db.execute(
                select(ConnectionTable).where(ConnectionTable.connection_id == str(connection.id))
            )
        ).scalars().all()
        if table_limit:
            ct_rows = ct_rows[:table_limit]

        # First pass: sample + per-table profile. Buffer payloads so the second
        # pass can infer join keys across the whole connection.
        payloads: list[tuple[ConnectionTable, dict]] = []
        sibling_meta: list[dict] = []
        for ct in ct_rows:
            try:
                payload = await asyncio.to_thread(
                    profiler.profile_table,
                    client,
                    ct.name,
                    columns_meta=ct.columns,
                    dialect=dialect,
                )
            except Exception as e:
                logger.info("query-intel: profile_table failed for %s: %s", ct.name, e)
                payload = None
            if payload is None:
                summary["skipped"] += 1
                continue
            payloads.append((ct, payload))
            sibling_meta.append({
                "name": ct.name,
                "columns": ct.columns,
                "unique_columns": payload.get("unique_columns", []),
            })

        # Second pass: join-key inference + persist.
        for ct, payload in payloads:
            try:
                payload["learned_join_keys"] = infer_join_keys(ct.name, payload, sibling_meta)
                await _upsert_profile(db, connection_id=str(connection.id), connection_table_id=str(ct.id), table_fqn=ct.name, payload=payload)
                summary["profiled"] += 1
            except Exception as e:
                logger.warning("query-intel: persist failed for %s: %s", ct.name, e)
                summary["errors"] += 1

    try:
        await db.commit()
    except Exception as e:
        logger.warning("query-intel: commit failed: %s", e)
        await db.rollback()
    return summary


async def _upsert_profile(
    db: AsyncSession,
    *,
    connection_id: str,
    connection_table_id: Optional[str],
    table_fqn: str,
    payload: dict,
) -> None:
    existing = (
        await db.execute(
            select(TableProfile).where(
                TableProfile.connection_id == connection_id,
                TableProfile.table_fqn == table_fqn,
            )
        )
    ).scalar_one_or_none()

    fields = dict(
        connection_table_id=connection_table_id,
        row_count_estimate=payload.get("row_count_estimate"),
        sample_rows=int(payload.get("sample_rows") or 0),
        column_profiles=payload.get("column_profiles") or {},
        value_dictionaries=payload.get("value_dictionaries") or {},
        unique_columns=payload.get("unique_columns") or [],
        learned_join_keys=payload.get("learned_join_keys") or [],
        profile_version=int(payload.get("profile_version") or 1),
        status=payload.get("status") or "ok",
        profiled_at=payload.get("profiled_at") or datetime.now(timezone.utc),
    )

    if existing is None:
        db.add(TableProfile(connection_id=connection_id, table_fqn=table_fqn, **fields))
    else:
        for k, v in fields.items():
            setattr(existing, k, v)


async def record_costs(
    db: AsyncSession,
    *,
    data_sources: Iterable[Any],
    executed_queries: list[str],
    query_timings: list[dict],
    config: Optional[QIConfig] = None,
) -> int:
    """Fold this run's query timings into the per-table cost summaries.

    Returns the number of profiles updated. No-op (returns 0) when cost feedback
    is disabled or there are no profiles to attribute costs to.
    """
    cfg = config or get_qi_config()
    if not cfg.cost_feedback_enabled or not query_timings:
        return 0

    data_sources = list(data_sources or [])
    # Use the first connection's dialect as a parsing hint (table extraction is
    # dialect-tolerant; this only improves it).
    dialect = None
    for ds in data_sources:
        for conn in getattr(ds, "connections", []) or []:
            dialect = dialect_for(getattr(conn, "type", None))
            break
        if dialect:
            break

    by_table = attribute_costs_to_tables(executed_queries or [], query_timings, dialect)
    if not by_table:
        return 0

    profiles = await get_profiles_for_tables(db, data_sources=data_sources, table_names=list(by_table.keys()))
    if not profiles:
        return 0

    by_name = {}
    for p in profiles:
        by_name.setdefault(_norm(p.table_fqn), p)

    updated = 0
    for table_name, obs in by_table.items():
        prof = by_name.get(_norm(table_name))
        if prof is None or obs is None:
            continue
        try:
            prof.cost_summary = merge_cost_summary(prof.cost_summary, obs, cfg.slow_query_ms)
            updated += 1
        except Exception as e:
            logger.info("query-intel: cost merge failed for %s: %s", table_name, e)

    if updated:
        try:
            await db.commit()
        except Exception as e:
            logger.warning("query-intel: cost commit failed: %s", e)
            await db.rollback()
            return 0
    return updated
