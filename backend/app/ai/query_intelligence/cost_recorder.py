"""Runtime cost feedback — close the loop from real execution back to profiles.

After a query runs, ``QueryCapturingClientWrapper`` records per-query timings
(wall-clock ms, rows, result bytes, the SQL, cache hit/miss, any error). This
module turns those raw timings into a rolling per-table ``cost_summary`` stored
on ``TableProfile``, so the next generation can be told "orders is slow — filter
and avoid SELECT *".

The math here is pure and unit-tested. Persistence (matching tables to profiles
and writing the rollup) lives in query_intelligence_service.py.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

try:
    import sqlglot
    from sqlglot import exp
    _HAS_SQLGLOT = True
except Exception:  # pragma: no cover
    sqlglot = None  # type: ignore
    exp = None  # type: ignore
    _HAS_SQLGLOT = False


def extract_tables(sql: str, dialect: Optional[str] = None) -> set[str]:
    """Return the base table names referenced by ``sql`` (best-effort).

    Names are returned lowercased and unqualified (the final identifier of a
    dotted name) so they line up with how the profiler keys tables. CTE names are
    excluded — they're query-local, not physical tables.
    """
    if not isinstance(sql, str) or not sql.strip() or not _HAS_SQLGLOT:
        return set()
    try:
        tree = sqlglot.parse_one(sql, read=dialect)
    except Exception:
        return set()
    if tree is None:
        return set()
    cte_names = {c.alias_or_name.lower() for c in tree.find_all(exp.CTE)}
    out: set[str] = set()
    for tbl in tree.find_all(exp.Table):
        name = (tbl.name or "").lower()
        if name and name not in cte_names:
            out.add(name)
    return out


def _percentile(values: list[float], pct: float) -> Optional[float]:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    k = (len(ordered) - 1) * pct
    lo = int(k)
    hi = min(lo + 1, len(ordered) - 1)
    frac = k - lo
    return ordered[lo] + (ordered[hi] - ordered[lo]) * frac


def summarize_timings(timings: list[dict]) -> Optional[dict]:
    """Collapse a list of per-query timing dicts into one observation summary.

    Considers only successful, real (non-cache-hit) executions for the latency
    aggregates — cache hits are ~free and would otherwise hide true source cost.
    Returns None when there's nothing measurable.
    """
    if not timings:
        return None
    durations: list[float] = []
    rows: list[float] = []
    byts: list[float] = []
    last_ms: Optional[float] = None
    for t in timings:
        if not isinstance(t, dict) or t.get("error"):
            continue
        ms = t.get("query_ms")
        if ms is None:
            continue
        last_ms = float(ms)
        if t.get("cache") == "hit":
            continue
        durations.append(float(ms))
        if t.get("rows") is not None:
            rows.append(float(t["rows"]))
        if t.get("result_bytes") is not None:
            byts.append(float(t["result_bytes"]))
    if not durations and last_ms is None:
        return None
    return {
        "observations": len(durations),
        "avg_query_ms": round(sum(durations) / len(durations), 1) if durations else None,
        "p95_query_ms": round(_percentile(durations, 0.95), 1) if durations else None,
        "max_query_ms": round(max(durations), 1) if durations else None,
        "avg_rows": round(sum(rows) / len(rows), 1) if rows else None,
        "avg_result_bytes": round(sum(byts) / len(byts), 1) if byts else None,
        "last_query_ms": round(last_ms, 1) if last_ms is not None else None,
    }


def merge_cost_summary(existing: Optional[dict], obs: dict, slow_query_ms: float) -> dict:
    """Fold a new observation into the rolling per-table cost summary.

    Uses a decaying weighted average (recent runs count more) so the summary
    tracks current performance without being whipsawed by a single slow run.
    """
    prev = existing if isinstance(existing, dict) else {}
    prev_n = int(prev.get("observations") or 0)
    new_n = int(obs.get("observations") or 0)
    total_n = prev_n + new_n

    def _blend(key: str) -> Optional[float]:
        pv = prev.get(key)
        nv = obs.get(key)
        if pv is None:
            return nv
        if nv is None:
            return pv
        # Weight history at 0.7, but never let it fully drown new signal.
        w_prev = 0.7 * (prev_n / total_n) if total_n else 0.7
        w_new = 1.0 - w_prev
        return round(pv * w_prev + nv * w_new, 1)

    avg_ms = _blend("avg_query_ms")
    p95 = obs.get("p95_query_ms")
    if prev.get("p95_query_ms") is not None and p95 is not None:
        p95 = round(max(float(prev["p95_query_ms"]) * 0.5, float(p95)), 1)
    elif p95 is None:
        p95 = prev.get("p95_query_ms")

    max_ms = max(
        [v for v in (prev.get("max_query_ms"), obs.get("max_query_ms")) if v is not None],
        default=None,
    )
    slow_basis = p95 if p95 is not None else avg_ms
    return {
        "observations": total_n,
        "avg_query_ms": avg_ms,
        "p95_query_ms": p95,
        "max_query_ms": max_ms,
        "avg_rows": _blend("avg_rows"),
        "avg_result_bytes": _blend("avg_result_bytes"),
        "last_query_ms": obs.get("last_query_ms"),
        "slow": bool(slow_basis is not None and slow_basis >= slow_query_ms),
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }


def attribute_costs_to_tables(
    executed_queries: list[str],
    timings: list[dict],
    dialect: Optional[str] = None,
) -> dict[str, dict]:
    """Map each table referenced this run to a summarized observation.

    A query's cost is attributed to every base table it scans (we can't split a
    join's cost across its inputs without a plan, so we don't pretend to). The
    per-query SQL is taken from the timing entry when present, else positionally
    from ``executed_queries``.
    """
    # Group timing entries by the tables their SQL touches.
    per_table: dict[str, list[dict]] = {}
    for i, t in enumerate(timings or []):
        sql = None
        if isinstance(t, dict):
            sql = t.get("sql")
        if not sql and i < len(executed_queries or []):
            sql = executed_queries[i]
        if not sql:
            continue
        for table in extract_tables(sql, dialect):
            per_table.setdefault(table, []).append(t)
    return {tbl: summarize_timings(ts) for tbl, ts in per_table.items() if summarize_timings(ts)}
