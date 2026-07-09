"""SQL rewrite + lint pass for agent-generated queries.

Two responsibilities:

  1. **Rewrite** (``optimize_sql``) — apply only *provably semantics-preserving*
     transforms to the SQL before it executes, plus one explicitly-opt-in
     guardrail (a safety LIMIT). The hard rule: a rewrite must never change the
     result the caller would have gotten, except the safety LIMIT which is gated
     behind its own flag and only ever *caps* an otherwise-unbounded scan.

  2. **Lint** (``lint_sql``) — produce advisory warnings (never mutate the SQL).
     These are surfaced back to the planner/coder so a *future* generation can
     fix them. When a TableProfile is supplied, lints get sharper: a filter on a
     low-cardinality column against a literal that was never observed in the
     data is flagged with the real candidate values.

Everything degrades to a no-op: if sqlglot is unavailable or the SQL doesn't
parse for the given dialect, ``optimize_sql`` returns the original string and
``lint_sql`` returns whatever cheap regex-level checks still apply.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Optional

from app.ai.query_intelligence.config import QIConfig, get_qi_config

try:  # optional dependency — mirrors result_lake.py
    import sqlglot
    from sqlglot import exp
    _HAS_SQLGLOT = True
except Exception:  # pragma: no cover - exercised only in envs without sqlglot
    sqlglot = None  # type: ignore
    exp = None  # type: ignore
    _HAS_SQLGLOT = False


# Connection ``type`` (as stored on Connection) -> sqlglot dialect name. Anything
# not listed parses with the permissive default dialect.
_DIALECT_MAP = {
    "postgresql": "postgres",
    "postgres": "postgres",
    "redshift": "redshift",
    "aws_redshift": "redshift",
    "mysql": "mysql",
    "mariadb": "mysql",
    "bigquery": "bigquery",
    "snowflake": "snowflake",
    "mssql": "tsql",
    "sqlserver": "tsql",
    "clickhouse": "clickhouse",
    "databricks": "databricks",
    "databricks_sql": "databricks",
    "spark": "spark",
    "duckdb": "duckdb",
    "athena": "presto",
    "aws_athena": "presto",
    "presto": "presto",
    "trino": "trino",
}


def dialect_for(source_type: Optional[str]) -> Optional[str]:
    """Map a data source ``type`` to a sqlglot dialect, or None for the default."""
    if not source_type:
        return None
    return _DIALECT_MAP.get(str(source_type).strip().lower())


@dataclass
class OptimizationResult:
    """Outcome of an ``optimize_sql`` call.

    ``sql`` is always safe to execute — it equals the input when nothing could
    be done. ``changed`` says whether ``sql`` differs from the input. ``notes``
    describe applied rewrites; ``warnings`` are advisory lint findings (the SQL
    is *not* modified to address them).
    """

    sql: str
    original_sql: str
    changed: bool = False
    notes: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "changed": self.changed,
            "notes": list(self.notes),
            "warnings": list(self.warnings),
        }


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #

def optimize_sql(
    sql: str,
    *,
    source_type: Optional[str] = None,
    dialect: Optional[str] = None,
    profile: Any = None,
    config: Optional[QIConfig] = None,
) -> OptimizationResult:
    """Rewrite ``sql`` with safe transforms and collect lint warnings.

    Never raises: any failure returns the original SQL untouched.
    """
    cfg = config or get_qi_config()
    result = OptimizationResult(sql=sql, original_sql=sql)
    if not isinstance(sql, str) or not sql.strip():
        return result

    dia = dialect or dialect_for(source_type)

    # Cheap regex lints always run — they don't need a parse and still apply
    # when sqlglot is missing or the parse fails.
    result.warnings.extend(_regex_lints(sql))

    if not _HAS_SQLGLOT:
        return result

    try:
        tree = sqlglot.parse_one(sql, read=dia)
    except Exception:
        # Unparseable for this dialect — leave SQL as-is, keep regex lints only.
        return result
    if tree is None:
        return result

    changed = False

    # (1) Sound rewrite: drop ORDER BY from non-root selects with no LIMIT/OFFSET.
    if _strip_pointless_order_by(tree):
        changed = True
        result.notes.append("removed ORDER BY in subquery without LIMIT (no effect on result)")

    # (2) Guardrail (opt-in, behavior-capping): add a safety LIMIT to an
    #     unbounded, non-aggregating top-level SELECT.
    if cfg.safety_limit_enabled:
        if _add_safety_limit(tree, cfg.safety_limit_rows):
            changed = True
            result.notes.append(
                f"added safety LIMIT {cfg.safety_limit_rows} to an unbounded scan (guardrail)"
            )

    # AST-level lints (more precise than regex).
    result.warnings.extend(_ast_lints(tree, profile))
    # De-dup warnings while preserving order.
    result.warnings = list(dict.fromkeys(result.warnings))

    if changed:
        try:
            new_sql = tree.sql(dialect=dia)
            # Only report a change if the rewrite is enabled; otherwise we still
            # surface notes/warnings but hand back the original SQL so behavior
            # is unchanged until the operator opts in.
            if cfg.rewrite_enabled:
                result.sql = new_sql
                result.changed = True
            else:
                result.notes = [f"(rewrite disabled) would have: {n}" for n in result.notes]
        except Exception:
            # Serialization failed — fall back to the original string.
            pass

    return result


def lint_sql(
    sql: str,
    *,
    source_type: Optional[str] = None,
    dialect: Optional[str] = None,
    profile: Any = None,
) -> list[str]:
    """Return advisory warnings for ``sql`` without modifying it."""
    warnings = list(_regex_lints(sql)) if isinstance(sql, str) else []
    if _HAS_SQLGLOT and isinstance(sql, str) and sql.strip():
        dia = dialect or dialect_for(source_type)
        try:
            tree = sqlglot.parse_one(sql, read=dia)
            if tree is not None:
                warnings.extend(_ast_lints(tree, profile))
        except Exception:
            pass
    return list(dict.fromkeys(warnings))


# --------------------------------------------------------------------------- #
# Rewrites
# --------------------------------------------------------------------------- #

def _strip_pointless_order_by(tree: "exp.Expression") -> bool:
    """Remove ORDER BY from any non-root SELECT that has no LIMIT/OFFSET.

    Standard SQL gives no ordering guarantee to a subquery's rows unless paired
    with a row-limiting clause, so the ORDER BY there is dead work the engine may
    even be forced to honor. Dropping it cannot change the final result set.
    Returns True if anything was removed.
    """
    removed = False
    for select in tree.find_all(exp.Select):
        if select is tree:
            continue  # never touch the outermost ordering
        order = select.args.get("order")
        if order is None:
            continue
        if select.args.get("limit") is not None or select.args.get("offset") is not None:
            continue  # ORDER BY + LIMIT is meaningful — keep it
        select.set("order", None)
        removed = True
    return removed


def _add_safety_limit(tree: "exp.Expression", limit_rows: int) -> bool:
    """Add ``LIMIT limit_rows`` to a top-level SELECT that has none.

    Skipped when the query already has a LIMIT, is an aggregation/DISTINCT (its
    cardinality is already bounded by grouping), or isn't a plain SELECT. This is
    a guardrail against accidental full-table scans, not a semantics-preserving
    rewrite — hence its own opt-in flag.
    """
    if not isinstance(tree, exp.Select):
        return False
    if tree.args.get("limit") is not None:
        return False
    if tree.args.get("group") is not None or tree.args.get("distinct") is not None:
        return False
    # Bare aggregate (e.g. SELECT COUNT(*) FROM t) returns one row — no cap needed.
    expressions = tree.args.get("expressions") or []
    if expressions and all(_is_aggregate(e) for e in expressions):
        return False
    try:
        tree.limit(limit_rows, copy=False)
        return True
    except Exception:
        return False


def _is_aggregate(node: "exp.Expression") -> bool:
    try:
        return bool(node.find(exp.AggFunc))
    except Exception:
        return False


# --------------------------------------------------------------------------- #
# Lints
# --------------------------------------------------------------------------- #

_SELECT_STAR_RE = re.compile(r"select\s+\*", re.IGNORECASE)


def _regex_lints(sql: str) -> list[str]:
    out: list[str] = []
    if _SELECT_STAR_RE.search(sql):
        out.append(
            "SELECT * scans every column — list only the columns the data model "
            "needs to cut scanned/returned bytes."
        )
    return out


def _ast_lints(tree: "exp.Expression", profile: Any) -> list[str]:
    out: list[str] = []

    # Cartesian product: a top-level SELECT joining 2+ tables with no ON/USING
    # and no WHERE to relate them.
    for select in tree.find_all(exp.Select):
        joins = select.args.get("joins") or []
        unqualified = [j for j in joins if not j.args.get("on") and not j.args.get("using")]
        cross = [j for j in unqualified if (j.kind or "").upper() != "CROSS"]
        if cross and select.args.get("where") is None:
            out.append(
                "possible cartesian product: tables joined without ON/USING and no "
                "WHERE relating them — add a join condition or confirm the cross join is intended."
            )
            break

    # Value-dictionary check: a literal filter on a low-cardinality column whose
    # value was never observed in the sampled data. Uses the learned profile.
    value_dicts = _profile_value_dicts(profile)
    if value_dicts:
        for eq in tree.find_all(exp.EQ):
            col = eq.find(exp.Column)
            lit = eq.find(exp.Literal)
            if col is None or lit is None or not lit.is_string:
                continue
            col_name = col.name
            known = value_dicts.get(col_name) or value_dicts.get(col_name.lower())
            if not known:
                continue
            val = lit.this
            known_lower = {str(v).lower() for v in known}
            if str(val).lower() not in known_lower:
                sample = ", ".join(repr(v) for v in list(known)[:8])
                out.append(
                    f"filter {col_name} = {val!r} matches no observed value; "
                    f"known values include: {sample}."
                )
    return out


def _profile_value_dicts(profile: Any) -> dict:
    """Extract {column: [values]} from a TableProfile-like object or dict."""
    if profile is None:
        return {}
    vd = None
    if isinstance(profile, dict):
        vd = profile.get("value_dictionaries")
    else:
        vd = getattr(profile, "value_dictionaries", None)
    return vd if isinstance(vd, dict) else {}
