"""Data profiler — learn about the actual data behind a connection.

Pulls a bounded sample from each table and derives, with pandas:
  * per-column null fraction, distinct estimate, min/max
  * value dictionaries for low-cardinality columns (real literals the model can
    filter on)
  * effectively-unique columns (candidate keys beyond declared PKs)
  * candidate join keys inferred from column name+type alignment to other
    tables' keys

The compute half (``profile_table_from_sample`` / ``infer_join_keys``) is pure
and unit-tested without any database. The I/O half (``DataProfiler``) issues at
most two read-only queries per table (a bounded sample + an optional COUNT) and
persists a ``TableProfile`` row. Everything is best-effort: a table that errors
or times out is recorded with ``status="partial"`` (or skipped) and never aborts
the run.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Iterable, Optional

import pandas as pd

from app.ai.query_intelligence.config import QIConfig, get_qi_config

logger = logging.getLogger(__name__)

PROFILE_VERSION = 1

# Column dtypes (as reported in schema metadata) we treat as dictionary-eligible.
_CATEGORICAL_HINTS = ("char", "text", "string", "enum", "bool", "category", "varchar")


def build_sample_sql(table_fqn: str, limit: int, dialect: Optional[str]) -> str:
    """Build a dialect-correct ``SELECT * ... <limit>`` sampling query.

    Uses sqlglot to transpile a canonical ``LIMIT`` form into the target dialect
    (so SQL Server gets ``TOP``, Oracle-style gets ``FETCH`` etc.) and falls back
    to a hand-written form when sqlglot is unavailable or the dialect needs TOP.
    """
    limit = max(1, int(limit))
    canonical = f"SELECT * FROM {table_fqn} LIMIT {limit}"
    try:
        import sqlglot

        return sqlglot.transpile(canonical, write=dialect)[0] if dialect else canonical
    except Exception:
        if dialect in ("tsql", "mssql", "sqlserver"):
            return f"SELECT TOP {limit} * FROM {table_fqn}"
        return canonical


def _is_categorical(dtype: Optional[str]) -> bool:
    if not dtype:
        return False
    d = str(dtype).lower()
    return any(h in d for h in _CATEGORICAL_HINTS)


def _json_safe(value: Any) -> Any:
    """Coerce a pandas/NumPy scalar to a JSON-serializable Python primitive."""
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    if isinstance(value, (str, bool, int, float)):
        return value
    # NumPy scalar -> Python scalar
    item = getattr(value, "item", None)
    if callable(item):
        try:
            return item()
        except Exception:
            pass
    return str(value)


def profile_table_from_sample(
    sample: pd.DataFrame,
    *,
    columns_meta: Optional[Iterable[dict]] = None,
    row_count_estimate: Optional[int] = None,
    config: Optional[QIConfig] = None,
) -> dict:
    """Derive a profile payload from an in-memory sample DataFrame.

    Pure function (no I/O). ``columns_meta`` is the schema's column list (used
    for dtype hints when the sample is empty). Returns the dict shape persisted
    onto ``TableProfile`` (minus identity/linkage fields).
    """
    cfg = config or get_qi_config()
    dtype_hints = {}
    for c in columns_meta or []:
        try:
            dtype_hints[c["name"]] = c.get("dtype")
        except Exception:
            continue

    sample_rows = int(len(sample))
    column_profiles: dict[str, dict] = {}
    value_dictionaries: dict[str, list] = {}
    unique_columns: list[str] = []

    cols = list(sample.columns)[: cfg.max_profiled_columns]
    for col in cols:
        series = sample[col]
        non_null = series.dropna()
        null_fraction = round(1.0 - (len(non_null) / sample_rows), 4) if sample_rows else None
        try:
            distinct = int(non_null.nunique())
        except Exception:
            distinct = None

        col_dtype = dtype_hints.get(col) or str(series.dtype)
        approx_unique = bool(distinct is not None and sample_rows > 0 and distinct == sample_rows and sample_rows >= 10)

        prof: dict[str, Any] = {
            "dtype": col_dtype,
            "null_fraction": null_fraction,
            "distinct_estimate": distinct,
            "approx_unique": approx_unique,
        }

        # min/max only for orderable, non-categorical-looking data.
        if distinct and not _is_categorical(col_dtype):
            try:
                prof["min"] = _json_safe(non_null.min())
                prof["max"] = _json_safe(non_null.max())
            except Exception:
                pass

        low_card = distinct is not None and 0 < distinct <= cfg.low_cardinality_max
        prof["low_cardinality"] = bool(low_card)
        column_profiles[col] = prof

        if approx_unique:
            unique_columns.append(col)

        # Build a value dictionary for low-cardinality, categorical-ish columns.
        if low_card and (_is_categorical(col_dtype) or distinct <= cfg.low_cardinality_max):
            try:
                values = [_json_safe(v) for v in non_null.unique().tolist()]
                values = [v for v in values if v is not None][: cfg.max_value_dict_values]
                if values:
                    value_dictionaries[col] = values
            except Exception:
                pass

    return {
        "row_count_estimate": row_count_estimate,
        "sample_rows": sample_rows,
        "column_profiles": column_profiles,
        "value_dictionaries": value_dictionaries,
        "unique_columns": unique_columns,
        "profile_version": PROFILE_VERSION,
    }


def infer_join_keys(
    table_name: str,
    table_profile: dict,
    other_tables: list[dict],
) -> list[dict]:
    """Infer candidate join keys for ``table_name`` from name+type alignment.

    ``other_tables`` is a list of ``{name, columns, unique_columns}`` for the
    sibling tables. A column here that shares a name (case-insensitive) with
    another table's unique/key column — and a compatible dtype family — becomes a
    candidate join with a confidence reflecting how strong the signal is. Value
    overlap is intentionally not probed here to keep the query budget bounded;
    the result is advisory and labeled ``source="name+type"``.
    """
    candidates: list[dict] = []
    my_cols = table_profile.get("column_profiles", {})
    for col_name, prof in my_cols.items():
        col_lower = col_name.lower()
        # Heuristic: only treat key-ish columns as join candidates.
        looks_keyish = col_lower.endswith("_id") or col_lower == "id" or col_lower.endswith("id")
        if not looks_keyish:
            continue
        my_family = _type_family(prof.get("dtype"))
        for other in other_tables:
            if other.get("name") == table_name:
                continue
            other_unique = {str(u).lower() for u in (other.get("unique_columns") or [])}
            for oc in other.get("columns", []) or []:
                oc_name = oc["name"] if isinstance(oc, dict) else oc
                if oc_name is None:
                    continue
                if oc_name.lower() != col_lower:
                    continue
                oc_family = _type_family(oc.get("dtype") if isinstance(oc, dict) else None)
                type_ok = my_family is None or oc_family is None or my_family == oc_family
                if not type_ok:
                    continue
                confidence = 0.5
                if oc_name.lower() in other_unique:
                    confidence += 0.3  # joins onto a unique/key column
                if col_lower != "id":
                    confidence += 0.1  # named *_id is a stronger signal than bare id
                candidates.append({
                    "left_column": col_name,
                    "right_table": other.get("name"),
                    "right_column": oc_name,
                    "confidence": round(min(confidence, 0.95), 2),
                    "match_rate": None,
                    "source": "name+type",
                })
    return candidates


def _type_family(dtype: Optional[str]) -> Optional[str]:
    if not dtype:
        return None
    d = str(dtype).lower()
    if any(t in d for t in ("int", "serial", "number", "numeric", "decimal", "float", "double")):
        return "number"
    if any(t in d for t in ("char", "text", "string", "uuid", "varchar")):
        return "string"
    if any(t in d for t in ("date", "time")):
        return "datetime"
    return None


class DataProfiler:
    """Issues the sampling queries and assembles per-table profile payloads.

    Connection-type aware (for dialect-correct sampling) but engine-agnostic: it
    only relies on ``client.execute_query(sql) -> DataFrame``. Persistence is
    handled by ``app/services/query_intelligence_service.py`` so this stays
    free of DB-session concerns and unit-testable with a fake client.
    """

    def __init__(self, config: Optional[QIConfig] = None) -> None:
        self.cfg = config or get_qi_config()

    def sample_table(self, client: Any, table_fqn: str, dialect: Optional[str]) -> Optional[pd.DataFrame]:
        sql = build_sample_sql(table_fqn, self.cfg.profile_sample_rows, dialect)
        try:
            df = client.execute_query(sql)
        except Exception as e:
            logger.info("profiler: sample failed for %s: %s", table_fqn, e)
            return None
        if isinstance(df, pd.DataFrame):
            return df
        try:
            return pd.DataFrame(df)
        except Exception:
            return None

    def count_rows(self, client: Any, table_fqn: str) -> Optional[int]:
        try:
            df = client.execute_query(f"SELECT COUNT(*) AS n FROM {table_fqn}")
            if isinstance(df, pd.DataFrame) and not df.empty:
                return int(df.iloc[0, 0])
        except Exception as e:
            logger.info("profiler: count failed for %s: %s", table_fqn, e)
        return None

    def profile_table(
        self,
        client: Any,
        table_fqn: str,
        *,
        columns_meta: Optional[Iterable[dict]] = None,
        dialect: Optional[str] = None,
        want_count: bool = True,
    ) -> Optional[dict]:
        """Sample + profile one table. Returns the payload dict or None if the
        table could not be sampled at all."""
        sample = self.sample_table(client, table_fqn, dialect)
        if sample is None:
            return None
        row_count = self.count_rows(client, table_fqn) if want_count else None
        payload = profile_table_from_sample(
            sample,
            columns_meta=columns_meta,
            row_count_estimate=row_count,
            config=self.cfg,
        )
        partial = sample.empty or (want_count and row_count is None)
        payload["status"] = "partial" if partial else "ok"
        payload["profiled_at"] = datetime.now(timezone.utc)
        return payload
