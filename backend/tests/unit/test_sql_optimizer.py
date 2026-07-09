"""Unit tests for the query-intelligence SQL optimizer (rewrite + lint).

Pure, no DB or data source. Rewrite-specific cases require sqlglot; the no-op /
regex-lint behavior is asserted unconditionally.
"""
from __future__ import annotations

import pytest

from app.ai.query_intelligence.config import QIConfig
from app.ai.query_intelligence import sql_optimizer as so

sqlglot = pytest.importorskip("sqlglot")


def _cfg(**kw) -> QIConfig:
    base = dict(enabled=True, rewrite_enabled=True, safety_limit_enabled=True, safety_limit_rows=1000)
    base.update(kw)
    return QIConfig(**base)


def test_dialect_mapping():
    assert so.dialect_for("postgresql") == "postgres"
    assert so.dialect_for("BigQuery") == "bigquery"
    assert so.dialect_for("mssql") == "tsql"
    assert so.dialect_for("totally-unknown") is None
    assert so.dialect_for(None) is None


def test_select_star_is_linted():
    res = so.optimize_sql("SELECT * FROM orders", source_type="postgresql", config=_cfg())
    assert any("SELECT *" in w for w in res.warnings)


def test_safety_limit_added_to_unbounded_scan():
    res = so.optimize_sql("SELECT id, name FROM orders", source_type="postgresql", config=_cfg(safety_limit_rows=500))
    assert res.changed
    assert "limit" in res.sql.lower()
    assert "500" in res.sql


def test_safety_limit_skipped_when_already_limited():
    res = so.optimize_sql("SELECT id FROM orders LIMIT 10", source_type="postgresql", config=_cfg())
    assert not res.changed
    assert res.sql.strip().lower().endswith("limit 10")


def test_safety_limit_skipped_for_aggregation():
    res = so.optimize_sql("SELECT count(*) FROM orders", source_type="postgresql", config=_cfg())
    assert not res.changed


def test_safety_limit_skipped_for_group_by():
    res = so.optimize_sql(
        "SELECT status, count(*) FROM orders GROUP BY status",
        source_type="postgresql",
        config=_cfg(),
    )
    assert not res.changed


def test_redundant_order_by_in_subquery_removed():
    sql = "SELECT * FROM (SELECT id FROM orders ORDER BY id) t"
    res = so.optimize_sql(sql, source_type="postgresql", config=_cfg(safety_limit_enabled=False))
    assert res.changed
    assert "order by" not in res.sql.lower()


def test_order_by_with_limit_in_subquery_is_kept():
    sql = "SELECT * FROM (SELECT id FROM orders ORDER BY id LIMIT 5) t"
    res = so.optimize_sql(sql, source_type="postgresql", config=_cfg(safety_limit_enabled=False))
    # The inner ORDER BY ... LIMIT is meaningful and must survive.
    assert "order by" in res.sql.lower()


def test_outer_order_by_is_never_removed():
    sql = "SELECT id FROM orders ORDER BY id"
    res = so.optimize_sql(sql, source_type="postgresql", config=_cfg(safety_limit_enabled=False))
    assert "order by" in res.sql.lower()


@pytest.mark.parametrize(
    "agg",
    ["array_agg(x)", "string_agg(x, ',')", "group_concat(x)", "listagg(x)", "json_agg(x)"],
)
def test_order_by_kept_when_order_sensitive_aggregate_consumes_it(agg):
    # e.g. SELECT array_agg(x) FROM (SELECT x FROM t ORDER BY x) s — the
    # subquery's ordering feeds the aggregate; stripping it changes the result.
    sql = f"SELECT {agg} FROM (SELECT x FROM t ORDER BY x) s"
    res = so.optimize_sql(sql, source_type="postgresql", config=_cfg(safety_limit_enabled=False))
    assert "order by" in res.sql.lower()
    assert not res.changed


def test_order_by_kept_in_distinct_subquery():
    # Postgres DISTINCT ON picks the surviving row based on ORDER BY.
    sql = "SELECT * FROM (SELECT DISTINCT ON (a) a, b FROM t ORDER BY a, b DESC) s"
    res = so.optimize_sql(sql, source_type="postgresql", config=_cfg(safety_limit_enabled=False))
    assert "order by" in res.sql.lower()


def test_order_by_still_stripped_with_order_insensitive_aggregate():
    sql = "SELECT count(*) FROM (SELECT x FROM t ORDER BY x) s"
    res = so.optimize_sql(sql, source_type="postgresql", config=_cfg(safety_limit_enabled=False))
    assert "order by" not in res.sql.lower()


def test_rewrite_disabled_returns_original_sql():
    res = so.optimize_sql(
        "SELECT id FROM orders",
        source_type="postgresql",
        config=_cfg(rewrite_enabled=False),
    )
    # Notes describe what *would* happen, but the SQL is untouched.
    assert res.sql == "SELECT id FROM orders"
    assert res.changed is False


def test_cartesian_product_warned():
    sql = "SELECT * FROM a JOIN b"
    res = so.optimize_sql(sql, source_type="postgresql", config=_cfg())
    assert any("cartesian" in w.lower() for w in res.warnings)


def test_value_dictionary_lint_flags_unknown_literal():
    profile = {"value_dictionaries": {"status": ["active", "churned", "trial"]}}
    res = so.optimize_sql(
        "SELECT id FROM users WHERE status = 'cancelled'",
        source_type="postgresql",
        profile=profile,
        config=_cfg(),
    )
    assert any("matches no observed value" in w for w in res.warnings)


def test_value_dictionary_lint_passes_known_literal():
    profile = {"value_dictionaries": {"status": ["active", "churned", "trial"]}}
    res = so.optimize_sql(
        "SELECT id FROM users WHERE status = 'active'",
        source_type="postgresql",
        profile=profile,
        config=_cfg(),
    )
    assert not any("matches no observed value" in w for w in res.warnings)


def test_unparseable_sql_is_returned_untouched():
    junk = "this is not sql at all );;"
    res = so.optimize_sql(junk, source_type="postgresql", config=_cfg())
    assert res.sql == junk
    assert res.changed is False


def test_empty_input_is_safe():
    res = so.optimize_sql("", source_type="postgresql", config=_cfg())
    assert res.sql == ""
    assert res.changed is False


def test_tsql_safety_limit_uses_top():
    res = so.optimize_sql("SELECT id FROM orders", source_type="mssql", config=_cfg(safety_limit_rows=50))
    assert res.changed
    # TSQL has no LIMIT — sqlglot renders the cap as TOP.
    assert "top" in res.sql.lower()
