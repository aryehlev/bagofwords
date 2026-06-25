"""Unit tests for the query-intelligence cost feedback math (pure, no DB)."""
from __future__ import annotations

import pytest

from app.ai.query_intelligence import cost_recorder as cr

sqlglot = pytest.importorskip("sqlglot")


def test_extract_tables_basic():
    assert cr.extract_tables("SELECT * FROM orders") == {"orders"}


def test_extract_tables_join_and_schema_qualified():
    tables = cr.extract_tables("SELECT * FROM analytics.orders o JOIN customers c ON o.cid = c.id")
    assert tables == {"orders", "customers"}


def test_extract_tables_excludes_ctes():
    sql = "WITH recent AS (SELECT * FROM orders) SELECT * FROM recent"
    assert cr.extract_tables(sql) == {"orders"}


def test_summarize_timings_ignores_cache_hits_for_latency():
    timings = [
        {"query_ms": 1000, "rows": 10, "result_bytes": 100, "cache": "miss"},
        {"query_ms": 5, "rows": 10, "result_bytes": 100, "cache": "hit"},
    ]
    summary = cr.summarize_timings(timings)
    assert summary["observations"] == 1  # only the miss counts
    assert summary["avg_query_ms"] == 1000


def test_summarize_timings_skips_errors():
    timings = [{"query_ms": 50, "error": "boom"}, {"query_ms": 200, "rows": 3, "cache": "miss"}]
    summary = cr.summarize_timings(timings)
    assert summary["observations"] == 1
    assert summary["avg_query_ms"] == 200


def test_summarize_timings_empty():
    assert cr.summarize_timings([]) is None
    assert cr.summarize_timings([{"error": "x"}]) is None


def test_merge_marks_slow_when_over_threshold():
    obs = {"observations": 1, "avg_query_ms": 6000, "p95_query_ms": 6000, "max_query_ms": 6000, "last_query_ms": 6000}
    merged = cr.merge_cost_summary(None, obs, slow_query_ms=4000)
    assert merged["slow"] is True
    assert merged["observations"] == 1


def test_merge_not_slow_when_under_threshold():
    obs = {"observations": 1, "avg_query_ms": 100, "p95_query_ms": 100, "max_query_ms": 100, "last_query_ms": 100}
    merged = cr.merge_cost_summary(None, obs, slow_query_ms=4000)
    assert merged["slow"] is False


def test_merge_accumulates_observations():
    obs1 = {"observations": 2, "avg_query_ms": 100, "p95_query_ms": 120, "max_query_ms": 150, "last_query_ms": 100}
    first = cr.merge_cost_summary(None, obs1, slow_query_ms=4000)
    obs2 = {"observations": 1, "avg_query_ms": 200, "p95_query_ms": 200, "max_query_ms": 250, "last_query_ms": 200}
    second = cr.merge_cost_summary(first, obs2, slow_query_ms=4000)
    assert second["observations"] == 3
    assert second["max_query_ms"] == 250  # max is monotonic
    # Blended average sits between the two.
    assert 100 <= second["avg_query_ms"] <= 200


def test_attribute_costs_to_tables_groups_by_table():
    executed = ["SELECT * FROM orders", "SELECT * FROM customers"]
    timings = [
        {"index": 0, "query_ms": 1000, "rows": 5, "sql": "SELECT * FROM orders", "cache": "miss"},
        {"index": 1, "query_ms": 200, "rows": 3, "sql": "SELECT * FROM customers", "cache": "miss"},
    ]
    by_table = cr.attribute_costs_to_tables(executed, timings)
    assert set(by_table.keys()) == {"orders", "customers"}
    assert by_table["orders"]["avg_query_ms"] == 1000


def test_attribute_costs_falls_back_to_executed_queries_positionally():
    executed = ["SELECT * FROM orders"]
    timings = [{"index": 0, "query_ms": 300, "rows": 1, "cache": "miss"}]  # no 'sql' key
    by_table = cr.attribute_costs_to_tables(executed, timings)
    assert "orders" in by_table
