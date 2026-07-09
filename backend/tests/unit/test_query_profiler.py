"""Unit tests for the data profiler's pure compute (no DB / no data source)."""
from __future__ import annotations

import pandas as pd

from app.ai.query_intelligence.config import QIConfig
from app.ai.query_intelligence import profiler as pf


def _cfg(**kw) -> QIConfig:
    base = dict(enabled=True, low_cardinality_max=10, max_value_dict_values=25, profile_sample_rows=1000)
    base.update(kw)
    return QIConfig(**base)


def test_build_sample_sql_default_limit():
    assert pf.build_sample_sql("orders", 100, "postgres").lower().startswith("select")
    assert "100" in pf.build_sample_sql("orders", 100, "postgres")


def test_build_sample_sql_tsql_uses_top():
    out = pf.build_sample_sql("orders", 25, "tsql").lower()
    assert "top" in out and "25" in out


def test_profile_low_cardinality_builds_value_dictionary():
    # 12 rows so the unique-column heuristic (needs a sample of >= 10) can fire.
    statuses = ["active", "churned", "trial"] * 4
    df = pd.DataFrame({"status": statuses, "id": list(range(1, 13))})
    cols = [{"name": "status", "dtype": "varchar"}, {"name": "id", "dtype": "integer"}]
    out = pf.profile_table_from_sample(df, columns_meta=cols, row_count_estimate=12, config=_cfg())

    assert out["sample_rows"] == 12
    assert set(out["value_dictionaries"]["status"]) == {"active", "churned", "trial"}
    # id is high-cardinality / unique — no dictionary, but flagged as a candidate key.
    assert "id" not in out["value_dictionaries"]
    assert "id" in out["unique_columns"]


def test_profile_null_fraction_and_distinct():
    df = pd.DataFrame({"col": [1, None, 3, None, 5]})
    out = pf.profile_table_from_sample(df, columns_meta=[{"name": "col", "dtype": "integer"}], config=_cfg())
    prof = out["column_profiles"]["col"]
    assert prof["null_fraction"] == 0.4
    assert prof["distinct_estimate"] == 3


def test_profile_high_cardinality_skips_dictionary():
    df = pd.DataFrame({"email": [f"user{i}@x.com" for i in range(50)]})
    out = pf.profile_table_from_sample(df, columns_meta=[{"name": "email", "dtype": "varchar"}], config=_cfg(low_cardinality_max=10))
    assert "email" not in out["value_dictionaries"]


def test_profile_empty_sample_is_safe():
    out = pf.profile_table_from_sample(pd.DataFrame(), config=_cfg())
    assert out["sample_rows"] == 0
    assert out["value_dictionaries"] == {}
    assert out["unique_columns"] == []


def test_infer_join_keys_matches_named_id_to_unique_key():
    orders = {
        "column_profiles": {
            "id": {"dtype": "integer"},
            "customer_id": {"dtype": "integer"},
            "amount": {"dtype": "numeric"},
        }
    }
    siblings = [
        {"name": "customers", "columns": [{"name": "id", "dtype": "integer"}], "unique_columns": []},
        {"name": "customer_id_lookup", "columns": [{"name": "customer_id", "dtype": "integer"}], "unique_columns": ["customer_id"]},
    ]
    keys = pf.infer_join_keys("orders", orders, siblings)
    targets = {(k["left_column"], k["right_table"], k["right_column"]) for k in keys}
    assert ("customer_id", "customer_id_lookup", "customer_id") in targets
    # Joining onto a unique column should carry higher confidence.
    unique_join = next(k for k in keys if k["right_table"] == "customer_id_lookup")
    assert unique_join["confidence"] >= 0.7


def test_infer_join_keys_skips_non_key_columns():
    tbl = {"column_profiles": {"amount": {"dtype": "numeric"}, "name": {"dtype": "varchar"}}}
    siblings = [{"name": "other", "columns": [{"name": "amount", "dtype": "numeric"}], "unique_columns": []}]
    assert pf.infer_join_keys("t", tbl, siblings) == []


class _FakeClient:
    """Minimal client: returns a fixed sample for SELECT *, a count for COUNT(*)."""

    def __init__(self, sample: pd.DataFrame, count: int):
        self._sample = sample
        self._count = count
        self.queries: list[str] = []

    def execute_query(self, sql: str):
        self.queries.append(sql)
        if "count(" in sql.lower():
            return pd.DataFrame({"n": [self._count]})
        return self._sample


def test_data_profiler_profile_table_end_to_end():
    sample = pd.DataFrame({"status": ["a", "b", "a"], "id": [1, 2, 3]})
    client = _FakeClient(sample, count=999)
    profiler = pf.DataProfiler(_cfg())
    payload = profiler.profile_table(client, "orders", columns_meta=[{"name": "status", "dtype": "varchar"}], dialect="postgres")
    assert payload is not None
    assert payload["row_count_estimate"] == 999
    assert payload["status"] == "ok"
    assert "status" in payload["value_dictionaries"]
    assert any("count(" in q.lower() for q in client.queries)


def test_data_profiler_handles_sample_failure():
    class _Boom:
        def execute_query(self, sql):
            raise RuntimeError("permission denied")

    profiler = pf.DataProfiler(_cfg())
    assert profiler.profile_table(_Boom(), "secret", dialect="postgres") is None
