"""Unit tests for the out-of-core LazyFrame query path.

Exercises the real streamers/consumers against SQLite (DBAPI), DuckDB (native
COPY) and pyarrow (Arrow consumer) — all dependencies, no live service needed.
"""
from __future__ import annotations

import os
import sqlite3
import time
from contextlib import contextmanager

import pandas as pd
import pyarrow as pa
import pytest

from app.data_sources.clients.lazy_frame import (
    LazyFrame,
    ResultTooLargeError,
    StreamConfig,
    _swept_roots,
    consume_arrow_to_lazyframe,
    consume_chunks_to_lazyframe,
    consume_row_dicts_to_lazyframe,
    lazy_from_dataframe,
    lazy_query_via_dbapi_cursor,
    lazy_query_via_dbapi_readsql,
    lazy_query_via_duckdb,
    lazy_query_via_sqlalchemy,
)


@pytest.fixture
def sqlite_db(tmp_path):
    path = tmp_path / "t.db"
    conn = sqlite3.connect(path)
    conn.execute("CREATE TABLE s(id INTEGER, region TEXT, amt INTEGER)")
    conn.executemany(
        "INSERT INTO s VALUES (?,?,?)",
        [(i, ["EU", "US", "APAC"][i % 3], i) for i in range(300)],
    )
    conn.commit()
    conn.close()

    @contextmanager
    def cm():
        c = sqlite3.connect(path)
        try:
            yield c
        finally:
            c.close()

    return cm


def test_lazy_from_dataframe_out_of_core(tmp_path, monkeypatch):
    monkeypatch.setenv("BOW_LAZY_DIR", str(tmp_path))
    df = pd.DataFrame({"region": ["EU", "US", "EU", "APAC"], "amount": [10, 20, 30, 40]})
    h = lazy_from_dataframe(df)
    try:
        assert h.row_count() == 4
        out = h.sql("SELECT region, SUM(amount) t FROM data GROUP BY region ORDER BY region").to_df()
        assert dict(zip(out.region, out.t)) == {"APAC": 40, "EU": 40, "US": 20}
        assert h.byte_size() > 0
    finally:
        h.close()


def test_dbapi_readsql_streaming(tmp_path, sqlite_db, monkeypatch):
    monkeypatch.setenv("BOW_LAZY_DIR", str(tmp_path))
    monkeypatch.setenv("BOW_LAZY_CHUNKSIZE", "50")
    h = lazy_query_via_dbapi_readsql(sqlite_db, "SELECT * FROM s")
    try:
        assert h.row_count() == 300
        total = h.sql("SELECT SUM(amt) t FROM data").to_df().iloc[0, 0]
        assert int(total) == sum(range(300))
    finally:
        h.close()


def test_dbapi_cursor_streaming_preserves_schema(tmp_path, sqlite_db, monkeypatch):
    monkeypatch.setenv("BOW_LAZY_DIR", str(tmp_path))
    monkeypatch.setenv("BOW_LAZY_CHUNKSIZE", "50")
    h = lazy_query_via_dbapi_cursor(sqlite_db, "SELECT id, region, amt FROM s")
    try:
        assert h.row_count() == 300
        assert h.columns == ["id", "region", "amt"]
    finally:
        h.close()


def test_native_duckdb_copy_zero_load(tmp_path, monkeypatch):
    monkeypatch.setenv("BOW_LAZY_DIR", str(tmp_path))
    duckdb = pytest.importorskip("duckdb")
    dbf = tmp_path / "x.duckdb"
    con = duckdb.connect(str(dbf))
    con.execute("CREATE TABLE s AS SELECT * FROM range(300) t(id)")
    con.close()

    @contextmanager
    def cm():
        c = duckdb.connect(str(dbf), read_only=True)
        try:
            yield c
        finally:
            c.close()

    h = lazy_query_via_duckdb(cm, "SELECT id FROM s WHERE id < 100")
    try:
        assert h.row_count() == 100
        assert int(h.sql("SELECT SUM(id) s FROM data").to_df().iloc[0, 0]) == sum(range(100))
    finally:
        h.close()


def test_arrow_consumer_mixed_batch_and_table(tmp_path, monkeypatch):
    monkeypatch.setenv("BOW_LAZY_DIR", str(tmp_path))
    batches = [
        pa.record_batch({"a": pa.array([1, 2, 3]), "b": pa.array(["x", "y", "z"])}),
        pa.table({"a": pa.array([4, 5]), "b": pa.array(["p", "q"])}),
    ]
    h = consume_arrow_to_lazyframe(iter(batches))
    try:
        assert h.row_count() == 5
        assert int(h.sql("SELECT SUM(a) s FROM data").to_df().iloc[0, 0]) == 15
    finally:
        h.close()


def test_row_dicts_consumer(tmp_path, monkeypatch):
    monkeypatch.setenv("BOW_LAZY_DIR", str(tmp_path))
    rows = [{"id": i, "meta": '{"k": %d}' % i} for i in range(200)]
    h = consume_row_dicts_to_lazyframe(iter(rows))
    try:
        assert h.row_count() == 200
        assert h.columns == ["id", "meta"]
    finally:
        h.close()


def test_chunk_consumer(tmp_path, monkeypatch):
    monkeypatch.setenv("BOW_LAZY_DIR", str(tmp_path))
    chunks = [pd.DataFrame({"v": range(64)}), pd.DataFrame({"v": range(64, 100)})]
    h = consume_chunks_to_lazyframe(iter(chunks))
    try:
        assert h.row_count() == 100
    finally:
        h.close()


def test_early_abort_on_row_cap(tmp_path, sqlite_db, monkeypatch):
    monkeypatch.setenv("BOW_LAZY_DIR", str(tmp_path))
    monkeypatch.setenv("BOW_LAZY_CHUNKSIZE", "50")
    monkeypatch.setenv("BOW_LAZY_MAX_ROWS", "100")
    with pytest.raises(ResultTooLargeError) as exc:
        lazy_query_via_dbapi_cursor(sqlite_db, "SELECT * FROM s")
    assert exc.value.status_code == 413
    # partial file must not be left behind
    assert list(tmp_path.glob("*.parquet")) == []


def test_close_removes_owned_temp_file(tmp_path, monkeypatch):
    monkeypatch.setenv("BOW_LAZY_DIR", str(tmp_path))
    h = lazy_from_dataframe(pd.DataFrame({"a": [1, 2, 3]}))
    src = h._source_path
    assert src.exists()
    h.close()
    assert not src.exists()


def test_schema_drift_all_null_later_chunk(tmp_path, monkeypatch):
    # A nullable numeric column that is all-NULL in one chunk infers a
    # different Arrow dtype (null) than the writer's (int64); the write must
    # cast, not abort the stream.
    monkeypatch.setenv("BOW_LAZY_DIR", str(tmp_path))
    chunks = [
        pd.DataFrame({"id": [1, 2], "v": [10, 20]}),
        pd.DataFrame({"id": [3, 4], "v": [None, None]}),
        pd.DataFrame({"id": [5], "v": [50]}),
    ]
    h = consume_chunks_to_lazyframe(iter(chunks))
    try:
        assert h.row_count() == 5
        assert int(h.sql("SELECT SUM(v) s FROM data").to_df().iloc[0, 0]) == 80
    finally:
        h.close()


def test_schema_drift_all_null_first_chunk(tmp_path, monkeypatch):
    # The anomalous chunk can also come FIRST: an all-NULL column infers
    # pa.null() and must be widened (to float64) before it locks the writer
    # schema, or every later non-null chunk would fail to write.
    monkeypatch.setenv("BOW_LAZY_DIR", str(tmp_path))
    chunks = [
        pd.DataFrame({"id": [1, 2], "v": [None, None]}),
        pd.DataFrame({"id": [3, 4], "v": [30, 40]}),
    ]
    h = consume_chunks_to_lazyframe(iter(chunks))
    try:
        assert h.row_count() == 4
        assert h.columns == ["id", "v"]
        assert int(h.sql("SELECT SUM(v) s FROM data").to_df().iloc[0, 0]) == 70
    finally:
        h.close()


def test_zero_row_sqlalchemy_preserves_columns(tmp_path, monkeypatch):
    sqlalchemy = pytest.importorskip("sqlalchemy")
    monkeypatch.setenv("BOW_LAZY_DIR", str(tmp_path))
    dbf = tmp_path / "z.db"
    conn = sqlite3.connect(dbf)
    conn.execute("CREATE TABLE s(id INTEGER, region TEXT)")
    conn.commit()
    conn.close()
    engine = sqlalchemy.create_engine(f"sqlite:///{dbf}")

    @contextmanager
    def cm():
        with engine.connect() as c:
            yield c

    h = lazy_query_via_sqlalchemy(cm, "SELECT id, region FROM s WHERE id < 0")
    try:
        assert h.row_count() == 0
        assert h.columns == ["id", "region"]
        out = h.sql("SELECT region FROM data").to_df()
        assert list(out.columns) == ["region"] and len(out) == 0
    finally:
        h.close()
        engine.dispose()


def test_zero_row_dbapi_readsql_preserves_columns(tmp_path, sqlite_db, monkeypatch):
    monkeypatch.setenv("BOW_LAZY_DIR", str(tmp_path))
    h = lazy_query_via_dbapi_readsql(sqlite_db, "SELECT id, region FROM s WHERE id < 0")
    try:
        assert h.row_count() == 0
        assert h.columns == ["id", "region"]
        assert list(h.sql("SELECT region FROM data").to_df().columns) == ["region"]
    finally:
        h.close()


def test_zero_row_dicts_with_columns_preserves_schema(tmp_path, monkeypatch):
    monkeypatch.setenv("BOW_LAZY_DIR", str(tmp_path))
    h = consume_row_dicts_to_lazyframe(iter([]), columns=["id", "name"])
    try:
        assert h.row_count() == 0
        assert h.columns == ["id", "name"]
    finally:
        h.close()


def test_chunk_consumer_closes_generator_on_abort(tmp_path, monkeypatch):
    # On abort the passed-in generator must be closed explicitly (releasing its
    # connection), not left suspended for a later GC pass.
    monkeypatch.setenv("BOW_LAZY_DIR", str(tmp_path))
    monkeypatch.setenv("BOW_LAZY_MAX_ROWS", "10")
    closed = []

    def gen():
        try:
            while True:
                yield pd.DataFrame({"v": range(50)})
        finally:
            closed.append(True)

    g = gen()  # hold a reference so refcount GC can't mask a missing close()
    with pytest.raises(ResultTooLargeError):
        consume_chunks_to_lazyframe(g)
    assert closed == [True]


def test_stale_lazy_files_swept_on_config_init(tmp_path, monkeypatch):
    monkeypatch.setenv("BOW_LAZY_DIR", str(tmp_path))
    old = tmp_path / "lazy_orphan.parquet"
    old.write_bytes(b"x")
    stale = time.time() - 25 * 3600
    os.utime(old, (stale, stale))
    fresh = tmp_path / "lazy_fresh.parquet"
    fresh.write_bytes(b"x")
    other = tmp_path / "keep.txt"  # non-lazy files must never be touched
    other.write_bytes(b"x")
    os.utime(other, (stale, stale))
    _swept_roots.discard(tmp_path)  # sweep runs once per root per process
    StreamConfig()
    assert not old.exists()
    assert fresh.exists()
    assert other.exists()


def test_mongodb_decimal128_conversion():
    pytest.importorskip("pymongo")
    bson = pytest.importorskip("bson")
    from app.data_sources.clients.mongodb_client import MongodbClient

    client = MongodbClient(host="localhost", database="x")
    doc = {
        "price": bson.Decimal128("19.99"),
        "nested": {"amt": bson.Decimal128("0.001")},
        "arr": [bson.Decimal128("2.5"), {"inner": bson.Decimal128("3.5")}],
    }
    client._convert_bson_types(doc)
    assert isinstance(doc["price"], float) and doc["price"] == pytest.approx(19.99)
    assert doc["nested"]["amt"] == pytest.approx(0.001)
    assert doc["arr"][0] == pytest.approx(2.5)
    assert doc["arr"][1]["inner"] == pytest.approx(3.5)
    # the converted doc must survive the columnar spill (raw Decimal128 raises)
    pa.Table.from_pandas(pd.DataFrame([{"price": doc["price"]}]))
