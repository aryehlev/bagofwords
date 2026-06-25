from sqlalchemy import Column, String, ForeignKey, JSON, Integer, DateTime, Index
from sqlalchemy.orm import relationship

from app.models.base import BaseSchema


class TableProfile(BaseSchema):
    """Learned data profile for a single table — the "knows about the data" half
    of the query-intelligence subsystem.

    Distinct from:
      * ConnectionTable — the *structural* schema (columns/pks/fks) and topology
        metrics computed from the graph, and
      * TableStats — *usage/feedback* signals aggregated from agent runs.

    TableProfile is computed by sampling the actual rows (see
    app/ai/query_intelligence/profiler.py): value dictionaries for low-cardinality
    columns, per-column null/distinct estimates, learned join keys and effective
    uniqueness, plus a rolling cost summary fed back from real query timings
    (see app/ai/query_intelligence/cost_recorder.py).

    Everything here is best-effort and advisory: a missing or stale profile only
    lowers generation quality, it never blocks query execution. Rows are keyed by
    (connection_id, table_fqn) so the same physical table is profiled once per
    connection regardless of how many domains activate it.
    """

    __tablename__ = "table_profiles"

    connection_id = Column(String(36), ForeignKey("connections.id"), nullable=False, index=True)
    # Soft link to the discovered schema row. Nullable so a profile can outlive a
    # schema refresh that re-creates ConnectionTable rows; matched back by FQN.
    connection_table_id = Column(String(36), ForeignKey("connection_tables.id"), nullable=True, index=True)

    # Fully-qualified table name as the generated SQL references it (e.g.
    # "analytics.public.orders" or just "orders"). The stable join key for
    # cost rollups and prompt rendering.
    table_fqn = Column(String, nullable=False)

    # Best-effort row count from the sampling pass (COUNT(*) or an approx).
    row_count_estimate = Column(Integer, nullable=True)
    # Number of rows actually sampled to derive the column stats below.
    sample_rows = Column(Integer, nullable=False, default=0)

    # Per-column learned stats:
    #   {col_name: {dtype, null_fraction, distinct_estimate, approx_unique(bool),
    #               min, max, low_cardinality(bool)}}
    column_profiles = Column(JSON, nullable=False, default=dict)

    # Category dictionaries for low-cardinality columns so the model filters with
    # real literals instead of guessing:
    #   {col_name: ["active", "churned", "trial", ...]}  (capped per column)
    value_dictionaries = Column(JSON, nullable=False, default=dict)

    # Columns that are effectively unique in the sample (distinct ~= row count) —
    # candidate keys beyond declared PKs. List of column names.
    unique_columns = Column(JSON, nullable=False, default=list)

    # Join relationships inferred from the data (value overlap), beyond declared
    # FKs:
    #   [{left_column, right_table, right_column, confidence, match_rate, source}]
    learned_join_keys = Column(JSON, nullable=False, default=list)

    # Rolling runtime-cost summary fed back from execution (cost_recorder):
    #   {observations, avg_query_ms, p95_query_ms, max_query_ms, avg_rows,
    #    avg_result_bytes, last_query_ms, slow(bool), updated_at}
    cost_summary = Column(JSON, nullable=True)

    # Monotonic version bumped whenever the profiling algorithm changes, so a
    # consumer can ignore profiles produced by an older profiler.
    profile_version = Column(Integer, nullable=False, default=1)
    # "ok" | "partial" | "failed" — partial means some sampling queries were
    # skipped (timeout/permission) but the row is still usable.
    status = Column(String, nullable=False, default="ok")
    profiled_at = Column(DateTime, nullable=True)

    connection = relationship("Connection")

    __table_args__ = (
        Index("ix_table_profiles_conn_fqn", "connection_id", "table_fqn", unique=True),
    )
