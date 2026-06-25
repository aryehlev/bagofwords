# Query Intelligence — learn about the data, optimize the queries

> Status: opt-in, OFF by default. Master switch `BOW_QUERY_INTEL_ENABLED=1`.

The agent generates Python that calls `ds_clients[...].execute_query(sql)`. Until
now the SQL it writes is grounded only in the *structural* schema (column
names/types) plus usage/feedback ranking (`TableStats`). Query Intelligence adds
a feedback loop grounded in the **actual data** and the **actual runtime cost**:

1. **Learn the data** — sample each table and persist value dictionaries, join
   keys, uniqueness and row counts (`TableProfile`).
2. **Feed it into generation** — render that profile into the coder prompt so the
   model filters with real literals, joins on the right keys, and is warned about
   slow tables.
3. **Rewrite + lint the SQL** — a sqlglot pass applies provably-safe rewrites and
   collects advisory warnings before the query runs.
4. **Close the loop on cost** — real per-query timings roll back onto the profile
   so "this table is slow" is learned, not guessed.

Design stance mirrors [`query-result-cache.md`](./query-result-cache.md): every
path degrades to a no-op. A missing profile, an unparseable query, or a failed
sampling pass only lowers generation quality — it never blocks or corrupts
execution.

## Components

| Concern | Module |
| --- | --- |
| Config (env flags) | `app/ai/query_intelligence/config.py` |
| SQL rewrite + lint | `app/ai/query_intelligence/sql_optimizer.py` |
| Data profiling (pure + I/O) | `app/ai/query_intelligence/profiler.py` |
| Cost feedback math | `app/ai/query_intelligence/cost_recorder.py` |
| Prompt rendering | `app/ai/query_intelligence/profile_formatter.py` |
| Persistence / orchestration | `app/services/query_intelligence_service.py` |
| Storage | `app/models/table_profile.py` (`table_profiles`) |

`TableProfile` is the third leg of the schema-metadata stool, complementing the
*structural* `ConnectionTable` and the *usage* `TableStats`. It is keyed by
`(connection_id, table_fqn)` and holds `column_profiles`, `value_dictionaries`,
`unique_columns`, `learned_join_keys`, and a rolling `cost_summary`.

## Data flow

```
profile (on demand)         generate (every create_data)          execute
─────────────────           ────────────────────────────         ─────────────────────────
POST /data_sources/{id}/    build_data_profile_context()          QueryCapturingClientWrapper
  profile                     -> get_profiles_for_tables()          .execute_query()
  -> profile_data_source()    -> format_profiles()                  -> optimize_sql()  (rewrite+lint)
    -> DataProfiler             -> CodeGenContext                    -> run, time it
    -> TableProfile rows           .data_profile_context           -> timings
                                 -> coder prompt <data_profile>     -> record_costs()
                                                                       -> merge_cost_summary()
                                                                       -> TableProfile.cost_summary
```

## SQL rewrites (sql_optimizer)

Only two transforms touch the SQL, both conservative:

* **Drop pointless ORDER BY** — an `ORDER BY` in a non-root subquery with no
  `LIMIT`/`OFFSET` has no defined effect on the final result; removing it is
  provably semantics-preserving. Always safe.
* **Safety LIMIT** (guardrail, gated by `BOW_QUERY_INTEL_SAFETY_LIMIT`) — caps an
  unbounded, non-aggregating top-level `SELECT`. This *does* change behavior (it
  truncates), so it lives behind its own flag and is skipped for queries that
  already `LIMIT`, `GROUP BY`, `DISTINCT`, or are bare aggregates.

Rewrites are only emitted when `BOW_QUERY_INTEL_REWRITE=1`; otherwise the pass
still runs and records what it *would* do (plus all lint warnings) without
altering the SQL.

### Lints (advisory, never mutate)

`SELECT *`, possible cartesian products, and — when a `TableProfile` is supplied
— filters on a low-cardinality column against a literal **never observed in the
sampled data** (surfaced with the real candidate values). Warnings ride along on
the per-query timing entry (`query_opt`) so the planner can act on them.

## Profiling (profiler)

Pure compute (`profile_table_from_sample`, `infer_join_keys`) is separated from
I/O (`DataProfiler`) so the math is unit-tested without a database. Per table the
profiler issues at most two read-only queries: a bounded `SELECT * ... LIMIT n`
(dialect-correct — SQL Server gets `TOP`) and an optional `COUNT(*)`. Column
stats, value dictionaries and uniqueness come from the sample via pandas. Join
keys are inferred from `*_id`/`id` name + type alignment to sibling tables'
keys (value-overlap confirmation is intentionally left out to bound the query
budget; results are labeled `source="name+type"`).

## Cost feedback (cost_recorder)

After each `create_data`, the captured `query_timings` are attributed to the
tables their SQL scans (`extract_tables` via sqlglot, CTEs excluded). Cache hits
are excluded from latency so the summary reflects true source cost. A decaying
weighted average folds the new observation into `TableProfile.cost_summary`; a
table whose p95 crosses `BOW_QUERY_INTEL_SLOW_MS` is flagged `slow`, which the
prompt renders as a "filter aggressively, avoid SELECT *" hint.

## Configuration

| Env var | Default | Effect |
| --- | --- | --- |
| `BOW_QUERY_INTEL_ENABLED` | `0` | Master switch. Off ⇒ every path is a no-op. |
| `BOW_QUERY_INTEL_REWRITE` | `0` | Apply SQL rewrites (else dry-run/lint only). |
| `BOW_QUERY_INTEL_SAFETY_LIMIT` | `1` | Enable the safety-LIMIT guardrail. |
| `BOW_QUERY_INTEL_SAFETY_LIMIT_ROWS` | `100000` | Cap rows for the guardrail. |
| `BOW_QUERY_INTEL_PROFILE_PROMPT` | `1` | Inject profiles into the coder prompt. |
| `BOW_QUERY_INTEL_COST_FEEDBACK` | `1` | Roll real timings back onto profiles. |
| `BOW_QUERY_INTEL_SLOW_MS` | `4000` | p95 threshold for the `slow` flag. |
| `BOW_QUERY_INTEL_SAMPLE_ROWS` | `5000` | Rows sampled per table when profiling. |
| `BOW_QUERY_INTEL_LOW_CARD_MAX` | `50` | Max distinct for a value dictionary. |

(The finer-grained flags only take effect while the master switch is on.)

## Rollout

1. Migrate (`alembic upgrade head` → `table_profiles`).
2. Turn on the master switch; profiles render into the prompt and cost feedback
   accumulates immediately (both read-only / additive).
3. Profile data sources via `POST /data_sources/{id}/profile` (idempotent upsert;
   safe to re-run, e.g. after a schema refresh).
4. Once comfortable, enable `BOW_QUERY_INTEL_REWRITE=1` to let the optimizer
   actually rewrite SQL.

## Tests

`tests/unit/test_sql_optimizer.py`, `tests/unit/test_query_profiler.py`,
`tests/unit/test_cost_recorder.py` — all pure, no DB or data source.
