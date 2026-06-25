"""Query Intelligence — learn about the data and optimize generated queries.

A best-effort, advisory subsystem that:
  * profiles the actual data (value dictionaries, join keys, uniqueness,
    null/distinct stats) — profiler.py
  * feeds that profile into query generation — profile_formatter.py
  * rewrites/lints generated SQL before it runs — sql_optimizer.py
  * closes the loop on real runtime cost — cost_recorder.py

Design stance mirrors result_lake.py: everything degrades to a no-op. A missing
profile, an unparseable query, or a failed sampling pass only lowers quality —
it never blocks or corrupts query execution. Tunable via QIConfig (env-driven),
OFF by default.
"""

from app.ai.query_intelligence.config import QIConfig, get_qi_config

__all__ = ["QIConfig", "get_qi_config"]
