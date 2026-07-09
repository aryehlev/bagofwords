"""Environment-driven configuration for the query-intelligence subsystem.

Mirrors the philosophy of result_lake.CacheConfig: everything is OFF by default
and individually togglable, so the feature can be rolled out one capability at a
time without code changes.
"""

from __future__ import annotations

import os
from dataclasses import dataclass


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ[name])
    except (KeyError, ValueError):
        return default


@dataclass(frozen=True)
class QIConfig:
    # Master switch. When False every entry point is a no-op regardless of the
    # finer-grained flags below.
    enabled: bool = False

    # Rewrite generated SQL before execution (sql_optimizer). Only provably
    # semantics-preserving rewrites are ever applied.
    rewrite_enabled: bool = False
    # Inject a safety LIMIT into non-aggregating, un-LIMITed SELECTs.
    safety_limit_enabled: bool = True
    safety_limit_rows: int = 100_000

    # Feed learned data profiles into the coder prompt (profile_formatter).
    profile_in_prompt_enabled: bool = True
    # Cap how much profile text we render per table to keep the prompt bounded.
    max_value_dict_values: int = 25
    max_profiled_columns: int = 60

    # Record real query costs back onto profiles (cost_recorder).
    cost_feedback_enabled: bool = True
    # A table whose p95 query time exceeds this is flagged "slow" in the prompt.
    slow_query_ms: float = 4000.0

    # --- Profiling (sampling) knobs ---
    # Rows to pull per table when sampling for column stats.
    profile_sample_rows: int = 5_000
    # A column with <= this many distinct values gets a value dictionary.
    low_cardinality_max: int = 50
    # Skip building dictionaries for tables larger than this (rough guard).
    profile_max_table_rows: int = 50_000_000
    # Per sampling query wall-clock budget.
    profile_query_timeout_seconds: int = 20

    @classmethod
    def from_env(cls) -> "QIConfig":
        return cls(
            enabled=_env_bool("BOW_QUERY_INTEL_ENABLED", False),
            rewrite_enabled=_env_bool("BOW_QUERY_INTEL_REWRITE", False),
            safety_limit_enabled=_env_bool("BOW_QUERY_INTEL_SAFETY_LIMIT", True),
            safety_limit_rows=_env_int("BOW_QUERY_INTEL_SAFETY_LIMIT_ROWS", 100_000),
            profile_in_prompt_enabled=_env_bool("BOW_QUERY_INTEL_PROFILE_PROMPT", True),
            max_value_dict_values=_env_int("BOW_QUERY_INTEL_MAX_DICT_VALUES", 25),
            max_profiled_columns=_env_int("BOW_QUERY_INTEL_MAX_COLUMNS", 60),
            cost_feedback_enabled=_env_bool("BOW_QUERY_INTEL_COST_FEEDBACK", True),
            slow_query_ms=float(_env_int("BOW_QUERY_INTEL_SLOW_MS", 4000)),
            profile_sample_rows=_env_int("BOW_QUERY_INTEL_SAMPLE_ROWS", 5_000),
            low_cardinality_max=_env_int("BOW_QUERY_INTEL_LOW_CARD_MAX", 50),
            profile_max_table_rows=_env_int("BOW_QUERY_INTEL_MAX_TABLE_ROWS", 50_000_000),
            profile_query_timeout_seconds=_env_int("BOW_QUERY_INTEL_PROFILE_TIMEOUT", 20),
        )


_CONFIG: QIConfig | None = None


def get_qi_config() -> QIConfig:
    """Process-wide singleton, read once from the environment."""
    global _CONFIG
    if _CONFIG is None:
        _CONFIG = QIConfig.from_env()
    return _CONFIG
