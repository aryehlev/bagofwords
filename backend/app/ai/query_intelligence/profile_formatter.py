"""Render learned data profiles into a compact prompt section.

This is the "feed stats into generation" bridge: the profiler learns real value
dictionaries, join keys, uniqueness and cost; this turns them into a few terse
lines the coder can act on (filter with real literals, join on the right key,
avoid scanning a known-slow table). Kept deliberately compact — profiles are
advisory grounding, not the schema itself.
"""

from __future__ import annotations

from typing import Any, Optional

from app.ai.query_intelligence.config import QIConfig, get_qi_config


def _get(profile: Any, key: str, default=None):
    if isinstance(profile, dict):
        return profile.get(key, default)
    return getattr(profile, key, default)


def format_profile(profile: Any, *, config: Optional[QIConfig] = None) -> str:
    """Render one table's profile as indented lines, or "" if nothing useful."""
    cfg = config or get_qi_config()
    name = _get(profile, "table_fqn") or _get(profile, "name") or "table"
    lines: list[str] = []

    row_count = _get(profile, "row_count_estimate")
    if row_count is not None:
        lines.append(f"    rows≈{row_count}")

    cost = _get(profile, "cost_summary")
    if isinstance(cost, dict) and cost.get("observations"):
        p95 = cost.get("p95_query_ms")
        avg = cost.get("avg_query_ms")
        tag = " [SLOW — filter aggressively, avoid SELECT *]" if cost.get("slow") else ""
        timing = p95 if p95 is not None else avg
        if timing is not None:
            lines.append(f"    observed query time≈{timing}ms{tag}")

    unique_cols = _get(profile, "unique_columns") or []
    if unique_cols:
        lines.append(f"    likely unique: {', '.join(str(c) for c in unique_cols[:8])}")

    joins = _get(profile, "learned_join_keys") or []
    shown_joins = [j for j in joins if isinstance(j, dict)][:6]
    for j in shown_joins:
        conf = j.get("confidence")
        conf_s = f" (~{conf})" if conf is not None else ""
        lines.append(
            f"    join: {j.get('left_column')} -> {j.get('right_table')}.{j.get('right_column')}{conf_s}"
        )

    value_dicts = _get(profile, "value_dictionaries") or {}
    if isinstance(value_dicts, dict):
        for col, values in list(value_dicts.items())[: cfg.max_profiled_columns]:
            if not values:
                continue
            shown = values[: cfg.max_value_dict_values]
            rendered = ", ".join(repr(v) for v in shown)
            more = "" if len(values) <= len(shown) else f", … (+{len(values) - len(shown)})"
            lines.append(f"    {col} ∈ {{{rendered}{more}}}")

    if not lines:
        return ""
    return f"-- profile: {name}\n" + "\n".join(lines)


def format_profiles(profiles: list[Any], *, config: Optional[QIConfig] = None) -> str:
    """Render a section for many tables. Returns "" when nothing to show."""
    cfg = config or get_qi_config()
    blocks = [format_profile(p, config=cfg) for p in (profiles or [])]
    blocks = [b for b in blocks if b]
    if not blocks:
        return ""
    header = (
        "Learned data profile (sampled from real rows — use real literal values "
        "for filters, the join keys below, and treat SLOW tables carefully):"
    )
    return header + "\n\n" + "\n\n".join(blocks)
