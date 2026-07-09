"""Ambient attribution for LLM usage records.

LLM calls happen deep in the agent / tool stack, far from where we know *who*
triggered the run and *against which report / data source*. Threading that
context through every ``LLM(...)`` constructor and ``inference*`` call site would
touch dozens of files. Instead the orchestrator (AgentV2) stamps a single
contextvar at the start of a run; the usage recorder reads it when it persists a
record.

Why a contextvar (and not thread-locals): the agent run is async, and the
recorder is scheduled with ``loop.create_task`` / ``run_coroutine_threadsafe``.
We read the value *synchronously at schedule time* — i.e. still inside the LLM
call's own context — and close over the snapshot, so it survives the hop onto a
background task or worker thread regardless of contextvar copy semantics.
"""

from __future__ import annotations

import contextlib
from contextvars import ContextVar
from typing import Optional, TypedDict


class UsageAttribution(TypedDict, total=False):
    organization_id: Optional[str]
    user_id: Optional[str]
    report_id: Optional[str]
    data_source_id: Optional[str]


_current_attribution: ContextVar[Optional[UsageAttribution]] = ContextVar(
    "llm_usage_attribution", default=None
)


def get_usage_attribution() -> UsageAttribution:
    """Return a snapshot of the current attribution (empty dict if unset)."""
    value = _current_attribution.get()
    return dict(value) if value else {}


def set_usage_attribution(attribution: Optional[UsageAttribution]):
    """Set the ambient attribution, returning the contextvar token for reset."""
    return _current_attribution.set(attribution or None)


def reset_usage_attribution(token) -> None:
    with contextlib.suppress(Exception):
        _current_attribution.reset(token)
