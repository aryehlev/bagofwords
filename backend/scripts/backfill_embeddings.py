"""One-off backfill of embeddings for instructions and steps.

Embeds existing published instructions and recent steps so semantic retrieval
has an index to query. Idempotent: rows whose ``content_hash`` is unchanged are
skipped, so it's safe to re-run (e.g. after a model switch, which changes
``model_id`` and forces a re-embed).

Usage:

    cd backend && source .venv/bin/activate
    python scripts/backfill_embeddings.py                # all orgs
    python scripts/backfill_embeddings.py --org-id <id>  # one org
    python scripts/backfill_embeddings.py --days 90 --batch-size 128
"""

from __future__ import annotations

import argparse
import asyncio
from typing import List, Optional, Tuple

from datetime import datetime, timedelta, timezone

from sqlalchemy import select

from app.settings.database import create_async_session_factory
from app.models.organization import Organization
from app.models.instruction import Instruction
from app.models.step import Step
from app.ai.context.semantic_search import SemanticSearch


def _data_model_text(data_model) -> str:
    parts: List[str] = []
    if isinstance(data_model, dict):
        for key in ("title", "name", "description"):
            val = data_model.get(key)
            if isinstance(val, str) and val.strip():
                parts.append(val.strip())
        for c in data_model.get("columns", []) or []:
            if isinstance(c, dict):
                name = c.get("generated_column_name") or c.get("name")
                if name:
                    parts.append(str(name))
    return " ".join(parts)


def _step_text(step: Step) -> str:
    parts = [_data_model_text(step.data_model or {})]
    if getattr(step, "prompt", None):
        parts.append(str(step.prompt))
    if getattr(step, "title", None):
        parts.append(str(step.title))
    return " ".join(p for p in parts if p).strip()


async def _chunked_index(ss: SemanticSearch, owner_type: str,
                         items: List[Tuple[str, str]], batch_size: int) -> int:
    total = 0
    for i in range(0, len(items), batch_size):
        total += await ss.index_texts(owner_type, items[i:i + batch_size])
    return total


async def backfill_org(session, org: Organization, days: int, batch_size: int) -> None:
    ss = SemanticSearch(session, org)

    inst_rows = (
        await session.execute(
            select(Instruction.id, Instruction.text).where(
                Instruction.organization_id == org.id,
                Instruction.status == "published",
                Instruction.deleted_at.is_(None),
            )
        )
    ).all()
    inst_items = [(str(iid), txt or "") for iid, txt in inst_rows]
    n_inst = await _chunked_index(ss, "instruction", inst_items, batch_size)

    since = datetime.now(timezone.utc) - timedelta(days=days) if days > 0 else None
    step_stmt = select(Step)
    if since is not None:
        step_stmt = step_stmt.where(Step.created_at >= since)
    step_rows = (await session.execute(step_stmt.limit(5000))).scalars().all()
    step_items = [(str(s.id), _step_text(s)) for s in step_rows]
    step_items = [(sid, txt) for sid, txt in step_items if txt]
    n_step = await _chunked_index(ss, "step", step_items, batch_size)

    print(f"[org={org.id}] embedded {n_inst} instructions, {n_step} steps")


async def main(org_id: Optional[str], days: int, batch_size: int) -> None:
    session_factory = create_async_session_factory()
    async with session_factory() as session:
        if org_id:
            org = await session.get(Organization, org_id)
            orgs = [org] if org else []
        else:
            orgs = (await session.execute(select(Organization))).scalars().all()
        for org in orgs:
            if org is None:
                continue
            await backfill_org(session, org, days, batch_size)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Backfill semantic embeddings.")
    p.add_argument("--org-id", default=None, help="Limit to a single organization id")
    p.add_argument("--days", type=int, default=180, help="Embed steps created in the last N days (0=all)")
    p.add_argument("--batch-size", type=int, default=128)
    args = p.parse_args()
    asyncio.run(main(args.org_id, args.days, args.batch_size))
