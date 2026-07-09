"""Guard against parallel feature branches leaving multiple alembic heads.

Two migrations sharing the same down_revision make ``alembic upgrade head``
fail with "Multiple head revisions are present". This walks the versions
directory textually (no alembic env / DB needed) and asserts the graph has
exactly one head.
"""
from __future__ import annotations

import re
from pathlib import Path

VERSIONS_DIR = Path(__file__).resolve().parents[2] / "alembic" / "versions"

_REVISION_RE = re.compile(r"^revision(?::\s*str)?\s*=\s*[\"']([^\"']+)[\"']", re.M)
_DOWN_REVISION_RE = re.compile(r"^down_revision(?::\s*[^=\n]+)?\s*=\s*(.+)$", re.M)


def _load_graph() -> tuple[dict[str, str], dict[str, list[str]]]:
    revisions: dict[str, str] = {}
    parents: dict[str, list[str]] = {}
    for path in VERSIONS_DIR.glob("*.py"):
        text = path.read_text()
        rev_match = _REVISION_RE.search(text)
        if not rev_match:
            continue
        rev = rev_match.group(1)
        revisions[rev] = path.name
        down_match = _DOWN_REVISION_RE.search(text)
        down_raw = down_match.group(1) if down_match else "None"
        parents[rev] = re.findall(r"[\"']([^\"']+)[\"']", down_raw)
    return revisions, parents


def test_versions_dir_exists_and_parses():
    revisions, _ = _load_graph()
    assert len(revisions) > 50  # sanity: the project has many migrations


def test_exactly_one_head():
    revisions, parents = _load_graph()
    referenced = {p for ps in parents.values() for p in ps}
    heads = sorted(r for r in revisions if r not in referenced)
    assert len(heads) == 1, (
        f"multiple alembic heads {heads} "
        f"({[revisions[h] for h in heads]}) — add a merge migration or re-parent"
    )


def test_all_parents_exist():
    revisions, parents = _load_graph()
    for rev, ps in parents.items():
        for p in ps:
            assert p in revisions, f"{revisions[rev]} references unknown down_revision {p!r}"
