"""merge upstream perf-index head with fork embeddings head

Revision ID: mrgperf01
Revises: ff8803de3eb2, perfidx01
Create Date: 2026-07-02 00:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = 'mrgperf01'
down_revision: Union[str, None] = ('ff8803de3eb2', 'perfidx01')
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

def upgrade() -> None:
    pass

def downgrade() -> None:
    pass
