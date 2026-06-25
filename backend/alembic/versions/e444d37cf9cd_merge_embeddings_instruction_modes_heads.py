"""merge embeddings + instruction modes heads

Revision ID: e444d37cf9cd
Revises: a1c2m0de9f01, e4m2b3d5t6p7
Create Date: 2026-06-25 08:57:10.584427

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'e444d37cf9cd'
down_revision: Union[str, None] = ('a1c2m0de9f01', 'e4m2b3d5t6p7')
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    pass


def downgrade() -> None:
    pass
