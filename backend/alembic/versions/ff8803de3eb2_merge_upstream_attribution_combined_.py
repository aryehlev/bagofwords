"""merge upstream attribution + combined embeddings heads

Revision ID: ff8803de3eb2
Revises: b1c2d3e4f5a6, e444d37cf9cd
Create Date: 2026-06-25 12:44:13.769734

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'ff8803de3eb2'
down_revision: Union[str, None] = ('b1c2d3e4f5a6', 'e444d37cf9cd')
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    pass


def downgrade() -> None:
    pass
