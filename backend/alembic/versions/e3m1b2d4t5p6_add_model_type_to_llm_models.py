"""add model_type + embedding_dim to llm_models

Revision ID: e3m1b2d4t5p6
Revises: d6d9a78b7b4a
Create Date: 2026-06-23 00:00:00.000000

Adds an explicit role to LLM models so embedding models can coexist with chat
models:

- ``llm_models.model_type`` ('chat' | 'embedding'): defaults to 'chat' so all
  existing rows keep driving inference unchanged.
- ``llm_models.embedding_dim``: fixed vector width for embedding models.

The per-org "default embedding model" pointer lives in
``organization_settings.config`` (key ``default_embedding_model_id``) and needs
no schema change.
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = 'e3m1b2d4t5p6'
down_revision: Union[str, None] = 'd6d9a78b7b4a'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        'llm_models',
        sa.Column('model_type', sa.String(), nullable=False, server_default='chat'),
    )
    op.add_column(
        'llm_models',
        sa.Column('embedding_dim', sa.Integer(), nullable=True),
    )


def downgrade() -> None:
    bind = op.get_bind()
    if bind.dialect.name == 'sqlite':
        with op.batch_alter_table('llm_models') as batch_op:
            batch_op.drop_column('embedding_dim')
            batch_op.drop_column('model_type')
    else:
        op.drop_column('llm_models', 'embedding_dim')
        op.drop_column('llm_models', 'model_type')
