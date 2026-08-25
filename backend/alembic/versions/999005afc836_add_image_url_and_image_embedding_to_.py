"""add_image_url_and_image_embedding_to_products

Revision ID: 999005afc836
Revises: 1e3771cc0e2e
Create Date: 2026-08-25 22:09:40.483099

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
import pgvector.sqlalchemy


# revision identifiers, used by Alembic.
revision: str = '999005afc836'
down_revision: Union[str, Sequence[str], None] = '1e3771cc0e2e'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.add_column('products', sa.Column('image_url', sa.String(length=1024), nullable=True))
    op.add_column('products', sa.Column('image_embedding', pgvector.sqlalchemy.Vector(1536), nullable=True))


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_column('products', 'image_embedding')
    op.drop_column('products', 'image_url')

