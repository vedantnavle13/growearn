"""add external identifiers

Revision ID: cf2488ae469d
Revises: 84d8d33414eb
Create Date: 2026-08-26 17:03:22.696826

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
import pgvector


# revision identifiers, used by Alembic.
revision: str = 'cf2488ae469d'
down_revision: Union[str, Sequence[str], None] = '84d8d33414eb'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.add_column('products', sa.Column('external_product_id', sa.String(length=255), nullable=True))
    op.create_index(op.f('ix_products_external_product_id'), 'products', ['external_product_id'], unique=False)
    op.create_unique_constraint('uq_products_merchant_external_id', 'products', ['merchant_id', 'external_product_id'])

    op.add_column('product_variants', sa.Column('external_variant_id', sa.String(length=255), nullable=True))
    op.create_index(op.f('ix_product_variants_external_variant_id'), 'product_variants', ['external_variant_id'], unique=False)

    op.add_column('customers', sa.Column('external_customer_id', sa.String(length=255), nullable=True))
    op.create_index(op.f('ix_customers_external_customer_id'), 'customers', ['external_customer_id'], unique=False)
    op.create_unique_constraint('uq_customers_merchant_external_id', 'customers', ['merchant_id', 'external_customer_id'])


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_constraint('uq_customers_merchant_external_id', 'customers', type_='unique')
    op.drop_index(op.f('ix_customers_external_customer_id'), table_name='customers')
    op.drop_column('customers', 'external_customer_id')

    op.drop_index(op.f('ix_product_variants_external_variant_id'), table_name='product_variants')
    op.drop_column('product_variants', 'external_variant_id')

    op.drop_constraint('uq_products_merchant_external_id', 'products', type_='unique')
    op.drop_index(op.f('ix_products_external_product_id'), table_name='products')
    op.drop_column('products', 'external_product_id')
