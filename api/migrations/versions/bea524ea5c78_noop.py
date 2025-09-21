"""noop

Revision ID: bea524ea5c78
Revises: 72a6c10dcc9a
Create Date: 2025-09-21 15:52:00

"""
from typing import Sequence, Union
from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision: str = 'bea524ea5c78'
down_revision: Union[str, Sequence[str], None] = '72a6c10dcc9a'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

def upgrade() -> None:
    pass

def downgrade() -> None:
    pass
