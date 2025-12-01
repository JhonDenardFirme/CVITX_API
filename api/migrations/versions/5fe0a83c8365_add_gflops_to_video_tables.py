"""add_gflops_to_video_tables

Revision ID: 5fe0a83c8365
Revises: 01ab88d22023
Create Date: 2025-12-01 15:04:15.419048

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '5fe0a83c8365'
down_revision: Union[str, Sequence[str], None] = '01ab88d22023'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    pass


def downgrade() -> None:
    """Downgrade schema."""
    pass
