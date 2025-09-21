from logging.config import fileConfig
from sqlalchemy import engine_from_config, pool
from alembic import context
import os
import sys
from dotenv import load_dotenv

# Ensure the app package is importable: add project root (~/cvitx/api) to sys.path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))            # .../api/migrations
API_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))         # .../api
if API_ROOT not in sys.path:
    sys.path.insert(0, API_ROOT)

# Load .env from project root (../.env)
load_dotenv(os.path.join(API_ROOT, ".env"))

config = context.config
if config.config_file_name:
    fileConfig(config.config_file_name)

# Import your SQLAlchemy Base metadata
from app.models import Base
target_metadata = Base.metadata

def include_object(object, name, type_, reflected, compare_to):
    """Prevent Alembic from auto-dropping objects that aren't in models.
    We apply ADD/ALTERs only; skip DROP of reflected tables/FKs."""
    # Skip DROPs of tables that exist in DB but not in metadata
    if type_ == 'table' and reflected and compare_to is None:
        return False
    # Skip DROPs of foreign key constraints that exist in DB but not in metadata
    if type_ == 'foreign_key_constraint' and reflected and compare_to is None:
        return False
    return True
def run_migrations_offline():
    context.configure(
        url=os.getenv("DB_URL"),
        target_metadata=target_metadata,
        literal_binds=True,
        compare_type=True,
        compare_server_default=True,
        include_object=include_object,
    )
    with context.begin_transaction():
        context.run_migrations()

def run_migrations_online():
    connectable = engine_from_config(
        {"sqlalchemy.url": os.getenv("DB_URL")},
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )
    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            compare_type=True,
            compare_server_default=True,
            include_object=include_object,
        )
        with context.begin_transaction():
            context.run_migrations()

if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
