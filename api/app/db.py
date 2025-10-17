from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
import os

engine = create_engine(os.environ["DB_URL"], pool_pre_ping=True, future=True)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)
