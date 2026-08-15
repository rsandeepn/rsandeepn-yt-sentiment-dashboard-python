import os

from dotenv import load_dotenv
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.orm import DeclarativeBase, sessionmaker


load_dotenv()
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./youtube_analyzer.db")

connect_args = {"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}
engine = create_engine(DATABASE_URL, connect_args=connect_args, pool_pre_ping=True)
SessionLocal = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)


class Base(DeclarativeBase):
    pass


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def ensure_analysis_job_columns():
    """Add job lifecycle columns for databases created before this milestone."""
    existing = {column["name"] for column in inspect(engine).get_columns("analyses")}
    dialect = engine.dialect.name
    timestamp_type = "TIMESTAMP WITH TIME ZONE" if dialect == "postgresql" else "DATETIME"
    additions = {
        "status": "VARCHAR(20) NOT NULL DEFAULT 'completed'",
        "progress": "INTEGER NOT NULL DEFAULT 100",
        "status_message": "VARCHAR(255)",
        "error_message": "TEXT",
        "updated_at": f"{timestamp_type}",
        "completed_at": f"{timestamp_type}",
    }
    missing = [(name, definition) for name, definition in additions.items() if name not in existing]
    if not missing:
        return

    with engine.begin() as connection:
        for name, definition in missing:
            connection.execute(text(f"ALTER TABLE analyses ADD COLUMN {name} {definition}"))
        connection.execute(
            text(
                "UPDATE analyses SET updated_at = created_at, completed_at = created_at "
                "WHERE updated_at IS NULL"
            )
        )


def ensure_analysis_source_columns():
    """Add cross-platform source fields and backfill existing YouTube analyses."""
    existing = {column["name"] for column in inspect(engine).get_columns("analyses")}
    additions = {
        "platform": "VARCHAR(20) NOT NULL DEFAULT 'youtube'",
        "content_type": "VARCHAR(20) NOT NULL DEFAULT 'video'",
        "content_id": "VARCHAR(255)",
        "content_url": "TEXT",
    }
    missing = [(name, definition) for name, definition in additions.items() if name not in existing]

    with engine.begin() as connection:
        for name, definition in missing:
            connection.execute(text(f"ALTER TABLE analyses ADD COLUMN {name} {definition}"))
        connection.execute(
            text(
                "UPDATE analyses SET "
                "platform = COALESCE(NULLIF(platform, ''), 'youtube'), "
                "content_type = COALESCE(NULLIF(content_type, ''), 'video'), "
                "content_id = COALESCE(NULLIF(content_id, ''), video_id), "
                "content_url = COALESCE(NULLIF(content_url, ''), video_url)"
            )
        )
        connection.execute(
            text("CREATE INDEX IF NOT EXISTS ix_analyses_platform ON analyses (platform)")
        )
        connection.execute(
            text("CREATE INDEX IF NOT EXISTS ix_analyses_content_id ON analyses (content_id)")
        )


def ensure_user_profile_columns():
    """Add profile fields without requiring existing Docker data to be reset."""
    existing = {column["name"] for column in inspect(engine).get_columns("users")}
    additions = {
        "first_name": "VARCHAR(100)",
        "last_name": "VARCHAR(100)",
    }
    missing = [(name, definition) for name, definition in additions.items() if name not in existing]
    if not missing:
        return

    with engine.begin() as connection:
        for name, definition in missing:
            connection.execute(text(f"ALTER TABLE users ADD COLUMN {name} {definition}"))


def ensure_user_security_columns():
    """Add password-reset session invalidation fields to existing databases."""
    existing = {column["name"] for column in inspect(engine).get_columns("users")}
    if "auth_version" in existing:
        return

    with engine.begin() as connection:
        connection.execute(
            text("ALTER TABLE users ADD COLUMN auth_version INTEGER NOT NULL DEFAULT 0")
        )
