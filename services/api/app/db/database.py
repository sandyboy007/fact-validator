from sqlmodel import SQLModel, create_engine, Session
import os
from pathlib import Path


def _resolve_db_path() -> str:
    # Canonical env var is FACT_VALIDATOR_DB.
    # Keep FACTVALIDATOR_DB as a legacy fallback for compatibility.
    db_from_env = (
        os.getenv("FACT_VALIDATOR_DB", "").strip()
        or os.getenv("FACTVALIDATOR_DB", "").strip()
    )
    if db_from_env:
        return os.path.abspath(db_from_env)

    repo_root = Path(__file__).resolve().parents[4]
    return str((repo_root / "data" / "fact_validator.db").resolve())


DB_PATH = _resolve_db_path()
engine = create_engine(f"sqlite:///{DB_PATH}", echo=False)

def init_db() -> None:
    SQLModel.metadata.create_all(engine)

def get_session() -> Session:
    return Session(engine)
