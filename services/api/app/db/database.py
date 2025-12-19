from sqlmodel import SQLModel, create_engine, Session
import os

DB_PATH = os.getenv("FACTVALIDATOR_DB", r"C:\Fact_Validator\services\api\data\fact_validator.db")
engine = create_engine(f"sqlite:///{DB_PATH}", echo=False)

def init_db() -> None:
    SQLModel.metadata.create_all(engine)

def get_session() -> Session:
    return Session(engine)
