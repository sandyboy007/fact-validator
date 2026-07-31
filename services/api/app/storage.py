import os
import json
import sqlite3
from typing import Any, Dict, List, Optional
from datetime import datetime
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

    repo_root = Path(__file__).resolve().parents[3]
    return str((repo_root / "data" / "fact_validator.db").resolve())


DB_PATH = _resolve_db_path()


def _ensure_dir(path: str) -> None:
    folder = os.path.dirname(path)
    if folder and not os.path.exists(folder):
        os.makedirs(folder, exist_ok=True)


def init_db() -> None:
    _ensure_dir(DB_PATH)
    with sqlite3.connect(DB_PATH) as con:
        con.execute(
            """
            CREATE TABLE IF NOT EXISTS runs (
              id INTEGER PRIMARY KEY AUTOINCREMENT,
              created_utc TEXT NOT NULL,
              input_type TEXT NOT NULL,
              url TEXT,
              text_preview TEXT,
              domain TEXT,
              mode TEXT,
              verifier TEXT,
              response_json TEXT NOT NULL
            )
            """
        )
        con.execute(
            """
            CREATE TABLE IF NOT EXISTS claim_memory (
              claim_key TEXT PRIMARY KEY,
              updated_utc TEXT NOT NULL,
              result_json TEXT NOT NULL
            )
            """
        )
        con.commit()


def save_run(
    input_type: str,
    url: Optional[str],
    text: Optional[str],
    domain: Optional[str],
    mode: str,
    verifier: str,
    response: Dict[str, Any],
) -> int:
    init_db()
    created_utc = datetime.utcnow().isoformat() + "Z"
    text_preview = (text or "")[:400] if text else None
    response_json = json.dumps(response, ensure_ascii=False)

    with sqlite3.connect(DB_PATH) as con:
        cur = con.execute(
            """
            INSERT INTO runs (created_utc, input_type, url, text_preview, domain, mode, verifier, response_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (created_utc, input_type, url, text_preview, domain, mode, verifier, response_json),
        )
        con.commit()
        return int(cur.lastrowid)


def list_runs(limit: int = 50) -> List[Dict[str, Any]]:
    init_db()
    with sqlite3.connect(DB_PATH) as con:
        con.row_factory = sqlite3.Row
        rows = con.execute(
            """
            SELECT id, created_utc, input_type, url, text_preview, domain, mode, verifier
            FROM runs
            ORDER BY id DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
        return [dict(r) for r in rows]


def get_run(run_id: int) -> Optional[Dict[str, Any]]:
    init_db()
    with sqlite3.connect(DB_PATH) as con:
        con.row_factory = sqlite3.Row
        row = con.execute("SELECT * FROM runs WHERE id = ?", (run_id,)).fetchone()
        if not row:
            return None
        d = dict(row)
        try:
            d["response"] = json.loads(d.get("response_json") or "{}")
        except Exception:
            d["response"] = {}
        return d


def export_runs(limit: int = 500) -> List[Dict[str, Any]]:
    init_db()
    with sqlite3.connect(DB_PATH) as con:
        con.row_factory = sqlite3.Row
        rows = con.execute(
            """
            SELECT id, created_utc, input_type, url, text_preview, domain, mode, verifier, response_json
            FROM runs
            ORDER BY id DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()

        out: List[Dict[str, Any]] = []
        for r in rows:
            d = dict(r)
            try:
                d["response"] = json.loads(d.get("response_json") or "{}")
            except Exception:
                d["response"] = {}
            out.append(d)
        return out


def get_claim_memory(claim_key: str) -> Optional[Dict[str, Any]]:
    init_db()
    with sqlite3.connect(DB_PATH) as con:
        row = con.execute(
            "SELECT updated_utc, result_json FROM claim_memory WHERE claim_key = ?",
            (claim_key,),
        ).fetchone()
        if not row:
            return None
        updated_utc, result_json = row
        try:
            payload = json.loads(result_json or "{}")
        except Exception:
            payload = {}
        return {
            "claim_key": claim_key,
            "updated_utc": updated_utc,
            "payload": payload,
        }


def save_claim_memory(claim_key: str, payload: Dict[str, Any]) -> None:
    init_db()
    updated_utc = datetime.utcnow().isoformat() + "Z"
    result_json = json.dumps(payload, ensure_ascii=False)
    with sqlite3.connect(DB_PATH) as con:
        con.execute(
            """
            INSERT INTO claim_memory (claim_key, updated_utc, result_json)
            VALUES (?, ?, ?)
            ON CONFLICT(claim_key)
            DO UPDATE SET updated_utc = excluded.updated_utc,
                          result_json = excluded.result_json
            """,
            (claim_key, updated_utc, result_json),
        )
        con.commit()
