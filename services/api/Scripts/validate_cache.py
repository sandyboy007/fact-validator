"""Validate runtime evidence-cache records and optionally quarantine invalid files.

Runtime cache records are not authoritative thesis evidence. This utility makes
cache corruption visible and prevents malformed records from being silently
accepted by the live application.

Usage:
  python services/api/Scripts/validate_cache.py
  python services/api/Scripts/validate_cache.py --quarantine
"""

from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

API_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = API_ROOT.parents[1]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate evidence cache JSON")
    parser.add_argument(
        "--cache-dir",
        default=str(REPO_ROOT / "data" / "cache"),
        help="Directory containing runtime cache JSON files",
    )
    parser.add_argument(
        "--quarantine",
        action="store_true",
        help="Move invalid files into cache-dir/quarantine",
    )
    parser.add_argument(
        "--report",
        default=str(REPO_ROOT / "data" / "cache_validation_report.json"),
        help="JSON report path",
    )
    return parser.parse_args()


def _validate_record(value: Any) -> list[str]:
    errors: list[str] = []
    if not isinstance(value, dict):
        return ["root must be a JSON object"]
    if not isinstance(value.get("ts"), (int, float)):
        errors.append("ts must be numeric")
    if not isinstance(value.get("results"), list):
        errors.append("results must be a list")
    return errors


def _quarantine(path: Path, quarantine_dir: Path) -> Path:
    quarantine_dir.mkdir(parents=True, exist_ok=True)
    destination = quarantine_dir / path.name
    counter = 1
    while destination.exists():
        destination = quarantine_dir / f"{path.stem}.{counter}{path.suffix}"
        counter += 1
    shutil.move(str(path), str(destination))
    return destination


def main() -> int:
    args = _parse_args()
    cache_dir = Path(args.cache_dir).resolve()
    quarantine_dir = cache_dir / "quarantine"
    report_path = Path(args.report).resolve()

    records: list[dict[str, Any]] = []
    for path in sorted(cache_dir.glob("*.json")):
        if path.name == "example_cache_record.json":
            continue
        status = "valid"
        errors: list[str] = []
        destination: str | None = None
        try:
            text = path.read_text(encoding="utf-8", errors="strict")
            value = json.loads(text)
            errors.extend(_validate_record(value))
        except UnicodeDecodeError as exc:
            errors.append(f"invalid UTF-8: {exc}")
        except json.JSONDecodeError as exc:
            errors.append(f"invalid JSON: {exc}")
        except OSError as exc:
            errors.append(f"read error: {exc}")

        if errors:
            status = "invalid"
            if args.quarantine and path.exists():
                destination = str(_quarantine(path, quarantine_dir))
                status = "quarantined"

        records.append(
            {
                "file": str(path),
                "status": status,
                "errors": errors,
                "quarantine_destination": destination,
            }
        )

    summary = {
        "total": len(records),
        "valid": sum(r["status"] == "valid" for r in records),
        "invalid": sum(r["status"] == "invalid" for r in records),
        "quarantined": sum(r["status"] == "quarantined" for r in records),
    }
    report = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "cache_dir": str(cache_dir),
        "quarantine_enabled": bool(args.quarantine),
        "summary": summary,
        "records": records,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(
        "Cache validation: "
        f"{summary['valid']} valid, {summary['invalid']} invalid, "
        f"{summary['quarantined']} quarantined"
    )
    print(f"Report: {report_path}")
    return 1 if summary["invalid"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
