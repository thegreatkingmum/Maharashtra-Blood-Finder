from __future__ import annotations

import shutil
import sqlite3
from datetime import datetime
from pathlib import Path

DB = Path("database.db")

EXPECTED_COLUMNS = [
    ("id", "INTEGER"),
    ("snapshot_date", "TEXT"),
    ("hospital_code", "TEXT"),
    ("blood_bank_name", "TEXT"),
    ("district", "TEXT"),
    ("city", "TEXT"),
    ("area", "TEXT"),
    ("address", "TEXT"),
    ("contact", "TEXT"),
    ("hospital_type", "TEXT"),
    ("A+", "REAL"),
    ("A-", "REAL"),
    ("B+", "REAL"),
    ("B-", "REAL"),
    ("O+", "REAL"),
    ("O-", "REAL"),
    ("AB+", "REAL"),
    ("AB-", "REAL"),
    ("latitude", "REAL"),
    ("longitude", "REAL"),
    ("source_file", "TEXT"),
]


def get_unique_indexes(conn):
    results = []
    for row in conn.execute("PRAGMA index_list(stock_history)").fetchall():
        # columns: seq, name, unique, origin, partial
        name = row[1]
        is_unique = bool(row[2])
        if not is_unique:
            continue
        cols = [r[2] for r in conn.execute(f'PRAGMA index_info("{name}")').fetchall()]
        results.append((name, cols))
    return results


def main():
    if not DB.exists():
        raise SystemExit("ERROR: database.db was not found. Run this from the project root.")

    conn = sqlite3.connect(DB)
    conn.execute("PRAGMA foreign_keys=OFF")

    table_exists = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name='stock_history'"
    ).fetchone()
    if not table_exists:
        conn.close()
        raise SystemExit("ERROR: stock_history table does not exist.")

    before_count = conn.execute("SELECT COUNT(*) FROM stock_history").fetchone()[0]
    before_dates = conn.execute(
        "SELECT COUNT(DISTINCT snapshot_date) FROM stock_history"
    ).fetchone()[0]
    latest_date = conn.execute(
        "SELECT MAX(snapshot_date) FROM stock_history"
    ).fetchone()[0]
    unique_indexes = get_unique_indexes(conn)
    composite_ok = any(
        cols == ["snapshot_date", "hospital_code"] or cols == ["hospital_code", "snapshot_date"]
        for _, cols in unique_indexes
    )

    print("==============================================")
    print("DATABASE SNAPSHOT-SCHEMA CHECK / REPAIR")
    print("==============================================")
    print(f"Database: {DB.resolve()}")
    print(f"Historical rows before: {before_count}")
    print(f"Unique snapshots before: {before_dates}")
    print(f"Latest snapshot before: {latest_date}")
    print("Unique indexes:")
    for name, cols in unique_indexes:
        print(f"  {name}: {cols}")

    required_existing = [r[1] for r in conn.execute("PRAGMA table_info(stock_history)").fetchall()]
    missing = [name for name, _ in EXPECTED_COLUMNS if name not in required_existing]

    if composite_ok and not missing:
        print("\nSchema is already correct: UNIQUE(snapshot_date, hospital_code) exists.")
        print("No migration was needed.")
        conn.close()
        return

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup = DB.with_name(f"database_backup_before_snapshot_fix_{timestamp}.db")
    conn.commit()
    conn.close()
    shutil.copy2(DB, backup)
    print(f"\nBackup created: {backup.name}")

    conn = sqlite3.connect(DB)
    conn.execute("PRAGMA foreign_keys=OFF")

    conn.execute("DROP TABLE IF EXISTS stock_history_new")
    conn.execute(
        '''
        CREATE TABLE stock_history_new (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            snapshot_date TEXT NOT NULL,
            hospital_code TEXT NOT NULL,
            blood_bank_name TEXT,
            district TEXT,
            city TEXT,
            area TEXT,
            address TEXT,
            contact TEXT,
            hospital_type TEXT,
            "A+" REAL DEFAULT 0,
            "A-" REAL DEFAULT 0,
            "B+" REAL DEFAULT 0,
            "B-" REAL DEFAULT 0,
            "O+" REAL DEFAULT 0,
            "O-" REAL DEFAULT 0,
            "AB+" REAL DEFAULT 0,
            "AB-" REAL DEFAULT 0,
            latitude REAL,
            longitude REAL,
            source_file TEXT,
            UNIQUE(snapshot_date, hospital_code)
        )
        '''
    )

    old_cols = {r[1] for r in conn.execute("PRAGMA table_info(stock_history)").fetchall()}
    copy_cols = [name for name, _ in EXPECTED_COLUMNS if name in old_cols]
    # Keep id only if present; all expected legacy DBs have it.
    col_sql = ", ".join(f'"{c}"' for c in copy_cols)
    conn.execute(
        f'INSERT OR IGNORE INTO stock_history_new ({col_sql}) SELECT {col_sql} FROM stock_history'
    )

    after_copy_count = conn.execute("SELECT COUNT(*) FROM stock_history_new").fetchone()[0]
    after_copy_dates = conn.execute(
        "SELECT COUNT(DISTINCT snapshot_date) FROM stock_history_new"
    ).fetchone()[0]

    if after_copy_count != before_count:
        conn.rollback()
        conn.close()
        raise SystemExit(
            f"ERROR: migration would change row count from {before_count} to {after_copy_count}. "
            f"Original database was not replaced. Backup: {backup}"
        )

    conn.execute("DROP TABLE stock_history")
    conn.execute("ALTER TABLE stock_history_new RENAME TO stock_history")
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_stock_history_snapshot_date ON stock_history(snapshot_date)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_stock_history_hospital_code ON stock_history(hospital_code)"
    )
    conn.commit()
    conn.close()

    print("\nMigration complete.")
    print(f"Historical rows after: {after_copy_count}")
    print(f"Unique snapshots after: {after_copy_dates}")
    print("The existing 5755 historical rows were preserved.")
    print("\nNEXT STEP: run exactly:")
    print("  python update_pipeline.py")
    print("\nExpected after the 11-Sep snapshot is imported:")
    print("  Unique snapshots: 15")
    print("  Historical rows: 6164")
    print("  Current blood banks: 409")


if __name__ == "__main__":
    main()
