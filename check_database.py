# ============================================================
# CHECK MAHARASHTRA BLOOD FINDER DATABASE
# ============================================================

import sqlite3
from pathlib import Path
import pandas as pd


DATABASE = Path("database.db")


# ============================================================
# CONNECT
# ============================================================

if not DATABASE.exists():

    raise FileNotFoundError(
        "database.db was not found."
    )


conn = sqlite3.connect(
    DATABASE
)


# ============================================================
# BASIC COUNTS
# ============================================================

history_count = pd.read_sql_query(
    """
    SELECT COUNT(*) AS count
    FROM stock_history
    """,
    conn
).iloc[0, 0]


current_count = pd.read_sql_query(
    """
    SELECT COUNT(*) AS count
    FROM blood_banks
    """,
    conn
).iloc[0, 0]


snapshot_count = pd.read_sql_query(
    """
    SELECT COUNT(
        DISTINCT snapshot_date
    ) AS count
    FROM stock_history
    """,
    conn
).iloc[0, 0]


print("\n==============================================")
print("DATABASE CHECK")
print("==============================================")

print(
    "Historical records:",
    history_count
)

print(
    "Current blood banks:",
    current_count
)

print(
    "Unique snapshots:",
    snapshot_count
)


# ============================================================
# SNAPSHOT INFORMATION
# ============================================================

snapshots = pd.read_sql_query(
    """
    SELECT
        snapshot_date,
        COUNT(*) AS bank_count
    FROM stock_history
    GROUP BY snapshot_date
    ORDER BY snapshot_date
    """,
    conn
)

print("\nSnapshot summary:")

print(
    snapshots.to_string(
        index=False
    )
)


# ============================================================
# CURRENT STOCK SAMPLE
# ============================================================

current = pd.read_sql_query(
    """
    SELECT
        hospital_code,
        blood_bank_name,
        district,
        area,
        city,
        "A+",
        "A-",
        "B+",
        "B-",
        "O+",
        "O-",
        "AB+",
        "AB-",
        snapshot_date
    FROM blood_banks
    ORDER BY district, city, blood_bank_name
    LIMIT 10
    """,
    conn
)

print("\nCurrent stock sample:")

print(
    current.to_string(
        index=False
    )
)


# ============================================================
# NEGATIVE STOCK CHECK
# ============================================================

negative_checks = {}

for group in [
    "A+", "A-", "B+", "B-",
    "O+", "O-", "AB+", "AB-"
]:

    result = pd.read_sql_query(
        f"""
        SELECT COUNT(*) AS count
        FROM blood_banks
        WHERE "{group}" < 0
        """,
        conn
    ).iloc[0, 0]

    negative_checks[group] = result


print("\nNegative stock counts:")

print(
    negative_checks
)


# ============================================================
# LOCATION COMPLETENESS
# ============================================================

location_check = pd.read_sql_query(
    """
    SELECT

        SUM(
            CASE
                WHEN district IS NULL
                  OR TRIM(district) = ''
                THEN 1
                ELSE 0
            END
        ) AS missing_district,

        SUM(
            CASE
                WHEN city IS NULL
                  OR TRIM(city) = ''
                THEN 1
                ELSE 0
            END
        ) AS missing_city,

        SUM(
            CASE
                WHEN area IS NULL
                  OR TRIM(area) = ''
                THEN 1
                ELSE 0
            END
        ) AS missing_area,

        SUM(
            CASE
                WHEN latitude IS NULL
                THEN 1
                ELSE 0
            END
        ) AS missing_latitude,

        SUM(
            CASE
                WHEN longitude IS NULL
                THEN 1
                ELSE 0
            END
        ) AS missing_longitude

    FROM blood_banks
    """,
    conn
)

print("\nLocation completeness:")

print(
    location_check.to_string(
        index=False
    )
)


# ============================================================
# CLOSE
# ============================================================

conn.close()

print(
    "\nDatabase check complete."
)