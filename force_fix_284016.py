# ============================================================
# FORCE LOCATION FIX — HOSPITAL 284016
# Maharashtra Blood Finder
# ============================================================

from pathlib import Path
import sqlite3
import pandas as pd


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

DATABASE_FILE = BASE_DIR / "database.db"

MASTER_CSV = (
    BASE_DIR
    / "data"
    / "master_blood_banks.csv"
)


# ============================================================
# TARGET
# ============================================================

HOSPITAL_CODE = "284016"

CORRECT_AREA = "Vile Parle"


# ============================================================
# CHECK FILES
# ============================================================

if not DATABASE_FILE.exists():

    raise FileNotFoundError(
        f"Database not found:\n{DATABASE_FILE}"
    )

if not MASTER_CSV.exists():

    raise FileNotFoundError(
        f"Master CSV not found:\n{MASTER_CSV}"
    )


# ============================================================
# FIX MASTER CSV
# ============================================================

master = pd.read_csv(
    MASTER_CSV
)

master["hospital_code"] = (
    master["hospital_code"]
    .astype(str)
    .str.strip()
)

target = master[
    master["hospital_code"] == HOSPITAL_CODE
]

print("\n==============================================")
print("MASTER CSV CHECK")
print("==============================================")

if target.empty:

    print(
        f"Hospital {HOSPITAL_CODE} "
        "not found in master CSV."
    )

else:

    print(
        "BEFORE:"
    )

    print(
        target[
            [
                "hospital_code",
                "blood_bank_name",
                "district",
                "area",
                "city"
            ]
        ].to_string(index=False)
    )

    # Force correct area.
    master.loc[
        master["hospital_code"] == HOSPITAL_CODE,
        "area"
    ] = CORRECT_AREA

    master.to_csv(
        MASTER_CSV,
        index=False
    )

    print(
        "\nMASTER CSV UPDATED:"
    )

    updated = master[
        master["hospital_code"] == HOSPITAL_CODE
    ]

    print(
        updated[
            [
                "hospital_code",
                "blood_bank_name",
                "district",
                "area",
                "city"
            ]
        ].to_string(index=False)
    )


# ============================================================
# OPEN SQLITE
# ============================================================

conn = sqlite3.connect(
    DATABASE_FILE
)

conn.row_factory = sqlite3.Row


try:

    # ========================================================
    # BEFORE DATABASE
    # ========================================================

    before = conn.execute(
        """
        SELECT
            hospital_code,
            blood_bank_name,
            district,
            area,
            city
        FROM blood_banks
        WHERE CAST(hospital_code AS TEXT) = ?
        """,
        (HOSPITAL_CODE,)
    ).fetchone()


    print("\n==============================================")
    print("DATABASE CHECK")
    print("==============================================")

    print("\nDATABASE BEFORE:")

    if before:

        print(
            dict(before)
        )

    else:

        print(
            f"Hospital {HOSPITAL_CODE} "
            "not found in database."
        )


    # ========================================================
    # FORCE DATABASE UPDATE
    # ========================================================

    cursor = conn.execute(
        """
        UPDATE blood_banks
        SET area = ?
        WHERE CAST(hospital_code AS TEXT) = ?
        """,
        (
            CORRECT_AREA,
            HOSPITAL_CODE
        )
    )


    print(
        "\nDatabase rows updated:",
        cursor.rowcount
    )


    conn.commit()


    # ========================================================
    # DATABASE AFTER
    # ========================================================

    after = conn.execute(
        """
        SELECT
            hospital_code,
            blood_bank_name,
            district,
            area,
            city
        FROM blood_banks
        WHERE CAST(hospital_code AS TEXT) = ?
        """,
        (HOSPITAL_CODE,)
    ).fetchone()


    print(
        "\nDATABASE AFTER:"
    )

    if after:

        print(
            dict(after)
        )

    else:

        print(
            f"Hospital {HOSPITAL_CODE} "
            "not found after update."
        )


    # ========================================================
    # FINAL ASSERTION
    # ========================================================

    if (
        after
        and
        str(after["area"]).strip().lower()
        == CORRECT_AREA.lower()
    ):

        print(
            "\n✅ SUCCESS"
        )

        print(
            f"Hospital {HOSPITAL_CODE} "
            f"area = {after['area']}"
        )

    else:

        raise RuntimeError(
            "Database area was not updated correctly."
        )


finally:

    conn.close()


print(
    "\n=============================================="
)

print(
    "FORCE FIX COMPLETE"
)

print(
    "=============================================="
)