# ============================================================
# FIX MASTER LOCATION DATA
# Maharashtra Blood Finder
# ============================================================

from pathlib import Path
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent

MASTER_FILE = (
    BASE_DIR
    / "data"
    / "master_blood_banks.csv"
)


# ============================================================
# LOAD MASTER
# ============================================================

if not MASTER_FILE.exists():

    raise FileNotFoundError(
        f"Master file not found:\n{MASTER_FILE}"
    )


df = pd.read_csv(
    MASTER_FILE
)


# ============================================================
# NORMALIZE HOSPITAL CODE
# ============================================================

df["hospital_code"] = (
    df["hospital_code"]
    .astype(str)
    .str.strip()
)


# ============================================================
# SHOW CURRENT RECORD
# ============================================================

target_code = "284016"

record = df[
    df["hospital_code"] == target_code
]

print("\nCurrent record:")

print(
    record[
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
# CORRECT AREA
# ============================================================

mask = (
    df["hospital_code"] == target_code
)

df.loc[
    mask,
    "area"
] = "Vile Parle"


# ============================================================
# VALIDATION
# ============================================================

updated_record = df[
    df["hospital_code"] == target_code
]

print("\nUpdated record:")

print(
    updated_record[
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
# SAVE
# ============================================================

df.to_csv(
    MASTER_FILE,
    index=False
)

print(
    "\nMaster file updated successfully."
)

print(
    f"Saved to:\n{MASTER_FILE}"
)