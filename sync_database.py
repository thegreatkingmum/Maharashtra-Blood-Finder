# ============================================================
# MAHARASHTRA BLOOD FINDER
# DATABASE LOCATION SYNCHRONIZATION
#
# Purpose:
# Synchronize the current SQLite blood_banks table with
# master_blood_banks.csv and coordinates.csv.
#
# This updates:
#   - district
#   - area
#   - city
#   - latitude
#   - longitude
#
# It does NOT change:
#   - current blood stock
#   - historical stock
#   - hospital code
# ============================================================

from pathlib import Path
import sqlite3
import pandas as pd


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

DATABASE_FILE = (
    BASE_DIR / "database.db"
)

MASTER_FILE = (
    BASE_DIR
    / "data"
    / "master_blood_banks.csv"
)

COORDINATE_FILE = (
    BASE_DIR
    / "data"
    / "coordinates.csv"
)


# ============================================================
# CHECK FILES
# ============================================================

for file_path in [
    DATABASE_FILE,
    MASTER_FILE,
    COORDINATE_FILE,
]:

    if not file_path.exists():

        raise FileNotFoundError(
            f"Required file not found:\n{file_path}"
        )


# ============================================================
# LOAD MASTER
# ============================================================

master_df = pd.read_csv(
    MASTER_FILE
)

master_df.columns = [
    str(column).strip()
    for column in master_df.columns
]

required_master_columns = [
    "hospital_code",
    "district",
    "area",
    "city",
]

missing_master_columns = [
    column
    for column in required_master_columns
    if column not in master_df.columns
]

if missing_master_columns:

    raise ValueError(
        "Master CSV is missing columns: "
        f"{missing_master_columns}"
    )


# ============================================================
# NORMALIZE MASTER HOSPITAL CODES
# ============================================================

master_df["hospital_code"] = (
    master_df["hospital_code"]
    .astype(str)
    .str.strip()
)


master_df = (
    master_df
    .drop_duplicates(
        "hospital_code"
    )
    .copy()
)


# ============================================================
# LOAD COORDINATES
# ============================================================

coordinates_df = pd.read_csv(
    COORDINATE_FILE
)

coordinates_df.columns = [
    str(column).strip()
    for column in coordinates_df.columns
]

required_coordinate_columns = [
    "hospital_code",
    "latitude",
    "longitude",
]

missing_coordinate_columns = [
    column
    for column in required_coordinate_columns
    if column not in coordinates_df.columns
]

if missing_coordinate_columns:

    raise ValueError(
        "Coordinates CSV is missing columns: "
        f"{missing_coordinate_columns}"
    )


coordinates_df["hospital_code"] = (
    coordinates_df["hospital_code"]
    .astype(str)
    .str.strip()
)


coordinates_df["latitude"] = pd.to_numeric(
    coordinates_df["latitude"],
    errors="coerce",
)


coordinates_df["longitude"] = pd.to_numeric(
    coordinates_df["longitude"],
    errors="coerce",
)


coordinates_df = (
    coordinates_df
    .drop_duplicates(
        "hospital_code"
    )
    .copy()
)


# ============================================================
# CONNECT DATABASE
# ============================================================

connection = sqlite3.connect(
    DATABASE_FILE
)

connection.row_factory = sqlite3.Row


try:

    # ========================================================
    # CHECK DATABASE TABLE
    # ========================================================

    table_check = connection.execute(
        """
        SELECT name
        FROM sqlite_master
        WHERE type = 'table'
          AND name = 'blood_banks'
        """
    ).fetchone()


    if table_check is None:

        raise ValueError(
            "blood_banks table does not exist in database.db"
        )


    # ========================================================
    # SHOW BEFORE VALUE FOR 284016
    # ========================================================

    before = connection.execute(
        """
        SELECT
            hospital_code,
            blood_bank_name,
            district,
            area,
            city,
            latitude,
            longitude
        FROM blood_banks
        WHERE hospital_code = ?
        """,
        ("284016",)
    ).fetchone()


    print(
        "\n=============================================="
    )

    print(
        "DATABASE SYNCHRONIZATION"
    )

    print(
        "=============================================="
    )


    print(
        "\nBEFORE synchronization:"
    )


    if before:

        print(
            dict(before)
        )

    else:

        print(
            "Hospital code 284016 "
            "was not found in current database."
        )


    # ========================================================
    # CREATE MASTER LOOKUP
    # ========================================================

    master_lookup = (
        master_df[
            [
                "hospital_code",
                "district",
                "area",
                "city",
            ]
        ]
        .set_index(
            "hospital_code"
        )
        .to_dict(
            orient="index"
        )
    )


    # ========================================================
    # CREATE COORDINATE LOOKUP
    # ========================================================

    coordinate_lookup = (
        coordinates_df[
            [
                "hospital_code",
                "latitude",
                "longitude",
            ]
        ]
        .set_index(
            "hospital_code"
        )
        .to_dict(
            orient="index"
        )
    )


    # ========================================================
    # GET CURRENT DATABASE BANKS
    # ========================================================

    current_rows = connection.execute(
        """
        SELECT hospital_code
        FROM blood_banks
        """
    ).fetchall()


    print(
        "\nCurrent database banks:",
        len(current_rows)
    )


    updated_location_count = 0

    updated_coordinate_count = 0

    missing_master_count = 0


    # ========================================================
    # UPDATE EACH CURRENT BLOOD BANK
    # ========================================================

    for row in current_rows:

        hospital_code = str(
            row["hospital_code"]
        ).strip()


        # ----------------------------------------------------
        # MASTER LOCATION
        # ----------------------------------------------------

        master_location = (
            master_lookup.get(
                hospital_code
            )
        )


        if master_location is not None:

            connection.execute(
                """
                UPDATE blood_banks
                SET
                    district = ?,
                    area = ?,
                    city = ?
                WHERE hospital_code = ?
                """,
                (
                    master_location.get(
                        "district"
                    ),
                    master_location.get(
                        "area"
                    ),
                    master_location.get(
                        "city"
                    ),
                    hospital_code,
                )
            )

            updated_location_count += 1

        else:

            missing_master_count += 1


        # ----------------------------------------------------
        # COORDINATES
        # ----------------------------------------------------

        coordinate = (
            coordinate_lookup.get(
                hospital_code
            )
        )


        if coordinate is not None:

            latitude = coordinate.get(
                "latitude"
            )

            longitude = coordinate.get(
                "longitude"
            )


            # Do not overwrite with invalid NaN.
            if (
                pd.notna(latitude)
                and pd.notna(longitude)
            ):

                connection.execute(
                    """
                    UPDATE blood_banks
                    SET
                        latitude = ?,
                        longitude = ?
                    WHERE hospital_code = ?
                    """,
                    (
                        float(latitude),
                        float(longitude),
                        hospital_code,
                    )
                )

                updated_coordinate_count += 1


    # ========================================================
    # COMMIT CHANGES
    # ========================================================

    connection.commit()


    # ========================================================
    # SHOW AFTER VALUE
    # ========================================================

    after = connection.execute(
        """
        SELECT
            hospital_code,
            blood_bank_name,
            district,
            area,
            city,
            latitude,
            longitude
        FROM blood_banks
        WHERE hospital_code = ?
        """,
        ("284016",)
    ).fetchone()


    print(
        "\nAFTER synchronization:"
    )


    if after:

        print(
            dict(after)
        )

    else:

        print(
            "Hospital code 284016 "
            "was not found."
        )


    # ========================================================
    # FINAL CHECK
    # ========================================================

    print(
        "\n=============================================="
    )

    print(
        "SYNC COMPLETE"
    )

    print(
        "=============================================="
    )

    print(
        "Location records synchronized:",
        updated_location_count
    )

    print(
        "Coordinate records synchronized:",
        updated_coordinate_count
    )

    print(
        "Banks missing from master:",
        missing_master_count
    )


    # ========================================================
    # VERIFY PAREL → VILE PARLE
    # ========================================================

    verification = connection.execute(
        """
        SELECT
            hospital_code,
            blood_bank_name,
            district,
            area,
            city
        FROM blood_banks
        WHERE hospital_code = ?
        """,
        ("284016",)
    ).fetchone()


    if verification:

        final_area = verification["area"]


        if (
            str(final_area).strip().lower()
            == "vile parle"
        ):

            print(
                "\nSUCCESS:"
            )

            print(
                "284016 area is now:",
                final_area
            )

        else:

            print(
                "\nWARNING:"
            )

            print(
                "284016 area is:",
                final_area
            )


finally:

    connection.close()


print(
    "\nDatabase synchronization finished."
)