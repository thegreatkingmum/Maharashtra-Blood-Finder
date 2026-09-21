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
#
# Additional permanent location normalization:
#   Dadar West -> Dadar
# ============================================================

from pathlib import Path
import sqlite3

import pandas as pd


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(
    __file__
).resolve().parent


DATABASE_FILE = (
    BASE_DIR
    / "database.db"
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
# PERMANENT LOCATION NORMALIZATION
# ============================================================

# These rules are applied AFTER reading the master CSV and
# BEFORE updating the live database.
#
# This protects the website from legacy/source-data spelling
# variations returning later.

AREA_NORMALIZATION = {

    "dadar west":
        "Dadar",

}


CITY_NORMALIZATION = {

    "panvel":
        "Panvel",

    "panvel ":
        "Panvel",

    "panvel municipal corporation":
        "Panvel",

    "panvel":
        "Panvel",

    "mira road":
        "Mira-Bhayandar",

    "Mira Road":
        "Mira-Bhayandar",

}


DISTRICT_NORMALIZATION = {

    "bid":
        "Beed",

    "Bid":
        "Beed",

    "raigarh":
        "Raigad",

    "Raigarh":
        "Raigad",

}


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
    for column
    in required_master_columns
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

master_df[
    "hospital_code"
] = (

    master_df[
        "hospital_code"
    ]
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
# NORMALIZE MASTER LOCATION VALUES
# ============================================================

def normalize_area(
    value
):

    if pd.isna(
        value
    ):

        return value

    cleaned = str(
        value
    ).strip()

    key = cleaned.lower()

    return AREA_NORMALIZATION.get(
        key,
        cleaned
    )


def normalize_city(
    value
):

    if pd.isna(
        value
    ):

        return value

    cleaned = str(
        value
    ).strip()

    key = cleaned.lower()

    return CITY_NORMALIZATION.get(
        key,
        cleaned
    )


def normalize_district(
    value
):

    if pd.isna(
        value
    ):

        return value

    cleaned = str(
        value
    ).strip()

    key = cleaned.lower()

    return DISTRICT_NORMALIZATION.get(
        key,
        cleaned
    )


master_df[
    "district"
] = (

    master_df[
        "district"
    ]
    .apply(
        normalize_district
    )

)


master_df[
    "area"
] = (

    master_df[
        "area"
    ]
    .apply(
        normalize_area
    )

)


master_df[
    "city"
] = (

    master_df[
        "city"
    ]
    .apply(
        normalize_city
    )

)


# ============================================================
# SHOW IMPORTANT MASTER CHECK
# ============================================================

master_check = master_df[
    master_df[
        "hospital_code"
    ]
    == "18023"
]


print()
print(
    "=============================================="
)

print(
    "MASTER LOCATION CHECK — 18023"
)

print(
    "=============================================="
)


if not master_check.empty:

    print(
        master_check[
            [
                "hospital_code",
                "district",
                "area",
                "city"
            ]
        ].to_string(
            index=False
        )
    )

else:

    print(
        "WARNING: hospital code 18023 "
        "not found in master."
    )


# ============================================================
# LOAD COORDINATES
# ============================================================

coordinates_df = pd.read_csv(
    COORDINATE_FILE
)


coordinates_df.columns = [

    str(column).strip()

    for column
    in coordinates_df.columns

]


required_coordinate_columns = [

    "hospital_code",
    "latitude",
    "longitude",

]


missing_coordinate_columns = [

    column

    for column
    in required_coordinate_columns

    if column
    not in coordinates_df.columns

]


if missing_coordinate_columns:

    raise ValueError(
        "Coordinates CSV is missing columns: "
        f"{missing_coordinate_columns}"
    )


coordinates_df[
    "hospital_code"
] = (

    coordinates_df[
        "hospital_code"
    ]
    .astype(str)
    .str.strip()

)


coordinates_df[
    "latitude"
] = pd.to_numeric(

    coordinates_df[
        "latitude"
    ],

    errors="coerce",

)


coordinates_df[
    "longitude"
] = pd.to_numeric(

    coordinates_df[
        "longitude"
    ],

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
            "blood_banks table does not exist "
            "in database.db"
        )


    # ========================================================
    # SHOW BEFORE VALUE FOR 18023
    # ========================================================

    before = connection.execute(
        """
        SELECT
            hospital_code,
            blood_bank_name,
            district,
            area,
            city,
            "A+",
            "O+",
            snapshot_date,
            source_file
        FROM blood_banks
        WHERE hospital_code = ?
        """,
        (
            "18023",
        )
    ).fetchone()


    print()
    print(
        "=============================================="
    )

    print(
        "DATABASE LOCATION SYNCHRONIZATION"
    )

    print(
        "=============================================="
    )


    print()
    print(
        "BEFORE synchronization — 18023:"
    )


    if before:

        print(
            dict(
                before
            )
        )

    else:

        print(
            "Hospital code 18023 "
            "was not found."
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


    print()
    print(
        "Current database banks:",
        len(
            current_rows
        )
    )


    updated_location_count = 0

    updated_coordinate_count = 0

    missing_master_count = 0


    # ========================================================
    # UPDATE EACH CURRENT BLOOD BANK
    # ========================================================

    for row in current_rows:

        hospital_code = str(
            row[
                "hospital_code"
            ]
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

            district = normalize_district(
                master_location.get(
                    "district"
                )
            )

            area = normalize_area(
                master_location.get(
                    "area"
                )
            )

            city = normalize_city(
                master_location.get(
                    "city"
                )
            )


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
                    district,
                    area,
                    city,
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
                pd.notna(
                    latitude
                )
                and
                pd.notna(
                    longitude
                )
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
                        float(
                            latitude
                        ),
                        float(
                            longitude
                        ),
                        hospital_code,
                    )
                )


                updated_coordinate_count += 1


    # ========================================================
    # EXPLICIT LEGACY ALIAS SAFETY
    #
    # This guarantees that Dadar West can never remain in the
    # live blood_banks table even if an old master copy or
    # legacy location value reappears.
    # ========================================================

    legacy_area_updates = connection.execute(
        """
        UPDATE blood_banks
        SET area = 'Dadar'
        WHERE LOWER(TRIM(area))
              = 'dadar west'
        """
    )


    legacy_area_count = (
        legacy_area_updates.rowcount
    )


    # ========================================================
    # NORMALIZE OTHER KNOWN LEGACY VALUES
    # ========================================================

    district_alias_updates = connection.execute(
        """
        UPDATE blood_banks
        SET district = 'Beed'
        WHERE LOWER(TRIM(district))
              = 'bid'
        """
    )


    district_alias_count = (
        district_alias_updates.rowcount
    )


    raigarh_updates = connection.execute(
        """
        UPDATE blood_banks
        SET district = 'Raigad'
        WHERE LOWER(TRIM(district))
              = 'raigarh'
        """
    )


    raigarh_count = (
        raigarh_updates.rowcount
    )


    # ========================================================
    # COMMIT
    # ========================================================

    connection.commit()


    # ========================================================
    # SHOW AFTER VALUE FOR 18023
    # ========================================================

    after = connection.execute(
        """
        SELECT
            hospital_code,
            blood_bank_name,
            district,
            area,
            city,
            "A+",
            "O+",
            snapshot_date,
            source_file
        FROM blood_banks
        WHERE hospital_code = ?
        """,
        (
            "18023",
        )
    ).fetchone()


    print()
    print(
        "AFTER synchronization — 18023:"
    )


    if after:

        print(
            dict(
                after
            )
        )

    else:

        print(
            "Hospital code 18023 "
            "was not found."
        )


    # ========================================================
    # FINAL COUNTS
    # ========================================================

    print()
    print(
        "=============================================="
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


    print(
        "Dadar West -> Dadar corrections:",
        legacy_area_count
    )


    print(
        "Bid -> Beed corrections:",
        district_alias_count
    )


    print(
        "Raigarh -> Raigad corrections:",
        raigarh_count
    )


    # ========================================================
    # VERIFY 18023
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
        (
            "18023",
        )
    ).fetchone()


    print()
    print(
        "FINAL 18023 VERIFICATION:"
    )


    if verification:

        print(
            dict(
                verification
            )
        )


        final_area = (
            verification[
                "area"
            ]
        )


        if (
            str(
                final_area
            )
            .strip()
            .lower()
            ==
            "dadar"
        ):

            print()
            print(
                "SUCCESS:"
            )

            print(
                "18023 area is permanently normalized to Dadar."
            )

        else:

            print()
            print(
                "WARNING:"
            )

            print(
                "18023 area is still:",
                final_area
            )


    # ========================================================
    # CHECK NO DADAR WEST REMAINS
    # ========================================================

    remaining_legacy = connection.execute(
        """
        SELECT COUNT(*)
        FROM blood_banks
        WHERE LOWER(TRIM(area))
              = 'dadar west'
        """
    ).fetchone()[0]


    print()
    print(
        "Remaining Dadar West rows:",
        remaining_legacy
    )


    if remaining_legacy == 0:

        print(
            "SUCCESS: No Dadar West values remain "
            "in the live blood_banks table."
        )

    else:

        print(
            "WARNING: Dadar West values still remain."
        )


finally:

    connection.close()


print()
print(
    "Database synchronization finished."
)