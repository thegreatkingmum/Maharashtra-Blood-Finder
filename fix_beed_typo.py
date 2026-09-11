# ============================================================
# FIX DISTRICT TYPO: BID -> BEED
# Maharashtra Blood Finder
#
# Purpose:
#   - Correct the district typo "Bid" to "Beed"
#   - Update SQLite database district fields
#   - Update website output CSV district fields
#   - Update matching Excel district fields when present
#
# This script does NOT change stock values, calculations,
# models, research logic, or methodology.
# ============================================================

from pathlib import Path
import sqlite3
import pandas as pd


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

DATABASE = BASE_DIR / "database.db"

OUTPUT_DIR = BASE_DIR / "data" / "outputs"


# ============================================================
# CONSTANTS
# ============================================================

OLD_DISTRICT = "Bid"
NEW_DISTRICT = "Beed"


# ============================================================
# HELPERS
# ============================================================

def normalize_text(value):
    if value is None:
        return ""

    return str(value).strip().lower()


def is_district_column(column_name):
    """
    Detect columns that represent district information.

    Examples:
        district
        donor_district
        recipient_district
        origin_district
        alternative_district
    """

    name = normalize_text(column_name)

    return (
        name == "district"
        or name.endswith("_district")
        or "district" in name
    )


# ============================================================
# FIX SQLITE DATABASE
# ============================================================

def fix_database():
    if not DATABASE.exists():
        print("WARNING: database.db not found.")
        return

    print()
    print("=" * 60)
    print("FIXING SQLITE DATABASE")
    print("=" * 60)

    conn = sqlite3.connect(DATABASE)

    try:
        cursor = conn.cursor()

        tables = cursor.execute(
            """
            SELECT name
            FROM sqlite_master
            WHERE type = 'table'
              AND name NOT LIKE 'sqlite_%'
            ORDER BY name
            """
        ).fetchall()

        total_updates = 0

        for table_row in tables:

            table_name = table_row[0]

            columns = cursor.execute(
                f'PRAGMA table_info("{table_name}")'
            ).fetchall()

            district_columns = [
                column[1]
                for column in columns
                if is_district_column(column[1])
            ]

            if not district_columns:
                continue

            print()
            print(f"Table: {table_name}")

            for column_name in district_columns:

                before = cursor.execute(
                    f'''
                    SELECT COUNT(*)
                    FROM "{table_name}"
                    WHERE LOWER(TRIM("{column_name}")) = ?
                    ''',
                    (OLD_DISTRICT.lower(),)
                ).fetchone()[0]

                if before == 0:
                    continue

                cursor.execute(
                    f'''
                    UPDATE "{table_name}"
                    SET "{column_name}" = ?
                    WHERE LOWER(TRIM("{column_name}")) = ?
                    ''',
                    (
                        NEW_DISTRICT,
                        OLD_DISTRICT.lower()
                    )
                )

                changed = cursor.rowcount

                total_updates += changed

                print(
                    f"  {column_name}: "
                    f"{changed} rows changed"
                )

        conn.commit()

        print()
        print(
            f"SQLite updates completed: "
            f"{total_updates}"
        )

    finally:
        conn.close()


# ============================================================
# FIX WEBSITE CSV OUTPUTS
# ============================================================

def fix_csv_outputs():
    if not OUTPUT_DIR.exists():
        print()
        print("WARNING: data/outputs directory not found.")
        return

    print()
    print("=" * 60)
    print("FIXING WEBSITE OUTPUT CSV FILES")
    print("=" * 60)

    csv_files = list(
        OUTPUT_DIR.rglob("*.csv")
    )

    changed_files = 0
    changed_rows = 0

    for csv_path in csv_files:

        try:
            df = pd.read_csv(
                csv_path
            )

        except Exception as exc:
            print(
                f"SKIPPED: {csv_path.name} "
                f"({exc})"
            )
            continue

        district_columns = [
            column
            for column in df.columns
            if is_district_column(column)
        ]

        if not district_columns:
            continue

        file_changed = False

        for column in district_columns:

            mask = (
                df[column]
                .fillna("")
                .astype(str)
                .str.strip()
                .str.lower()
                == OLD_DISTRICT.lower()
            )

            count = int(mask.sum())

            if count == 0:
                continue

            df.loc[
                mask,
                column
            ] = NEW_DISTRICT

            changed_rows += count
            file_changed = True

            print(
                f"{csv_path.name} | "
                f"{column}: {count} rows"
            )

        if file_changed:

            df.to_csv(
                csv_path,
                index=False
            )

            changed_files += 1

    print()
    print(
        f"CSV files changed: {changed_files}"
    )

    print(
        f"CSV district values corrected: "
        f"{changed_rows}"
    )


# ============================================================
# FIX EXCEL FILES IN PROJECT
# ============================================================

def fix_excel_files():
    """
    Correct exact district values in Excel files that are
    present in the project directory.

    Only columns containing 'district' are modified.
    """

    print()
    print("=" * 60)
    print("FIXING EXCEL DISTRICT VALUES")
    print("=" * 60)

    excel_files = []

    for pattern in (
        "*.xlsx",
        "*.xls"
    ):
        excel_files.extend(
            BASE_DIR.rglob(pattern)
        )

    # Avoid virtual environments and git internals.
    excel_files = [
        path
        for path in excel_files
        if ".git" not in path.parts
        and ".venv" not in path.parts
        and "venv" not in path.parts
        and "__pycache__" not in path.parts
    ]

    if not excel_files:
        print(
            "No Excel files found in project."
        )
        return

    changed_files = 0
    changed_rows = 0

    for excel_path in excel_files:

        try:
            workbook = pd.ExcelFile(
                excel_path
            )

        except Exception as exc:
            print(
                f"SKIPPED: {excel_path.name} "
                f"({exc})"
            )
            continue

        sheets_to_write = {}
        file_changed = False

        for sheet_name in workbook.sheet_names:

            try:
                df = pd.read_excel(
                    excel_path,
                    sheet_name=sheet_name
                )

            except Exception:
                continue

            district_columns = [
                column
                for column in df.columns
                if is_district_column(column)
            ]

            if not district_columns:
                sheets_to_write[sheet_name] = df
                continue

            sheet_changed = False

            for column in district_columns:

                mask = (
                    df[column]
                    .fillna("")
                    .astype(str)
                    .str.strip()
                    .str.lower()
                    == OLD_DISTRICT.lower()
                )

                count = int(mask.sum())

                if count == 0:
                    continue

                df.loc[
                    mask,
                    column
                ] = NEW_DISTRICT

                changed_rows += count
                sheet_changed = True
                file_changed = True

                print(
                    f"{excel_path.name} | "
                    f"{sheet_name} | "
                    f"{column}: {count} rows"
                )

            sheets_to_write[sheet_name] = df

        if not file_changed:
            continue

        # Rewrite workbook.
        with pd.ExcelWriter(
            excel_path,
            engine="openpyxl"
        ) as writer:

            for sheet_name, df in sheets_to_write.items():

                df.to_excel(
                    writer,
                    sheet_name=sheet_name,
                    index=False
                )

        changed_files += 1

    print()
    print(
        f"Excel files changed: {changed_files}"
    )

    print(
        f"Excel district values corrected: "
        f"{changed_rows}"
    )


# ============================================================
# VERIFY DISTRICT COUNT
# ============================================================

def verify_database():
    if not DATABASE.exists():
        return

    print()
    print("=" * 60)
    print("VERIFYING DISTRICT COUNT")
    print("=" * 60)

    conn = sqlite3.connect(
        DATABASE
    )

    try:
        tables = conn.execute(
            """
            SELECT name
            FROM sqlite_master
            WHERE type = 'table'
              AND name NOT LIKE 'sqlite_%'
            ORDER BY name
            """
        ).fetchall()

        if not tables:
            print(
                "No database tables found."
            )
            return

        # Prefer the main blood_banks table.
        target_table = None

        for row in tables:
            if row[0] == "blood_banks":
                target_table = row[0]
                break

        if target_table is None:
            print(
                "blood_banks table not found."
            )
            return

        columns = [
            row[1]
            for row in conn.execute(
                f'PRAGMA table_info("{target_table}")'
            ).fetchall()
        ]

        district_column = None

        for column in columns:

            if normalize_text(column) == "district":
                district_column = column
                break

        if district_column is None:
            print(
                "District column not found."
            )
            return

        districts = conn.execute(
            f'''
            SELECT DISTINCT TRIM("{district_column}")
            FROM "{target_table}"
            WHERE "{district_column}" IS NOT NULL
              AND TRIM("{district_column}") != ''
            ORDER BY TRIM("{district_column}")
            '''
        ).fetchall()

        district_names = [
            row[0]
            for row in districts
        ]

        print()
        print(
            "District count:",
            len(district_names)
        )

        print()
        print("Districts:")

        for district in district_names:
            print(
                f"  {district}"
            )

        print()

        if "Bid" in district_names:
            print(
                "ERROR: Bid still exists."
            )
        else:
            print(
                "SUCCESS: Bid has been "
                "fully replaced by Beed."
            )

        if len(district_names) == 36:
            print(
                "SUCCESS: Maharashtra has "
                "36 districts in the website data."
            )
        else:
            print(
                "WARNING: district count is",
                len(district_names),
                "instead of 36."
            )

    finally:
        conn.close()


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 60)
    print("MAHARASHTRA BLOOD FINDER")
    print("DISTRICT TYPO CORRECTION")
    print("=" * 60)

    print()
    print(
        f'Changing "{OLD_DISTRICT}" '
        f'-> "{NEW_DISTRICT}"'
    )

    fix_database()

    fix_csv_outputs()

    fix_excel_files()

    verify_database()

    print()
    print("=" * 60)
    print("DISTRICT TYPO FIX COMPLETE")
    print("=" * 60)
    print()
    print(
        "No stock calculations, ML models, "
        "research calculations or methodology "
        "were changed."
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()