# ============================================================
# MAHARASHTRA BLOOD FINDER
# FULL DAILY UPDATE PIPELINE
#
# Updates:
#   1  Maharashtra availability
#   2  BBRI
#   3  DBARI
#   4  District × blood-group hotspots
#   5  Geographic vulnerability
#   6  Stockout prediction
#   7  XAI / feature importance
#   8  Blood-group comparison
#   9  Network resilience
#  10  Redistribution
#  11  Alternative blood-bank recommendations
# ============================================================

import json
import re
import sqlite3
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


# ============================================================
# OPTIONAL LIBRARIES
# ============================================================

try:
    import joblib
except ImportError:
    joblib = None

try:
    import networkx as nx
except ImportError:
    nx = None

try:
    from sklearn.ensemble import (
        HistGradientBoostingClassifier,
        RandomForestClassifier
    )
    from sklearn.inspection import permutation_importance
    from sklearn.metrics import (
        accuracy_score,
        precision_score,
        recall_score,
        f1_score,
        roc_auc_score
    )
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

SNAPSHOT_DIR = BASE_DIR / "snapshots"
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = DATA_DIR / "outputs"
MODEL_DIR = BASE_DIR / "models"

MASTER_FILE = DATA_DIR / "master_blood_banks.csv"
COORDINATE_FILE = DATA_DIR / "coordinates.csv"

DATABASE = BASE_DIR / "database.db"


# ============================================================
# CONSTANTS
# ============================================================

BLOOD_GROUPS = [
    "A+",
    "A-",
    "B+",
    "B-",
    "O+",
    "O-",
    "AB+",
    "AB-"
]

LOW_STOCK_MAX = 5
SAFETY_STOCK = 5

NETWORK_RADIUS_KM = 50
REDISTRIBUTION_RADIUS_KM = 150

MIN_HOTSPOT_OBSERVATIONS = 30

RANDOM_STATE = 42

# A Maharashtra-wide e-RaktKosh snapshot should contain hundreds of banks.
# Fail closed instead of rebuilding research outputs from a truncated PDF parse.
MIN_EXPECTED_SOURCE_ROWS = 100


# ============================================================
# KNOWN COORDINATE ANOMALIES
# ============================================================

BAD_COORDINATE_CODES = {
    "18135",
    "18126",
    "18061",
    "284117",
    "282735",
    "280143",
    "282683",
    "283233",
    "283290",
    "283289",
    "283287",
    "283264",
    "283142",
    "283261",
    "283198",
    "283298",
    "282741"
}


# ============================================================
# MODEL FEATURES
# ============================================================

MODEL_FEATURES = [
    "stock",
    "previous_stock",
    "stock_change",
    "rolling_mean_3",
    "rolling_std_3",
    "previous_stockout",
    "previous_low_stock",
    "stock_vs_rolling_mean",
    "trend",
    "recovered_from_previous_stockout"
]


# ============================================================
# PRINT HELPERS
# ============================================================

def banner(title):

    print()
    print("=" * 64)
    print(title)
    print("=" * 64)


def step(title):

    print()
    print("-" * 64)
    print(title)
    print("-" * 64)


# ============================================================
# GENERAL HELPERS
# ============================================================

def normalize_code(value):

    if pd.isna(value):
        return None

    value = str(value).strip()

    if value.endswith(".0"):
        value = value[:-2]

    return value


def safe_numeric(series):

    return pd.to_numeric(
        series,
        errors="coerce"
    )


def percentile_rank(series):

    values = pd.to_numeric(
        series,
        errors="coerce"
    )

    if values.dropna().empty:

        return pd.Series(
            0,
            index=values.index
        )

    return (
        values
        .rank(
            method="average",
            pct=True
        )
        * 100
    ).fillna(0)


def clean_for_csv(df):

    return (
        df.copy()
        .replace(
            [
                np.inf,
                -np.inf
            ],
            np.nan
        )
    )


def haversine_km(
    lat1,
    lon1,
    lat2,
    lon2
):

    lat1 = np.radians(lat1)
    lon1 = np.radians(lon1)

    lat2 = np.radians(lat2)
    lon2 = np.radians(lon2)

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = (
        np.sin(dlat / 2) ** 2
        +
        np.cos(lat1)
        *
        np.cos(lat2)
        *
        np.sin(dlon / 2) ** 2
    )

    return (
        6371.0088
        *
        2
        *
        np.arcsin(
            np.sqrt(a)
        )
    )


# ============================================================
# DIRECTORIES
# ============================================================

def ensure_directories():

    folders = [

        OUTPUT_DIR,

        MODEL_DIR,

        OUTPUT_DIR / "objective1",
        OUTPUT_DIR / "objective2",
        OUTPUT_DIR / "objective3",
        OUTPUT_DIR / "objective4",
        OUTPUT_DIR / "objective5",
        OUTPUT_DIR / "objective6",
        OUTPUT_DIR / "objective7",
        OUTPUT_DIR / "objective8",
        OUTPUT_DIR / "objective9",
        OUTPUT_DIR / "objective10",
        OUTPUT_DIR / "objective11"

    ]

    for folder in folders:

        folder.mkdir(
            parents=True,
            exist_ok=True
        )


# ============================================================
# MASTER
# ============================================================

def load_master():

    if not MASTER_FILE.exists():

        raise FileNotFoundError(
            f"Master file not found: {MASTER_FILE}"
        )

    master = pd.read_csv(
        MASTER_FILE
    )

    master.columns = [
        str(c).strip()
        for c in master.columns
    ]

    rename_map = {}

    for column in master.columns:

        lower = (
            str(column)
            .strip()
            .lower()
        )

        if lower in {
            "hospital code",
            "hospitalcode"
        }:

            rename_map[column] = (
                "hospital_code"
            )

        elif lower in {
            "blood bank name",
            "bank name"
        }:

            rename_map[column] = (
                "blood_bank_name"
            )

    master = master.rename(
        columns=rename_map
    )

    required = [
        "hospital_code",
        "blood_bank_name",
        "district",
        "city",
        "area"
    ]

    missing = [
        column
        for column in required
        if column not in master.columns
    ]

    if missing:

        raise ValueError(
            "Master file missing columns: "
            + ", ".join(missing)
        )

    master[
        "hospital_code"
    ] = (
        master[
            "hospital_code"
        ]
        .map(
            normalize_code
        )
    )

    for column in [
        "district",
        "city",
        "area"
    ]:

        master[column] = (
            master[column]
            .fillna("")
            .astype(str)
            .str.strip()
        )

    return master


# ============================================================
# COORDINATES
# ============================================================

def load_coordinates():

    if not COORDINATE_FILE.exists():

        print(
            "WARNING: Coordinate file not found."
        )

        return pd.DataFrame(
            columns=[
                "hospital_code",
                "latitude",
                "longitude"
            ]
        )

    coordinates = pd.read_csv(
        COORDINATE_FILE
    )

    coordinates.columns = [
        str(c).strip()
        for c in coordinates.columns
    ]

    rename_map = {}

    for column in coordinates.columns:

        lower = (
            str(column)
            .strip()
            .lower()
        )

        if lower in {
            "hospital code",
            "hospitalcode"
        }:

            rename_map[column] = (
                "hospital_code"
            )

        elif lower == "lat":

            rename_map[column] = (
                "latitude"
            )

        elif lower in {
            "lon",
            "lng"
        }:

            rename_map[column] = (
                "longitude"
            )

    coordinates = coordinates.rename(
        columns=rename_map
    )

    for column in [
        "hospital_code",
        "latitude",
        "longitude"
    ]:

        if column not in coordinates.columns:

            coordinates[column] = np.nan

    coordinates[
        "hospital_code"
    ] = (
        coordinates[
            "hospital_code"
        ]
        .map(
            normalize_code
        )
    )

    coordinates[
        "latitude"
    ] = safe_numeric(
        coordinates[
            "latitude"
        ]
    )

    coordinates[
        "longitude"
    ] = safe_numeric(
        coordinates[
            "longitude"
        ]
    )

    return (
        coordinates[
            [
                "hospital_code",
                "latitude",
                "longitude"
            ]
        ]
        .drop_duplicates(
            "hospital_code"
        )
    )


# ============================================================
# SNAPSHOT TEXT
# ============================================================

def read_snapshot_text(path):

    suffix = path.suffix.lower()

    if suffix == ".json":

        return path.read_text(
            encoding="utf-8",
            errors="ignore"
        )

    if suffix in {
        ".txt",
        ".text"
    }:

        return path.read_text(
            encoding="utf-8",
            errors="ignore"
        )

    if suffix == ".pdf":

        try:

            import pypdf

            reader = pypdf.PdfReader(
                str(path)
            )

            text_parts = []

            for page in reader.pages:

                try:

                    text_parts.append(
                        page.extract_text()
                        or ""
                    )

                except Exception:
                    continue

            text = "\n".join(
                text_parts
            )

            if text.strip():

                return text

        except Exception as exc:

            print(
                "pypdf extraction warning:",
                exc
            )

        # PDF fallback
        try:

            import fitz

            document = fitz.open(
                str(path)
            )

            text_parts = []

            for page in document:

                text_parts.append(
                    page.get_text()
                )

            document.close()

            text = "\n".join(
                text_parts
            )

            if text.strip():

                return text

        except Exception as exc:

            print(
                "PyMuPDF extraction warning:",
                exc
            )

        raise RuntimeError(
            f"Could not extract text from {path.name}"
        )

    raise ValueError(
        f"Unsupported snapshot type: {path.name}"
    )


# ============================================================
# ROBUST JSON RECORD EXTRACTION
#
# IMPORTANT:
# We no longer require the ENTIRE PDF-extracted JSON
# array to decode successfully.
#
# We identify each top-level record using hospitalCode
# and balanced JSON braces.
# ============================================================

def extract_json_object_at(
    text,
    start
):

    depth = 0

    in_string = False

    escaped = False

    for index in range(
        start,
        len(text)
    ):

        char = text[index]

        if in_string:

            if escaped:

                escaped = False

            elif char == "\\":
                escaped = True

            elif char == '"':
                in_string = False

            continue

        if char == '"':

            in_string = True

            continue

        if char == "{":

            depth += 1

        elif char == "}":

            depth -= 1

            if depth == 0:

                return text[
                    start:index + 1
                ]

    return None


def find_record_start(text, marker_position):
    """
    Find the opening brace of the top-level blood-bank object that
    contains the hospitalCode marker.

    PDF text extraction can insert whitespace/newlines between JSON
    tokens. Therefore we first look for the characteristic pattern:
        { ... "hospitalCode":
    and only then fall back to a balanced-brace search.
    """

    window_start = max(0, marker_position - 1000)
    prefix = text[window_start:marker_position + 1]

    # Best case: the record starts directly with { then hospitalCode.
    direct_matches = list(
        re.finditer(
            r'\{\s*"hospitalCode"\s*:',
            prefix,
            flags=re.IGNORECASE
        )
    )

    if direct_matches:
        return window_start + direct_matches[-1].start()

    # Fallback: search backwards for candidate object starts and
    # validate them by actually decoding a balanced JSON object.
    candidate_positions = [
        m.start()
        for m in re.finditer(
            r"\{",
            text[window_start:marker_position + 1]
        )
    ]

    for relative_start in reversed(candidate_positions):
        absolute_start = window_start + relative_start

        candidate = extract_json_object_at(
            text,
            absolute_start
        )

        if candidate is None:
            continue

        try:
            obj = json.loads(candidate)
        except Exception:
            continue

        if (
            isinstance(obj, dict)
            and normalize_code(obj.get("hospitalCode"))
        ):
            return absolute_start

    return None


def extract_source_records(text):
    """
    Extract the complete list of e-RaktKosh blood-bank records from
    PDF-extracted JSON text.

    PDF extraction may corrupt the outer JSON array because page
    boundaries introduce extra text. We therefore use two strategies:

    1. Decode the complete JSON array when possible.
    2. Otherwise locate every top-level { "hospitalCode": ... } object
       independently and decode it with balanced braces.

    This is intentionally PDF-only in the main pipeline.
    """

    if not text:
        return []

    # --------------------------------------------------------
    # Method 1 — complete JSON array.
    # --------------------------------------------------------
    hospital_match = re.search(
        r'"hospitalCode"\s*:',
        text,
        flags=re.IGNORECASE
    )

    if hospital_match:
        # Prefer the nearest '[' before the first hospitalCode.
        array_start = text.rfind(
            "[",
            0,
            hospital_match.start()
        )

        if array_start >= 0:
            try:
                decoder = json.JSONDecoder()

                parsed, _ = decoder.raw_decode(
                    text[array_start:]
                )

                if isinstance(parsed, list):
                    valid = [
                        item
                        for item in parsed
                        if (
                            isinstance(item, dict)
                            and normalize_code(
                                item.get("hospitalCode")
                            )
                        )
                    ]

                    if valid:
                        # pypdf can return only the first page-level JSON
                        # array and then stop at a page boundary. In that
                        # case raw_decode() may falsely look successful.
                        # Accept the complete-array result only when the
                        # number of decoded hospital records matches the
                        # number of hospitalCode markers in the text.
                        marker_count = len(
                            re.findall(
                                r'"hospitalCode"\s*:',
                                text,
                                flags=re.IGNORECASE
                            )
                        )

                        if len(valid) == marker_count:
                            return valid

            except Exception:
                pass

    # --------------------------------------------------------
    # Method 2 — direct top-level record starts.
    #
    # This is the critical fix for PDFs where page extraction
    # breaks the outer array.
    # --------------------------------------------------------
    start_matches = list(
        re.finditer(
            r'\{\s*"hospitalCode"\s*:',
            text,
            flags=re.IGNORECASE
        )
    )

    records = []
    seen_codes = set()

    for match in start_matches:

        start = match.start()

        candidate = extract_json_object_at(
            text,
            start
        )

        if candidate is None:
            continue

        try:
            obj = json.loads(candidate)
        except Exception:
            continue

        if not isinstance(obj, dict):
            continue

        code = normalize_code(
            obj.get("hospitalCode")
        )

        if not code:
            continue

        if code in seen_codes:
            continue

        seen_codes.add(code)
        records.append(obj)

    # --------------------------------------------------------
    # Method 3 — final fallback using hospitalCode markers.
    # --------------------------------------------------------
    if not records:

        matches = list(
            re.finditer(
                r'"hospitalCode"\s*:',
                text,
                flags=re.IGNORECASE
            )
        )

        for match in matches:

            start = find_record_start(
                text,
                match.start()
            )

            if start is None:
                continue

            candidate = extract_json_object_at(
                text,
                start
            )

            if candidate is None:
                continue

            try:
                obj = json.loads(candidate)
            except Exception:
                continue

            if not isinstance(obj, dict):
                continue

            code = normalize_code(
                obj.get("hospitalCode")
            )

            if not code or code in seen_codes:
                continue

            seen_codes.add(code)
            records.append(obj)

    return records


# ============================================================
# SNAPSHOT DATE
# ============================================================

def determine_snapshot_date(
    records,
    file_path
):

    dates = []

    for record in records:

        value = record.get(
            "entrydate"
        )

        if not value:
            continue

        parsed = pd.to_datetime(
            value,
            errors="coerce"
        )

        if pd.notna(parsed):

            dates.append(
                parsed.date()
            )

    if dates:

        counts = (
            pd.Series(dates)
            .value_counts()
        )

        return pd.Timestamp(
            counts.index[0]
        ).date()

    # Filename fallback.
    match = re.search(
        r"(\d{1,2})[-_](\d{1,2})[-_](\d{2,4})",
        file_path.stem
    )

    if match:

        day = int(
            match.group(1)
        )

        month = int(
            match.group(2)
        )

        year = int(
            match.group(3)
        )

        if year < 100:
            year += 2000

        return pd.Timestamp(
            year=year,
            month=month,
            day=day
        ).date()

    raise ValueError(
        f"Could not determine date for {file_path.name}"
    )


# ============================================================
# STOCK
# ============================================================

GROUP_PATTERNS = {

    "A+":
        r"A\+Ve",

    "A-":
        r"A-Ve",

    "B+":
        r"B\+Ve",

    "B-":
        r"B-Ve",

    "O+":
        r"O\+Ve",

    "O-":
        r"O-Ve",

    "AB+":
        r"AB\+Ve",

    "AB-":
        r"AB-Ve"

}


def parse_stock_string(
    value
):

    result = {
        group: 0
        for group in BLOOD_GROUPS
    }

    if value is None:

        return result

    text = str(value)

    for group, pattern in GROUP_PATTERNS.items():

        match = re.search(
            rf"{pattern}\s*:\s*(-?\d+(?:\.\d+)?)",
            text,
            flags=re.IGNORECASE
        )

        if match:

            number = float(
                match.group(1)
            )

            result[group] = max(
                0,
                number
            )

    return result


def extract_record_stock(
    record
):

    stock = {
        group: 0
        for group in BLOOD_GROUPS
    }

    components = (
        record.get(
            "components"
        )
        or {}
    )

    packed_rbc = None

    if isinstance(
        components,
        dict
    ):

        for key, value in components.items():

            if (
                str(key)
                .strip()
                .lower()
                ==
                "packed red blood cells"
            ):

                packed_rbc = value
                break

    if isinstance(
        packed_rbc,
        dict
    ):

        stock.update(
            parse_stock_string(
                packed_rbc.get(
                    "available_WithQty"
                )
            )
        )

    if all(
        value == 0
        for value in stock.values()
    ):

        stock.update(
            parse_stock_string(
                record.get(
                    "available_WithQty"
                )
            )
        )

    return stock


# ============================================================
# PARSE SNAPSHOT
# ============================================================

def parse_snapshot(
    path,
    master,
    coordinates
):

    text = read_snapshot_text(
        path
    )

    records = extract_source_records(
        text
    )

    if not records:

        # Diagnostics — helps identify future source changes.
        print(
            "DEBUG: hospitalCode occurrences =",
            len(
                re.findall(
                    r'"hospitalCode"\s*:',
                    text,
                    flags=re.IGNORECASE
                )
            )
        )

        print(
            "DEBUG: text length =",
            len(text)
        )

        print(
            "DEBUG: first 500 characters:"
        )

        print(
            repr(
                text[
                    :500
                ]
            )
        )

        raise ValueError(
            "No JSON source records found."
        )

    snapshot_date = determine_snapshot_date(
        records,
        path
    )

    rows = []

    for record in records:

        code = normalize_code(
            record.get(
                "hospitalCode"
            )
        )

        if not code:
            continue

        stock = extract_record_stock(
            record
        )

        rows.append({

            "hospital_code":
                code,

            "blood_bank_name":
                record.get(
                    "hospitalname",
                    ""
                ),

            "address":
                record.get(
                    "hospitaladd",
                    ""
                ),

            "contact":
                record.get(
                    "hospitalcontact",
                    ""
                ),

            "hospital_type":
                record.get(
                    "hospitalType",
                    ""
                ),

            **stock,

            "snapshot_date":
                str(
                    snapshot_date
                ),

            "source_file":
                path.name

        })

    snapshot = pd.DataFrame(
        rows
    )

    if snapshot.empty:

        return snapshot

    snapshot = snapshot.merge(

        master[
            [
                "hospital_code",
                "district",
                "city",
                "area"
            ]
        ],

        on="hospital_code",

        how="left"

    )

    snapshot = snapshot.merge(

        coordinates,

        on="hospital_code",

        how="left"

    )

    for group in BLOOD_GROUPS:

        snapshot[group] = (
            safe_numeric(
                snapshot[group]
            )
            .fillna(0)
            .clip(lower=0)
        )

    return snapshot


# ============================================================
# DATABASE
# ============================================================

def initialize_database():

    conn = sqlite3.connect(
        DATABASE
    )

    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS stock_history (

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

            UNIQUE(
                snapshot_date,
                hospital_code
            )

        )
        """
    )

    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS blood_banks (

            hospital_code TEXT PRIMARY KEY,

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

            snapshot_date TEXT,

            source_file TEXT

        )
        """
    )

    # Old versions may already have update_log.
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS update_log (

            id INTEGER PRIMARY KEY AUTOINCREMENT,

            run_time TEXT,

            snapshot_date TEXT,

            source_file TEXT,

            source_rows INTEGER,

            valid_rows INTEGER,

            new_rows INTEGER,

            status TEXT,

            message TEXT

        )
        """
    )

    conn.commit()

    conn.close()

    ensure_database_schema()


def ensure_database_schema():
    """Migrate older database schemas used by previous pipeline versions."""

    conn = sqlite3.connect(DATABASE)

    table_requirements = {

        "blood_banks": {
            "source_file": "TEXT"
        },

        "stock_history": {
            "source_file": "TEXT"
        },

        "update_log": {
            "run_time": "TEXT",
            "snapshot_date": "TEXT",
            "source_file": "TEXT",
            "source_rows": "INTEGER",
            "valid_rows": "INTEGER",
            "new_rows": "INTEGER",
            "status": "TEXT",
            "message": "TEXT"
        }
    }

    for table, required_columns in table_requirements.items():

        existing_columns = {
            row[1]
            for row in conn.execute(
                f"PRAGMA table_info({table})"
            ).fetchall()
        }

        for column, sql_type in required_columns.items():

            if column not in existing_columns:

                print(
                    f"Migrating {table}: adding {column}"
                )

                conn.execute(
                    f"ALTER TABLE {table} ADD COLUMN {column} {sql_type}"
                )

    conn.commit()
    conn.close()


def ensure_update_log_schema():
    # Backward-compatible wrapper retained for older callers.
    ensure_database_schema()


def insert_snapshot(
    snapshot
):

    if snapshot.empty:

        return 0

    columns = [

        "snapshot_date",
        "hospital_code",
        "blood_bank_name",
        "district",
        "city",
        "area",
        "address",
        "contact",
        "hospital_type",

        "A+",
        "A-",
        "B+",
        "B-",
        "O+",
        "O-",
        "AB+",
        "AB-",

        "latitude",
        "longitude",
        "source_file"

    ]

    conn = sqlite3.connect(
        DATABASE
    )

    inserted = 0

    for _, row in snapshot.iterrows():

        values = []

        for column in columns:

            value = row.get(
                column
            )

            if pd.isna(value):

                value = None

            values.append(
                value
            )

        before = conn.total_changes

        conn.execute(
            """
            INSERT OR IGNORE INTO stock_history (

                snapshot_date,
                hospital_code,
                blood_bank_name,
                district,
                city,
                area,
                address,
                contact,
                hospital_type,

                "A+",
                "A-",
                "B+",
                "B-",
                "O+",
                "O-",
                "AB+",
                "AB-",

                latitude,
                longitude,
                source_file

            )
            VALUES (
                ?, ?, ?, ?, ?, ?, ?, ?, ?,
                ?, ?, ?, ?, ?, ?, ?, ?,
                ?, ?, ?
            )
            """,
            values
        )

        if conn.total_changes > before:

            inserted += 1

    conn.commit()
    conn.close()

    return inserted


def rebuild_current_table():

    conn = sqlite3.connect(
        DATABASE
    )

    latest_date = conn.execute(
        """
        SELECT MAX(snapshot_date)
        FROM stock_history
        """
    ).fetchone()[0]

    if not latest_date:

        conn.close()

        return pd.DataFrame()

    latest = pd.read_sql_query(

        """
        SELECT *
        FROM stock_history
        WHERE snapshot_date = ?
        """,

        conn,

        params=[
            latest_date
        ]

    )

    conn.execute(
        "DELETE FROM blood_banks"
    )

    columns = [

        "hospital_code",
        "blood_bank_name",
        "district",
        "city",
        "area",
        "address",
        "contact",
        "hospital_type",

        "A+",
        "A-",
        "B+",
        "B-",
        "O+",
        "O-",
        "AB+",
        "AB-",

        "latitude",
        "longitude",

        "snapshot_date",
        "source_file"

    ]

    placeholders = ",".join(
        ["?"] * len(columns)
    )

    insert_columns = ",".join(

        f'"{column}"'
        if column in BLOOD_GROUPS
        else column

        for column in columns

    )

    sql = f"""
        INSERT INTO blood_banks (
            {insert_columns}
        )
        VALUES ({placeholders})
    """

    for _, row in latest.iterrows():

        values = []

        for column in columns:

            value = row.get(
                column
            )

            if pd.isna(value):

                value = None

            values.append(
                value
            )

        conn.execute(
            sql,
            values
        )

    conn.commit()
    conn.close()

    return latest


def export_database_csvs():

    conn = sqlite3.connect(
        DATABASE
    )

    current = pd.read_sql_query(
        """
        SELECT *
        FROM blood_banks
        """,
        conn
    )

    history = pd.read_sql_query(
        """
        SELECT *
        FROM stock_history
        ORDER BY
            snapshot_date,
            hospital_code
        """,
        conn
    )

    conn.close()

    current.to_csv(
        OUTPUT_DIR /
        "current_blood_stock.csv",
        index=False
    )

    history.to_csv(
        OUTPUT_DIR /
        "blood_stock_history.csv",
        index=False
    )

    return current, history


# ============================================================
# LONG DATA
# ============================================================

def make_long_history(
    history
):

    parts = []

    for group in BLOOD_GROUPS:

        temp = history[
            [
                "snapshot_date",
                "hospital_code",
                "blood_bank_name",
                "district",
                "city",
                "area",
                group
            ]
        ].copy()

        temp["blood_group"] = group

        temp["stock"] = safe_numeric(
            temp[group]
        ).fillna(0)

        parts.append(
            temp.drop(
                columns=[group]
            )
        )

    result = pd.concat(
        parts,
        ignore_index=True
    )

    result[
        "snapshot_date"
    ] = pd.to_datetime(
        result[
            "snapshot_date"
        ],
        errors="coerce"
    )

    return (
        result
        .sort_values(
            [
                "hospital_code",
                "blood_group",
                "snapshot_date"
            ]
        )
        .reset_index(
            drop=True
        )
    )


# ============================================================
# OBJECTIVE 1
# ============================================================

def objective1(
    history
):

    step(
        "OBJECTIVE 1 — Maharashtra Availability"
    )

    long_df = make_long_history(
        history
    )

    summary = (
        long_df
        .groupby(
            "blood_group"
        )
        .agg(

            observations=("stock", "count"),

            average_stock=("stock", "mean"),

            median_stock=("stock", "median"),

            total_units=("stock", "sum"),

            stockout_units=(
                "stock",
                lambda x:
                (x == 0).sum()
            ),

            low_stock_units=(
                "stock",
                lambda x:
                x.between(
                    1,
                    LOW_STOCK_MAX
                ).sum()
            )

        )
        .reset_index()
    )

    summary[
        "stockout_percentage"
    ] = (
        summary[
            "stockout_units"
        ]
        /
        summary[
            "observations"
        ]
        * 100
    )

    summary[
        "low_stock_percentage"
    ] = (
        summary[
            "low_stock_units"
        ]
        /
        summary[
            "observations"
        ]
        * 100
    )

    summary[
        "availability_class"
    ] = np.select(

        [

            summary[
                "average_stock"
            ] == 0,

            summary[
                "average_stock"
            ] <= LOW_STOCK_MAX

        ],

        [

            "Stockout",

            "Low Stock"

        ],

        default="Available"

    )

    summary.to_csv(

        OUTPUT_DIR /
        "objective1" /
        "objective1_maharashtra_availability.csv",

        index=False

    )

    district = (
        long_df
        .groupby(
            "district"
        )
        .agg(

            observations=("stock", "count"),

            average_stock=("stock", "mean"),

            median_stock=("stock", "median"),

            total_units=("stock", "sum"),

            stockout_percentage=(
                "stock",
                lambda x:
                (x == 0).mean()
                * 100
            ),

            low_stock_percentage=(
                "stock",
                lambda x:
                x.between(
                    1,
                    LOW_STOCK_MAX
                ).mean()
                * 100
            )

        )
        .reset_index()
    )

    district.to_csv(

        OUTPUT_DIR /
        "objective1" /
        "objective1_district_availability.csv",

        index=False

    )

    district_group = (
        long_df
        .groupby(
            [
                "district",
                "blood_group"
            ]
        )
        .agg(

            observations=("stock", "count"),

            average_stock=("stock", "mean"),

            median_stock=("stock", "median"),

            total_units=("stock", "sum"),

            stockout_percentage=(
                "stock",
                lambda x:
                (x == 0).mean()
                * 100
            ),

            low_stock_percentage=(
                "stock",
                lambda x:
                x.between(
                    1,
                    LOW_STOCK_MAX
                ).mean()
                * 100
            )

        )
        .reset_index()
    )

    district_group.to_csv(

        OUTPUT_DIR /
        "objective1" /
        "objective1_district_bloodgroup.csv",

        index=False

    )

    print(
        "Objective 1 complete."
    )


# ============================================================
# RECOVERY
# ============================================================

def recovery_by_bank_group(
    long_df
):

    data = (
        long_df
        .sort_values(
            [
                "hospital_code",
                "blood_group",
                "snapshot_date"
            ]
        )
        .copy()
    )

    data[
        "previous_stock"
    ] = data.groupby(
        [
            "hospital_code",
            "blood_group"
        ]
    )[
        "stock"
    ].shift(1)

    events = data[
        data[
            "previous_stock"
        ] == 0
    ].copy()

    events[
        "recovered"
    ] = (
        events[
            "stock"
        ] > 0
    ).astype(int)

    result = (
        events
        .groupby(
            [
                "hospital_code",
                "blood_group"
            ]
        )
        .agg(

            stockout_events=(
                "recovered",
                "count"
            ),

            recoveries=(
                "recovered",
                "sum"
            )

        )
        .reset_index()
    )

    result[
        "recovery_rate"
    ] = np.where(

        result[
            "stockout_events"
        ] > 0,

        result[
            "recoveries"
        ]
        /
        result[
            "stockout_events"
        ]
        * 100,

        np.nan

    )

    return result


# ============================================================
# OBJECTIVE 2
# ============================================================

def objective2(
    long_df
):

    step(
        "OBJECTIVE 2 — BBRI"
    )

    recovery = recovery_by_bank_group(
        long_df
    )

    metrics = (
        long_df
        .groupby(
            [
                "hospital_code",
                "blood_bank_name",
                "district",
                "city",
                "area"
            ]
        )
        .agg(

            observations=(
                "stock",
                "count"
            ),

            stockout_frequency=(
                "stock",
                lambda x:
                (x == 0).mean()
            ),

            low_stock_frequency=(
                "stock",
                lambda x:
                x.between(
                    1,
                    LOW_STOCK_MAX
                ).mean()
            ),

            stock_volatility=(
                "stock",
                lambda x:
                (
                    x.std(ddof=0)
                    /
                    (
                        x.mean()
                        if x.mean() != 0
                        else 1
                    )
                )
            )

        )
        .reset_index()
    )

    metrics[
        "stock_volatility"
    ] = (
        metrics[
            "stock_volatility"
        ]
        .replace(
            [
                np.inf,
                -np.inf
            ],
            np.nan
        )
        .fillna(0)
    )

    recovery_bank = (
        recovery
        .groupby(
            "hospital_code"
        )
        .agg(
            recovery_rate=(
                "recovery_rate",
                "mean"
            )
        )
        .reset_index()
    )

    metrics = metrics.merge(
        recovery_bank,
        on="hospital_code",
        how="left"
    )

    metrics[
        "stockout_score"
    ] = percentile_rank(
        metrics[
            "stockout_frequency"
        ]
    )

    metrics[
        "low_stock_score"
    ] = percentile_rank(
        metrics[
            "low_stock_frequency"
        ]
    )

    metrics[
        "volatility_score"
    ] = percentile_rank(
        metrics[
            "stock_volatility"
        ]
    )

    metrics[
        "recovery_rate"
    ] = metrics[
        "recovery_rate"
    ].fillna(100)

    metrics[
        "recovery_score"
    ] = (
        100
        -
        metrics[
            "recovery_rate"
        ]
    )

    metrics[
        "BBRI"
    ] = (

        0.35
        *
        metrics[
            "stockout_score"
        ]

        +

        0.25
        *
        metrics[
            "low_stock_score"
        ]

        +

        0.20
        *
        metrics[
            "volatility_score"
        ]

        +

        0.20
        *
        metrics[
            "recovery_score"
        ]

    )

    metrics[
        "risk_class"
    ] = np.select(

        [

            metrics["BBRI"] < 25,

            metrics["BBRI"] < 50,

            metrics["BBRI"] < 75

        ],

        [

            "Low",
            "Moderate",
            "High"

        ],

        default="Critical"

    )

    metrics.to_csv(

        OUTPUT_DIR /
        "objective2" /
        "objective2_BBRI_corrected.csv",

        index=False

    )

    print(
        "BBRI rows:",
        len(metrics)
    )

    return metrics


# ============================================================
# OBJECTIVE 3
# ============================================================

def objective3(
    bbri,
    long_df
):

    step(
        "OBJECTIVE 3 — DBARI"
    )

    district = (
        bbri
        .groupby(
            "district"
        )
        .agg(

            average_BBRI=(
                "BBRI",
                "mean"
            ),

            median_BBRI=(
                "BBRI",
                "median"
            ),

            highest_BBRI=(
                "BBRI",
                "max"
            ),

            blood_banks=(
                "hospital_code",
                "nunique"
            )

        )
        .reset_index()
    )

    stock = (
        long_df
        .groupby(
            "district"
        )
        .agg(

            stockout_percentage=(
                "stock",
                lambda x:
                (x == 0).mean()
                * 100
            ),

            low_stock_percentage=(
                "stock",
                lambda x:
                x.between(
                    1,
                    LOW_STOCK_MAX
                ).mean()
                * 100
            )

        )
        .reset_index()
    )

    result = district.merge(
        stock,
        on="district",
        how="left"
    )

    result[
        "BBRI_component"
    ] = percentile_rank(
        result[
            "average_BBRI"
        ]
    )

    result[
        "stockout_component"
    ] = percentile_rank(
        result[
            "stockout_percentage"
        ]
    )

    result[
        "low_stock_component"
    ] = percentile_rank(
        result[
            "low_stock_percentage"
        ]
    )

    result[
        "DBARI"
    ] = (

        0.50
        *
        result[
            "BBRI_component"
        ]

        +

        0.30
        *
        result[
            "stockout_component"
        ]

        +

        0.20
        *
        result[
            "low_stock_component"
        ]

    )

    result[
        "risk_class"
    ] = np.select(

        [

            result["DBARI"] < 25,

            result["DBARI"] < 50,

            result["DBARI"] < 75

        ],

        [

            "Low",
            "Moderate",
            "High"

        ],

        default="Very High"

    )

    result.to_csv(

        OUTPUT_DIR /
        "objective3" /
        "objective3_DBARI.csv",

        index=False

    )

    print(
        "DBARI districts:",
        len(result)
    )

    return result


# ============================================================
# OBJECTIVE 4
# ============================================================

def objective4(
    long_df
):

    step(
        "OBJECTIVE 4 — Hotspots"
    )

    recovery = recovery_by_bank_group(
        long_df
    )

    bank_district = (
        long_df[
            [
                "hospital_code",
                "district"
            ]
        ]
        .drop_duplicates()
    )

    recovery = recovery.merge(
        bank_district,
        on="hospital_code",
        how="left"
    )

    recovery_district = (
        recovery
        .groupby(
            [
                "district",
                "blood_group"
            ]
        )
        .agg(
            recovery_rate=(
                "recovery_rate",
                "mean"
            )
        )
        .reset_index()
    )

    base = (
        long_df
        .groupby(
            [
                "district",
                "blood_group"
            ]
        )
        .agg(

            observations=(
                "stock",
                "count"
            ),

            blood_banks=(
                "hospital_code",
                "nunique"
            ),

            average_stock=(
                "stock",
                "mean"
            ),

            median_stock=(
                "stock",
                "median"
            ),

            stockout_percentage=(
                "stock",
                lambda x:
                (x == 0).mean()
                * 100
            ),

            low_stock_percentage=(
                "stock",
                lambda x:
                x.between(
                    1,
                    LOW_STOCK_MAX
                ).mean()
                * 100
            ),

            stock_volatility=(
                "stock",
                lambda x:
                (
                    x.std(ddof=0)
                    /
                    (
                        x.mean()
                        if x.mean() != 0
                        else 1
                    )
                )
            )

        )
        .reset_index()
    )

    base = base.merge(
        recovery_district,
        on=[
            "district",
            "blood_group"
        ],
        how="left"
    )

    eligible = base[
        base[
            "observations"
        ]
        >=
        MIN_HOTSPOT_OBSERVATIONS
    ].copy()

    eligible[
        "stock_volatility"
    ] = (
        eligible[
            "stock_volatility"
        ]
        .replace(
            [
                np.inf,
                -np.inf
            ],
            np.nan
        )
        .fillna(0)
    )

    for column in [
        "stockout_percentage",
        "low_stock_percentage",
        "stock_volatility"
    ]:

        eligible[
            column + "_score"
        ] = percentile_rank(
            eligible[column]
        )

    eligible[
        "recovery_weakness_score"
    ] = percentile_rank(
        100
        -
        eligible[
            "recovery_rate"
        ].fillna(100)
    )

    eligible[
        "hotspot_score"
    ] = (

        0.25
        *
        eligible[
            "stockout_percentage_score"
        ]

        +

        0.25
        *
        eligible[
            "low_stock_percentage_score"
        ]

        +

        0.25
        *
        eligible[
            "stock_volatility_score"
        ]

        +

        0.25
        *
        eligible[
            "recovery_weakness_score"
        ]

    )

    q50 = eligible[
        "hotspot_score"
    ].quantile(0.50)

    q75 = eligible[
        "hotspot_score"
    ].quantile(0.75)

    q90 = eligible[
        "hotspot_score"
    ].quantile(0.90)

    eligible[
        "hotspot_class"
    ] = np.select(

        [

            eligible[
                "hotspot_score"
            ] >= q90,

            eligible[
                "hotspot_score"
            ] >= q75,

            eligible[
                "hotspot_score"
            ] >= q50

        ],

        [

            "Critical",
            "High",
            "Moderate"

        ],

        default="Lower"

    )

    eligible.to_csv(

        OUTPUT_DIR /
        "objective4" /
        "objective4_district_bloodgroup_hotspots.csv",

        index=False

    )

    eligible[
        eligible[
            "hotspot_class"
        ]
        ==
        "Critical"
    ].to_csv(

        OUTPUT_DIR /
        "objective4" /
        "objective4_critical_hotspots.csv",

        index=False

    )

    eligible[
        eligible[
            "hotspot_class"
        ].isin(
            [
                "High",
                "Critical"
            ]
        )
    ].to_csv(

        OUTPUT_DIR /
        "objective4" /
        "objective4_high_and_critical_hotspots.csv",

        index=False

    )

    print(
        "Hotspot rows:",
        len(eligible)
    )


# ============================================================
# OBJECTIVE 5
# ============================================================

def objective5(
    bbri,
    dbari,
    master,
    coordinates
):

    step(
        "OBJECTIVE 5 — Geographic Vulnerability"
    )

    geo = master.merge(
        coordinates,
        on="hospital_code",
        how="inner"
    )

    geo[
        "latitude"
    ] = safe_numeric(
        geo["latitude"]
    )

    geo[
        "longitude"
    ] = safe_numeric(
        geo["longitude"]
    )

    geo = geo[
        geo["latitude"].notna()
        &
        geo["longitude"].notna()
        &
        (geo["latitude"] != 0)
        &
        (geo["longitude"] != 0)
    ].copy()

    geo = geo[
        ~geo[
            "hospital_code"
        ].isin(
            BAD_COORDINATE_CODES
        )
    ].copy()

    geo = geo.merge(

        bbri[
            [
                "hospital_code",
                "BBRI"
            ]
        ],

        on="hospital_code",

        how="left"

    )

    geo = geo.merge(

        dbari[
            [
                "district",
                "DBARI"
            ]
        ],

        on="district",

        how="left"

    )

    coords = geo[
        [
            "latitude",
            "longitude"
        ]
    ].to_numpy()

    nearest = []
    local_counts = []

    for i in range(
        len(coords)
    ):

        distances = haversine_km(

            coords[i, 0],
            coords[i, 1],

            coords[:, 0],
            coords[:, 1]

        )

        distances[i] = np.inf

        nearest.append(
            float(
                np.min(
                    distances
                )
            )
        )

        local_counts.append(
            int(
                (
                    (
                        distances <= 50
                    )
                    &
                    np.isfinite(
                        distances
                    )
                ).sum()
            )
        )

    geo[
        "nearest_bank_distance_km"
    ] = nearest

    geo[
        "banks_within_50km"
    ] = local_counts

    geo[
        "BBRI_score"
    ] = percentile_rank(
        geo["BBRI"]
    )

    geo[
        "DBARI_score"
    ] = percentile_rank(
        geo["DBARI"]
    )

    geo[
        "isolation_score"
    ] = percentile_rank(
        geo[
            "nearest_bank_distance_km"
        ]
    )

    geo[
        "local_access_score"
    ] = percentile_rank(
        -geo[
            "banks_within_50km"
        ]
    )

    geo[
        "GVS"
    ] = (

        0.40
        *
        geo["BBRI_score"]

        +

        0.25
        *
        geo["DBARI_score"]

        +

        0.20
        *
        geo["isolation_score"]

        +

        0.15
        *
        geo["local_access_score"]

    )

    geo[
        "geographic_vulnerability"
    ] = np.select(

        [

            geo["GVS"] < 25,

            geo["GVS"] < 50,

            geo["GVS"] < 75

        ],

        [

            "Low",
            "Moderate",
            "High"

        ],

        default="Very High"

    )

    geo.to_csv(

        OUTPUT_DIR /
        "objective5" /
        "objective5_geographic_vulnerability_final.csv",

        index=False

    )

    (
        geo
        .groupby(
            "district"
        )
        .agg(

            geographic_banks=(
                "hospital_code",
                "nunique"
            ),

            average_GVS=(
                "GVS",
                "mean"
            ),

            median_GVS=(
                "GVS",
                "median"
            ),

            highest_GVS=(
                "GVS",
                "max"
            ),

            average_nearest_distance_km=(
                "nearest_bank_distance_km",
                "mean"
            )

        )
        .reset_index()
        .to_csv(

            OUTPUT_DIR /
            "objective5" /
            "objective5_district_geographic_vulnerability_final.csv",

            index=False

        )
    )

    print(
        "Geographic banks:",
        len(geo)
    )

    return geo


# ============================================================
# MODEL DATA
# ============================================================

def prepare_model_dataset(
    long_df
):

    data = (
        long_df
        .sort_values(
            [
                "hospital_code",
                "blood_group",
                "snapshot_date"
            ]
        )
        .copy()
    )

    grouped = data.groupby(
        [
            "hospital_code",
            "blood_group"
        ]
    )

    data[
        "previous_stock"
    ] = grouped[
        "stock"
    ].shift(1)

    data[
        "stock_change"
    ] = (
        data[
            "stock"
        ]
        -
        data[
            "previous_stock"
        ]
    )

    data[
        "rolling_mean_3"
    ] = grouped[
        "stock"
    ].transform(
        lambda x:
        x.rolling(
            3,
            min_periods=1
        ).mean()
    )

    data[
        "rolling_std_3"
    ] = grouped[
        "stock"
    ].transform(
        lambda x:
        x.rolling(
            3,
            min_periods=1
        ).std()
    ).fillna(0)

    data[
        "previous_stockout"
    ] = (
        data[
            "previous_stock"
        ]
        .fillna(-1)
        .eq(0)
        .astype(int)
    )

    data[
        "previous_low_stock"
    ] = (
        data[
            "previous_stock"
        ]
        .fillna(-1)
        .between(
            1,
            LOW_STOCK_MAX
        )
        .astype(int)
    )

    data[
        "stock_vs_rolling_mean"
    ] = (
        data[
            "stock"
        ]
        -
        data[
            "rolling_mean_3"
        ]
    )

    data[
        "trend"
    ] = (
        data[
            "stock_change"
        ]
        .fillna(0)
    )

    data[
        "recovered_from_previous_stockout"
    ] = (

        (
            data[
                "previous_stockout"
            ] == 1
        )

        &
        
        (
            data[
                "stock"
            ] > 0
        )

    ).astype(int)

    data[
        "next_stock"
    ] = grouped[
        "stock"
    ].shift(-1)

    data = data[
        data[
            "next_stock"
        ].notna()
        &
        data[
            "previous_stock"
        ].notna()
    ].copy()

    data[
        "target_stockout"
    ] = (
        data[
            "next_stock"
        ]
        .eq(0)
        .astype(int)
    )

    data[
        MODEL_FEATURES
    ] = (
        data[
            MODEL_FEATURES
        ]
        .replace(
            [
                np.inf,
                -np.inf
            ],
            np.nan
        )
        .fillna(0)
    )

    return data


# ============================================================
# OBJECTIVE 6
# ============================================================

def objective6(
    long_df
):

    step(
        "OBJECTIVE 6 — Stockout Prediction"
    )

    if not SKLEARN_AVAILABLE:

        print(
            "scikit-learn unavailable."
        )

        return None

    data = prepare_model_dataset(
        long_df
    )

    dates = (
        data[
            "snapshot_date"
        ]
        .drop_duplicates()
        .sort_values()
        .tolist()
    )

    if len(dates) < 3:

        raise ValueError(
            "At least 3 snapshots are required."
        )

    split_index = max(
        1,
        int(
            len(dates) * 0.80
        )
    )

    if split_index >= len(dates):

        split_index = (
            len(dates) - 1
        )

    cutoff = dates[
        split_index
    ]

    train = data[
        data[
            "snapshot_date"
        ] < cutoff
    ].copy()

    test = data[
        data[
            "snapshot_date"
        ] >= cutoff
    ].copy()

    X_train = train[
        MODEL_FEATURES
    ].astype(float)

    y_train = train[
        "target_stockout"
    ].astype(int)

    X_test = test[
        MODEL_FEATURES
    ].astype(float)

    y_test = test[
        "target_stockout"
    ].astype(int)

    hgb = HistGradientBoostingClassifier(
        random_state=RANDOM_STATE
    )

    rf = RandomForestClassifier(
        n_estimators=300,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        class_weight="balanced_subsample"
    )

    hgb.fit(
        X_train,
        y_train
    )

    rf.fit(
        X_train,
        y_train
    )

    comparison = []

    prediction_df = test[
        [
            "snapshot_date",
            "hospital_code",
            "blood_bank_name",
            "district",
            "blood_group"
        ]
    ].copy()

    for name, model in {

        "HistGradientBoosting":
            hgb,

        "Random Forest":
            rf

    }.items():

        probability = model.predict_proba(
            X_test
        )[:, 1]

        predicted = (
            probability >= 0.5
        ).astype(int)

        try:

            auc = roc_auc_score(
                y_test,
                probability
            )

        except Exception:

            auc = np.nan

        comparison.append({

            "model":
                name,

            "accuracy":
                accuracy_score(
                    y_test,
                    predicted
                ),

            "precision":
                precision_score(
                    y_test,
                    predicted,
                    zero_division=0
                ),

            "recall":
                recall_score(
                    y_test,
                    predicted,
                    zero_division=0
                ),

            "f1":
                f1_score(
                    y_test,
                    predicted,
                    zero_division=0
                ),

            "roc_auc":
                auc

        })

        if name == "HistGradientBoosting":

            prediction_df[
                "hgb_probability"
            ] = probability

            prediction_df[
                "hgb_prediction"
            ] = predicted

    comparison_df = pd.DataFrame(
        comparison
    )

    comparison_df.to_csv(

        OUTPUT_DIR /
        "objective6" /
        "objective6_model_comparison.csv",

        index=False

    )

    pd.DataFrame({

        "feature":
            MODEL_FEATURES

    }).to_csv(

        OUTPUT_DIR /
        "objective6" /
        "objective6_model_features.csv",

        index=False

    )

    prediction_df.to_csv(

        OUTPUT_DIR /
        "objective6" /
        "objective6_stockout_predictions.csv",

        index=False

    )

    if joblib is not None:

        joblib.dump(

            hgb,

            MODEL_DIR /
            "histgradientboosting_stockout_model.pkl"

        )

        joblib.dump(

            rf,

            MODEL_DIR /
            "random_forest_stockout_model.pkl"

        )

    print(
        comparison_df
    )

    return {

        "data":
            data,

        "train":
            train,

        "test":
            test,

        "hgb":
            hgb,

        "rf":
            rf,

        "comparison":
            comparison_df,

        "test_X":
            X_test,

        "test_y":
            y_test

    }


# ============================================================
# OBJECTIVE 7
# ============================================================

def objective7(
    model_result
):

    step(
        "OBJECTIVE 7 — XAI"
    )

    if (
        model_result is None
        or
        not SKLEARN_AVAILABLE
    ):

        return

    importance = permutation_importance(

        model_result["hgb"],

        model_result["test_X"],

        model_result["test_y"],

        scoring="roc_auc",

        n_repeats=5,

        random_state=RANDOM_STATE,

        n_jobs=-1

    )

    result = pd.DataFrame({

        "feature":
            MODEL_FEATURES,

        "importance_mean":
            importance.importances_mean,

        "importance_std":
            importance.importances_std

    })

    result = (
        result
        .sort_values(
            "importance_mean",
            ascending=False
        )
        .reset_index(
            drop=True
        )
    )

    result[
        "rank"
    ] = (
        np.arange(
            len(result)
        )
        + 1
    )

    result.to_csv(

        OUTPUT_DIR /
        "objective7" /
        "objective7_feature_importance.csv",

        index=False

    )

    explanation_rows = []

    for _, row in result.iterrows():

        explanation_rows.append({

            "feature":
                row["feature"],

            "importance":
                row["importance_mean"],

            "rank":
                row["rank"],

            "interpretation":
                (
                    f"{row['feature']} "
                    f"permutation importance = "
                    f"{row['importance_mean']:.4f}"
                )

        })

    pd.DataFrame(
        explanation_rows
    ).to_csv(

        OUTPUT_DIR /
        "objective7" /
        "objective7_stockout_explanations.csv",

        index=False

    )


# ============================================================
# OBJECTIVE 8
# ============================================================

def objective8(
    model_result
):

    step(
        "OBJECTIVE 8 — Blood Groups"
    )

    if model_result is None:

        return

    test = model_result[
        "test"
    ].copy()

    probabilities = (
        model_result[
            "hgb"
        ]
        .predict_proba(
            model_result[
                "test_X"
            ]
        )[:, 1]
    )

    test[
        "prediction_probability"
    ] = probabilities

    rows = []

    for group in BLOOD_GROUPS:

        subset = test[
            test[
                "blood_group"
            ]
            ==
            group
        ]

        if subset.empty:

            continue

        y = subset[
            "target_stockout"
        ]

        predicted = (
            subset[
                "prediction_probability"
            ]
            >= 0.5
        ).astype(int)

        try:

            auc = roc_auc_score(
                y,
                subset[
                    "prediction_probability"
                ]
            )

        except Exception:

            auc = np.nan

        rows.append({

            "blood_group":
                group,

            "observations":
                len(subset),

            "average_stock":
                subset[
                    "stock"
                ].mean(),

            "stockout_percentage":
                y.mean()
                * 100,

            "low_stock_percentage":
                subset[
                    "stock"
                ]
                .between(
                    1,
                    LOW_STOCK_MAX
                )
                .mean()
                * 100,

            "accuracy":
                accuracy_score(
                    y,
                    predicted
                ),

            "precision":
                precision_score(
                    y,
                    predicted,
                    zero_division=0
                ),

            "recall":
                recall_score(
                    y,
                    predicted,
                    zero_division=0
                ),

            "f1":
                f1_score(
                    y,
                    predicted,
                    zero_division=0
                ),

            "roc_auc":
                auc

        })

    pd.DataFrame(
        rows
    ).to_csv(

        OUTPUT_DIR /
        "objective8" /
        "objective8_blood_group_comparison.csv",

        index=False

    )


# ============================================================
# OBJECTIVE 9
# ============================================================

def objective9(
    geo
):

    step(
        "OBJECTIVE 9 — Network Resilience"
    )

    if nx is None:

        print(
            "networkx unavailable."
        )

        return None

    if geo.empty:

        return None

    nodes = (
        geo[
            [
                "hospital_code",
                "blood_bank_name",
                "district",
                "latitude",
                "longitude"
            ]
        ]
        .drop_duplicates(
            "hospital_code"
        )
    )

    G = nx.Graph()

    for _, row in nodes.iterrows():

        G.add_node(
            row["hospital_code"],
            blood_bank_name=
                row["blood_bank_name"],
            district=
                row["district"]
        )

    records = nodes.to_dict(
        orient="records"
    )

    for i in range(
        len(records)
    ):

        for j in range(
            i + 1,
            len(records)
        ):

            a = records[i]
            b = records[j]

            distance = haversine_km(

                a["latitude"],
                a["longitude"],

                b["latitude"],
                b["longitude"]

            )

            if (
                distance <=
                NETWORK_RADIUS_KM
                and
                distance > 0
            ):

                G.add_edge(

                    a[
                        "hospital_code"
                    ],

                    b[
                        "hospital_code"
                    ],

                    distance_km=
                        float(
                            distance
                        )

                )

    degree = nx.degree_centrality(
        G
    )

    betweenness = nx.betweenness_centrality(
        G,
        weight="distance_km"
    )

    closeness = nx.closeness_centrality(
        G,
        distance="distance_km"
    )

    centrality_rows = []

    for node in G.nodes():

        attrs = G.nodes[
            node
        ]

        centrality_rows.append({

            "hospital_code":
                node,

            "blood_bank_name":
                attrs.get(
                    "blood_bank_name"
                ),

            "district":
                attrs.get(
                    "district"
                ),

            "degree":
                G.degree(
                    node
                ),

            "degree_centrality":
                degree.get(
                    node,
                    0
                ),

            "betweenness_centrality":
                betweenness.get(
                    node,
                    0
                ),

            "closeness_centrality":
                closeness.get(
                    node,
                    0
                )

        })

    centrality = pd.DataFrame(
        centrality_rows
    )

    centrality.to_csv(

        OUTPUT_DIR /
        "objective9" /
        "objective9_network_centrality.csv",

        index=False

    )

    baseline_lcc = max(

        [
            len(component)
            for component
            in nx.connected_components(
                G
            )
        ]

        or

        [0]

    )

    candidates = (
        centrality
        .sort_values(
            "betweenness_centrality",
            ascending=False
        )
        .head(20)
    )

    failures = []

    for _, row in candidates.iterrows():

        node = row[
            "hospital_code"
        ]

        H = G.copy()

        H.remove_node(
            node
        )

        after_lcc = max(

            [
                len(component)
                for component
                in nx.connected_components(
                    H
                )
            ]

            or

            [0]

        )

        loss = max(
            baseline_lcc
            -
            after_lcc,
            0
        )

        loss_percent = (

            loss
            /
            baseline_lcc
            * 100

            if baseline_lcc
            else 0

        )

        failures.append({

            "hospital_code":
                node,

            "blood_bank_name":
                row[
                    "blood_bank_name"
                ],

            "district":
                row[
                    "district"
                ],

            "baseline_lcc":
                baseline_lcc,

            "largest_component_after_failure":
                after_lcc,

            "lcc_node_loss":
                loss,

            "lcc_loss_percent":
                loss_percent

        })

    failure_df = pd.DataFrame(
        failures
    )

    failure_df[
        "failure_impact_score"
    ] = percentile_rank(
        failure_df[
            "lcc_loss_percent"
        ]
    )

    failure_df.to_csv(

        OUTPUT_DIR /
        "objective9" /
        "objective9_failure_impact.csv",

        index=False

    )

    critical = centrality.merge(

        failure_df[
            [
                "hospital_code",
                "failure_impact_score"
            ]
        ],

        on="hospital_code",

        how="left"

    )

    critical[
        "betweenness_score"
    ] = percentile_rank(
        critical[
            "betweenness_centrality"
        ]
    )

    critical[
        "degree_score"
    ] = percentile_rank(
        critical[
            "degree_centrality"
        ]
    )

    critical[
        "failure_impact_score"
    ] = critical[
        "failure_impact_score"
    ].fillna(0)

    critical[
        "network_criticality"
    ] = (

        0.40
        *
        critical[
            "betweenness_score"
        ]

        +

        0.30
        *
        critical[
            "degree_score"
        ]

        +

        0.30
        *
        critical[
            "failure_impact_score"
        ]

    )

    critical = (
        critical
        .sort_values(
            "network_criticality",
            ascending=False
        )
        .head(20)
    )

    critical.to_csv(

        OUTPUT_DIR /
        "objective9" /
        "objective9_critical_network_nodes.csv",

        index=False

    )

    print(
        "Network nodes:",
        len(G.nodes())
    )

    print(
        "Network edges:",
        len(G.edges())
    )

    return {
        "graph": G,
        "centrality": centrality,
        "critical": critical
    }


# ============================================================
# OBJECTIVE 10
# ============================================================

def objective10(
    latest,
    geo
):

    step(
        "OBJECTIVE 10 — Redistribution"
    )

    if latest.empty:

        return

    # Coordinates are authoritative from Objective 5 / coordinates.csv.
    # latest already contains latitude/longitude, so exclude them before
    # merging to avoid pandas creating latitude_x/latitude_y columns.
    latest_for_merge = latest.drop(
        columns=[
            "latitude",
            "longitude"
        ],
        errors="ignore"
    )

    working = latest_for_merge.merge(

        geo[
            [
                "hospital_code",
                "latitude",
                "longitude"
            ]
        ],

        on="hospital_code",

        how="inner"

    )

    working = working[
        ~working[
            "hospital_code"
        ].isin(
            BAD_COORDINATE_CODES
        )
    ].copy()

    recommendations = []

    for group in BLOOD_GROUPS:

        group_df = working[
            [
                "hospital_code",
                "blood_bank_name",
                "district",
                "latitude",
                "longitude",
                group
            ]
        ].copy()

        group_df[
            "surplus"
        ] = (
            group_df[
                group
            ]
            -
            SAFETY_STOCK
        ).clip(
            lower=0
        )

        group_df[
            "deficit"
        ] = (
            SAFETY_STOCK
            -
            group_df[
                group
            ]
        ).clip(
            lower=0
        )

        donors = group_df[
            group_df[
                "surplus"
            ] > 0
        ]

        recipients = group_df[
            group_df[
                "deficit"
            ] > 0
        ]

        if donors.empty or recipients.empty:

            continue

        pairs = []

        for _, donor in donors.iterrows():

            for _, recipient in recipients.iterrows():

                distance = haversine_km(

                    donor[
                        "latitude"
                    ],

                    donor[
                        "longitude"
                    ],

                    recipient[
                        "latitude"
                    ],

                    recipient[
                        "longitude"
                    ]

                )

                if (
                    distance <=
                    REDISTRIBUTION_RADIUS_KM
                    and
                    distance > 0
                ):

                    pairs.append({

                        "blood_group":
                            group,

                        "donor_code":
                            donor[
                                "hospital_code"
                            ],

                        "donor_name":
                            donor[
                                "blood_bank_name"
                            ],

                        "donor_district":
                            donor[
                                "district"
                            ],

                        "recipient_code":
                            recipient[
                                "hospital_code"
                            ],

                        "recipient_name":
                            recipient[
                                "blood_bank_name"
                            ],

                        "recipient_district":
                            recipient[
                                "district"
                            ],

                        "distance_km":
                            float(
                                distance
                            ),

                        "possible_units":
                            min(
                                float(
                                    donor[
                                        "surplus"
                                    ]
                                ),
                                float(
                                    recipient[
                                        "deficit"
                                    ]
                                )
                            )

                    })

        pairs = sorted(

            pairs,

            key=lambda row: (

                -row[
                    "possible_units"
                ],

                row[
                    "distance_km"
                ]

            )

        )

        donor_remaining = {

            normalize_code(
                row[
                    "hospital_code"
                ]
            ):

            float(
                row[
                    "surplus"
                ]
            )

            for _, row in donors.iterrows()

        }

        recipient_remaining = {

            normalize_code(
                row[
                    "hospital_code"
                ]
            ):

            float(
                row[
                    "deficit"
                ]
            )

            for _, row in recipients.iterrows()

        }

        for pair in pairs:

            donor_code = normalize_code(
                pair[
                    "donor_code"
                ]
            )

            recipient_code = normalize_code(
                pair[
                    "recipient_code"
                ]
            )

            units = min(

                donor_remaining[
                    donor_code
                ],

                recipient_remaining[
                    recipient_code
                ]

            )

            if units <= 0:
                continue

            donor_remaining[
                donor_code
            ] -= units

            recipient_remaining[
                recipient_code
            ] -= units

            pair[
                "optimized_units"
            ] = int(
                units
            )

            pair[
                "transport_km_units"
            ] = (
                pair[
                    "distance_km"
                ]
                *
                pair[
                    "optimized_units"
                ]
            )

            recommendations.append(
                pair
            )

    result = pd.DataFrame(
        recommendations
    )

    if result.empty:

        result = pd.DataFrame(
            columns=[

                "blood_group",
                "donor_code",
                "donor_name",
                "donor_district",
                "recipient_code",
                "recipient_name",
                "recipient_district",
                "distance_km",
                "optimized_units",
                "transport_km_units"

            ]
        )

    result.to_csv(

        OUTPUT_DIR /
        "objective10" /
        "objective10_optimized_redistribution.csv",

        index=False

    )

    result.to_csv(

        OUTPUT_DIR /
        "objective10" /
        "objective10_feasible_transfers_network_aware.csv",

        index=False

    )

    summary = pd.DataFrame({

        "metric": [

            "transfer_rows",
            "total_units",
            "average_distance_km",
            "maximum_distance_km",
            "zero_distance_transfers"

        ],

        "value": [

            len(result),

            (
                result[
                    "optimized_units"
                ].sum()
                if not result.empty
                else 0
            ),

            (
                result[
                    "distance_km"
                ].mean()
                if not result.empty
                else 0
            ),

            (
                result[
                    "distance_km"
                ].max()
                if not result.empty
                else 0
            ),

            (
                (
                    result[
                        "distance_km"
                    ] == 0
                ).sum()
                if not result.empty
                else 0
            )

        ]

    })

    summary.to_csv(

        OUTPUT_DIR /
        "objective10" /
        "objective10_redistribution_summary.csv",

        index=False

    )

    result[
        result[
            "distance_km"
        ] == 0
    ].to_csv(

        OUTPUT_DIR /
        "objective10" /
        "objective10_zero_distance_check.csv",

        index=False

    )

    print(
        "Redistribution rows:",
        len(result)
    )

    print(
        "Redistribution units:",
        int(
            result[
                "optimized_units"
            ].sum()
        )
        if not result.empty
        else 0
    )


# ============================================================
# OBJECTIVE 11
# ============================================================

def objective11(
    latest,
    bbri,
    geo
):

    step(
        "OBJECTIVE 11 — Alternatives"
    )

    if latest.empty:

        return

    current = latest.merge(

        bbri[
            [
                "hospital_code",
                "BBRI"
            ]
        ],

        on="hospital_code",

        how="left"

    )

    current = current.merge(

        geo[
            [
                "hospital_code",
                "latitude",
                "longitude"
            ]
        ],

        on="hospital_code",

        how="inner"

    )

    current = current[
        ~current[
            "hospital_code"
        ].isin(
            BAD_COORDINATE_CODES
        )
    ].copy()

    rows = []

    for _, origin in current.iterrows():

        origin_code = normalize_code(
            origin[
                "hospital_code"
            ]
        )

        for group in BLOOD_GROUPS:

            origin_stock = float(
                origin[group]
            )

            if origin_stock > SAFETY_STOCK:

                continue

            for _, alternative in current.iterrows():

                alternative_code = normalize_code(
                    alternative[
                        "hospital_code"
                    ]
                )

                if (
                    alternative_code
                    ==
                    origin_code
                ):

                    continue

                alternative_stock = float(
                    alternative[group]
                )

                if alternative_stock <= 0:

                    continue

                distance = haversine_km(

                    float(
                        origin[
                            "latitude"
                        ]
                    ),

                    float(
                        origin[
                            "longitude"
                        ]
                    ),

                    float(
                        alternative[
                            "latitude"
                        ]
                    ),

                    float(
                        alternative[
                            "longitude"
                        ]
                    )

                )

                if distance > REDISTRIBUTION_RADIUS_KM:

                    continue

                alt_bbri = alternative.get(
                    "BBRI"
                )

                if pd.isna(
                    alt_bbri
                ):

                    alt_bbri = 50

                alt_bbri = float(
                    alt_bbri
                )

                availability_score = (

                    alternative_stock
                    /
                    max(
                        current[
                            group
                        ].max(),
                        1
                    )
                    * 100

                )

                bbri_score = max(
                    0,
                    100 - alt_bbri
                )

                proximity_score = (

                    1
                    -
                    distance
                    /
                    REDISTRIBUTION_RADIUS_KM

                ) * 100

                decision_score = (

                    0.50
                    *
                    availability_score

                    +

                    0.25
                    *
                    bbri_score

                    +

                    0.25
                    *
                    proximity_score

                )

                rows.append({

                    "Origin Code":
                        origin_code,

                    "Origin Bank":
                        origin[
                            "blood_bank_name"
                        ],

                    "Origin District":
                        origin[
                            "district"
                        ],

                    "Blood Group":
                        group,

                    "Origin Stock":
                        origin_stock,

                    "Alternative Code":
                        alternative_code,

                    "Alternative Bank":
                        alternative[
                            "blood_bank_name"
                        ],

                    "Alternative District":
                        alternative[
                            "district"
                        ],

                    "Alternative Stock":
                        alternative_stock,

                    "Predicted Stockout Probability":
                        np.nan,

                    "Alternative BBRI":
                        alt_bbri,

                    "Distance km":
                        distance,

                    "Decision Support Score":
                        decision_score

                })

    ranking = pd.DataFrame(
        rows
    )

    if ranking.empty:

        ranking = pd.DataFrame(
            columns=[

                "Origin Code",
                "Origin Bank",
                "Origin District",
                "Blood Group",
                "Origin Stock",
                "Alternative Code",
                "Alternative Bank",
                "Alternative District",
                "Alternative Stock",
                "Predicted Stockout Probability",
                "Alternative BBRI",
                "Distance km",
                "Decision Support Score",
                "Rank"

            ]
        )

    else:

        ranking = (
            ranking
            .sort_values(
                [
                    "Origin Code",
                    "Blood Group",
                    "Decision Support Score"
                ],
                ascending=[
                    True,
                    True,
                    False
                ]
            )
        )

        ranking[
            "Rank"
        ] = (
            ranking
            .groupby(
                [
                    "Origin Code",
                    "Blood Group"
                ]
            )
            .cumcount()
            + 1
        )

    ranking.to_csv(

        OUTPUT_DIR /
        "objective11" /
        "Maharashtra_Alternative_Blood_Bank_Ranking.csv",

        index=False

    )

    print(
        "Objective 11 rows:",
        len(ranking)
    )


# ============================================================
# UPDATE LOG
# ============================================================

def write_update_log(

    snapshot_date,

    source_file,

    source_rows,

    valid_rows,

    new_rows,

    status,

    message=""

):

    conn = sqlite3.connect(
        DATABASE
    )

    conn.execute(
        """
        INSERT INTO update_log (

            run_time,
            snapshot_date,
            source_file,
            source_rows,
            valid_rows,
            new_rows,
            status,
            message

        )

        VALUES (

            datetime('now'),
            ?, ?, ?, ?, ?, ?, ?

        )
        """,

        [

            str(
                snapshot_date
                or ""
            ),

            source_file,

            int(
                source_rows
            ),

            int(
                valid_rows
            ),

            int(
                new_rows
            ),

            status,

            message

        ]

    )

    conn.commit()

    conn.close()


# ============================================================
# MAIN
# ============================================================

def main():

    banner(
        "MAHARASHTRA BLOOD FINDER\n"
        "FULL DAILY UPDATE PIPELINE"
    )

    ensure_directories()

    print(
        "Snapshot folder:",
        SNAPSHOT_DIR
    )

    print(
        "Master file:",
        MASTER_FILE
    )

    print(
        "Coordinate file:",
        COORDINATE_FILE
    )

    print(
        "Database:",
        DATABASE
    )

    master = load_master()

    coordinates = load_coordinates()

    print(
        "Master blood banks:",
        master[
            "hospital_code"
        ].nunique()
    )

    print(
        "Coordinate records:",
        len(coordinates)
    )

    initialize_database()

    snapshot_files = sorted(

        [

            path

            for path in SNAPSHOT_DIR.rglob("*")

            if (

                path.is_file()

                and

                path.suffix.lower() == ".pdf"

            )

        ],

        key=lambda p:
            p.name

    )

    print(
        "Snapshots found:",
        len(snapshot_files)
    )

    successful = 0
    failed = 0

    total_processed = 0
    total_inserted = 0

    # ========================================================
    # PROCESS SNAPSHOTS
    # ========================================================

    for path in snapshot_files:

        print()
        print(
            "Processing:",
            path.name
        )

        try:

            text = read_snapshot_text(
                path
            )

            records = extract_source_records(
                text
            )

            if not records:

                raise ValueError(
                    "No JSON source records found."
                )

            if len(records) < MIN_EXPECTED_SOURCE_ROWS:

                hospital_occurrences = len(
                    re.findall(
                        r'"hospitalCode"\s*:',
                        text,
                        flags=re.IGNORECASE
                    )
                )

                raise ValueError(
                    "PDF parse sanity check failed: "
                    f"only {len(records)} source records extracted "
                    f"from {path.name}; "
                    f"hospitalCode occurrences={hospital_occurrences}. "
                    "Snapshot was NOT imported."
                )

            snapshot = parse_snapshot(

                path,

                master,

                coordinates

            )

            source_rows = len(
                records
            )

            valid_rows = len(
                snapshot
            )

            if snapshot.empty:

                raise ValueError(
                    "No valid rows after parsing."
                )

            snapshot_date = snapshot[
                "snapshot_date"
            ].iloc[0]

            new_rows = insert_snapshot(
                snapshot
            )

            successful += 1

            total_processed += (
                valid_rows
            )

            total_inserted += (
                new_rows
            )

            write_update_log(

                snapshot_date,

                path.name,

                source_rows,

                valid_rows,

                new_rows,

                "SUCCESS",

                ""

            )

            print(
                "Source records:",
                source_rows
            )

            print(
                "Snapshot date:",
                snapshot_date
            )

            print(
                "Valid rows:",
                valid_rows
            )

            print(
                "New history rows inserted:",
                new_rows
            )

        except Exception as exc:

            failed += 1

            print(
                "FAILED:",
                exc
            )

            try:

                write_update_log(

                    "",

                    path.name,

                    0,

                    0,

                    0,

                    "FAILED",

                    str(exc)

                )

            except Exception as log_error:

                print(
                    "WARNING: "
                    "Could not write failure log:",
                    log_error
                )

    # ========================================================
    # REBUILD CURRENT DATA
    # ========================================================

    latest = rebuild_current_table()

    current, history = (
        export_database_csvs()
    )

    print()
    print(
        "Snapshots found:",
        len(snapshot_files)
    )

    print(
        "Successful snapshots:",
        successful
    )

    print(
        "Failed snapshots:",
        failed
    )

    print(
        "Rows processed:",
        total_processed
    )

    print(
        "New history rows:",
        total_inserted
    )

    print(
        "Unique snapshots:",
        (
            history[
                "snapshot_date"
            ].nunique()
            if not history.empty
            else 0
        )
    )

    print(
        "Historical rows:",
        len(history)
    )

    print(
        "Current blood banks:",
        len(current)
    )

    # ========================================================
    # IMPORTANT:
    # Do not run Objectives if no snapshots succeeded.
    # ========================================================

    if history.empty:

        raise RuntimeError(
            "No historical data exists in SQLite. "
            "Objective generation stopped."
        )

    # ========================================================
    # OBJECTIVES 1–11
    # ========================================================

    long_df = make_long_history(
        history
    )

    objective1(
        history
    )

    bbri = objective2(
        long_df
    )

    dbari = objective3(
        bbri,
        long_df
    )

    objective4(
        long_df
    )

    geo = objective5(

        bbri,

        dbari,

        master,

        coordinates

    )

    model_result = objective6(
        long_df
    )

    objective7(
        model_result
    )

    objective8(
        model_result
    )

    objective9(
        geo
    )

    objective10(

        latest,

        geo

    )

    objective11(

        latest,

        bbri,

        geo

    )

    # ========================================================
    # FINAL
    # ========================================================

    banner(
        "UPDATE COMPLETE — OBJECTIVES 1–11 REFRESHED"
    )

    print(
        "Successful snapshots:",
        successful
    )

    print(
        "Failed snapshots:",
        failed
    )

    print(
        "Historical rows:",
        len(history)
    )

    print(
        "Current blood banks:",
        len(current)
    )

    print(
        "Output directory:",
        OUTPUT_DIR
    )

    print()
    print(
        "Website outputs are ready."
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()