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

try:
    from scipy.optimize import linprog
    from scipy.sparse import lil_matrix, csr_matrix
    SCIPY_AVAILABLE = True
except ImportError:
    linprog = None
    lil_matrix = None
    csr_matrix = None
    SCIPY_AVAILABLE = False


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
    Extract the complete e-RaktKosh blood-bank record array from PDF text.

    The exported PDFs are visually JSON, but PDF text extraction introduces:
      * page-break newlines inside JSON strings (for example ``O-\nVe``)
      * literal control characters in some addresses
      * the repeated text ``Pretty print`` at page boundaries

    We normalize those PDF artifacts first and then decode the whole array.
    This is much safer than trying to reconstruct each record independently,
    because the source itself contains nested JSON objects.
    """

    if not text:
        return []

    # --------------------------------------------------------
    # PDF-text normalization
    # --------------------------------------------------------
    cleaned = text.replace("Pretty print", " ")
    cleaned = cleaned.replace("\r", " ").replace("\n", " ").replace("\t", " ")

    # Remove remaining ASCII control characters that pypdf can expose from
    # the source PDF. Keep ordinary printable Unicode intact.
    cleaned = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", " ", cleaned)

    # --------------------------------------------------------
    # Decode the JSON array itself.
    # --------------------------------------------------------
    array_start = cleaned.find("[")
    if array_start < 0:
        return []

    try:
        decoder = json.JSONDecoder()
        parsed, _ = decoder.raw_decode(cleaned[array_start:])
    except Exception:
        parsed = None

    if isinstance(parsed, list):
        valid = [
            item
            for item in parsed
            if isinstance(item, dict)
            and normalize_code(item.get("hospitalCode"))
        ]

        marker_count = len(
            re.findall(
                r'"hospitalCode"\s*:',
                cleaned,
                flags=re.IGNORECASE
            )
        )

        if valid and len(valid) == marker_count:
            return valid

    # --------------------------------------------------------
    # Conservative fallback: locate top-level objects beginning with
    # bldgrpcode1. This handles a future PDF whose outer array is damaged.
    # --------------------------------------------------------
    records = []
    seen_codes = set()

    record_starts = re.finditer(
        r'\{\s*"bldgrpcode1"\s*:',
        cleaned,
        flags=re.IGNORECASE
    )

    for match in record_starts:
        candidate = extract_json_object_at(cleaned, match.start())
        if candidate is None:
            continue

        try:
            obj = json.loads(candidate)
        except Exception:
            continue

        if not isinstance(obj, dict):
            continue

        code = normalize_code(obj.get("hospitalCode"))
        if not code or code in seen_codes:
            continue

        seen_codes.add(code)
        records.append(obj)

    # Last-resort marker-based recovery retained for unusual source changes.
    if not records:
        matches = list(
            re.finditer(
                r'"hospitalCode"\s*:',
                cleaned,
                flags=re.IGNORECASE
            )
        )

        for match in matches:
            start = find_record_start(cleaned, match.start())
            if start is None:
                continue

            candidate = extract_json_object_at(cleaned, start)
            if candidate is None:
                continue

            try:
                obj = json.loads(candidate)
            except Exception:
                continue

            if not isinstance(obj, dict):
                continue

            code = normalize_code(obj.get("hospitalCode"))
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

    # PDF extraction sometimes inserts a space/newline between the sign
    # and the "Ve" suffix, e.g. ``O-\nVe``. Allow optional whitespace.
    "A+":
        r"A\+\s*Ve",

    "A-":
        r"A-\s*Ve",

    "B+":
        r"B\+\s*Ve",

    "B-":
        r"B-\s*Ve",

    "O+":
        r"O\+\s*Ve",

    "O-":
        r"O-\s*Ve",

    "AB+":
        r"AB\+\s*Ve",

    "AB-":
        r"AB-\s*Ve"

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

# ============================================================
# OBJECTIVE 2 SUPPORT
# Corrected episode-based recovery for BBRI
# ============================================================

def bbri_recovery_metrics(
    long_df
):

    """
    Calculate recovery metrics for BBRI using stockout episodes,
    not repeated zero-stock observations.

    A stockout episode begins when a bank/blood-group is at zero
    and was not already in an ongoing zero-stock episode.

    The episode is considered recovered when stock first becomes
    positive again in a later snapshot.

    Consecutive zero observations therefore belong to one episode.
    """

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

    episode_rows = []

    for (
        hospital_code,
        blood_group
    ), group in data.groupby(
        [
            "hospital_code",
            "blood_group"
        ],
        sort=False
    ):

        group = (
            group
            .sort_values(
                "snapshot_date"
            )
            .reset_index(
                drop=True
            )
        )

        stocks = (
            pd.to_numeric(
                group["stock"],
                errors="coerce"
            )
            .fillna(0)
            .to_numpy()
        )

        in_episode = False
        start_index = None

        for i, stock_value in enumerate(
            stocks
        ):

            if (
                stock_value <= 0
                and not in_episode
            ):

                in_episode = True
                start_index = i

                continue

            if (
                stock_value > 0
                and in_episode
                and start_index is not None
            ):

                duration_snapshots = (
                    i
                    -
                    start_index
                )

                episode_rows.append({

                    "hospital_code":
                        hospital_code,

                    "blood_group":
                        blood_group,

                    "recovered":
                        1,

                    "recovery_snapshots":
                        duration_snapshots

                })

                in_episode = False
                start_index = None

        if (
            in_episode
            and start_index is not None
        ):

            episode_rows.append({

                "hospital_code":
                    hospital_code,

                "blood_group":
                    blood_group,

                "recovered":
                    0,

                "recovery_snapshots":
                    np.nan

            })

    if not episode_rows:

        return pd.DataFrame(
            columns=[
                "hospital_code",
                "stockout_episodes",
                "recovered_episodes",
                "unresolved_stockout_episodes",
                "recovery_rate",
                "mean_recovery_snapshots",
                "median_recovery_snapshots"
            ]
        )

    episodes = pd.DataFrame(
        episode_rows
    )

    result = (
        episodes
        .groupby(
            "hospital_code"
        )
        .agg(

            stockout_episodes=(
                "recovered",
                "count"
            ),

            recovered_episodes=(
                "recovered",
                "sum"
            ),

            mean_recovery_snapshots=(
                "recovery_snapshots",
                "mean"
            ),

            median_recovery_snapshots=(
                "recovery_snapshots",
                "median"
            )

        )
        .reset_index()
    )

    result[
        "unresolved_stockout_episodes"
    ] = (
        result[
            "stockout_episodes"
        ]
        -
        result[
            "recovered_episodes"
        ]
    )

    result[
        "recovery_rate"
    ] = np.where(

        result[
            "stockout_episodes"
        ] > 0,

        result[
            "recovered_episodes"
        ]
        /
        result[
            "stockout_episodes"
        ]
        * 100,

        100.0

    )

    return result


# ============================================================
# OBJECTIVE 2
# Corrected Blood Bank Risk Index (BBRI)
# ============================================================

def objective2(
    long_df
):

    step(
        "OBJECTIVE 2 — BBRI (REWORKED)"
    )

    # --------------------------------------------------------
    # Dimension 1: Shortage severity
    # --------------------------------------------------------
    # Stock 0..5 is treated as an ordered shortage state.
    # 0 units = maximum severity (100)
    # 5 units = minimum shortage severity within the low-stock band
    # >5 units = no shortage-severity contribution
    #
    # This combines stockout + low-stock into ONE dimension,
    # avoiding double counting them as separate BBRI components.
    # --------------------------------------------------------

    def shortage_severity_score(series):

        values = (
            pd.to_numeric(
                series,
                errors="coerce"
            )
            .fillna(0)
            .clip(lower=0)
        )

        severity = np.where(

            values <= LOW_STOCK_MAX,

            (
                LOW_STOCK_MAX
                +
                1
                -
                values
            )
            /
            (
                LOW_STOCK_MAX
                +
                1
            ),

            0.0

        )

        return float(
            np.mean(
                severity
            )
            *
            100
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

            shortage_severity_score=(
                "stock",
                shortage_severity_score
            )

        )
        .reset_index()
    )

    # --------------------------------------------------------
    # Dimension 2: inventory volatility
    # --------------------------------------------------------
    # Calculate CV separately for each blood group first, then
    # average the group-level volatility. This prevents a single
    # high-volume blood group from dominating the bank-level score.
    # The bounded transform maps CV to 0..100 without using
    # cross-sectional percentile ranks.
    # --------------------------------------------------------

    group_volatility = (
        long_df
        .groupby(
            [
                "hospital_code",
                "blood_group"
            ]
        )
        .agg(

            mean_stock=(
                "stock",
                "mean"
            ),

            std_stock=(
                "stock",
                lambda x:
                x.std(
                    ddof=0
                )
            )

        )
        .reset_index()
    )

    group_volatility[
        "group_cv"
    ] = np.where(

        group_volatility[
            "mean_stock"
        ] > 0,

        group_volatility[
            "std_stock"
        ]
        /
        group_volatility[
            "mean_stock"
        ],

        0.0

    )

    group_volatility[
        "group_volatility_score"
    ] = (
        100
        *
        group_volatility[
            "group_cv"
        ]
        /
        (
            1
            +
            group_volatility[
                "group_cv"
            ]
        )
    )

    volatility = (
        group_volatility
        .groupby(
            "hospital_code"
        )
        .agg(

            stock_volatility=(
                "group_cv",
                "mean"
            ),

            volatility_score=(
                "group_volatility_score",
                "mean"
            )

        )
        .reset_index()
    )

    metrics = metrics.merge(
        volatility,
        on="hospital_code",
        how="left"
    )

    # --------------------------------------------------------
    # Dimension 3: recovery weakness
    # --------------------------------------------------------
    # Recovery is based on stockout EPISODES rather than counting
    # every consecutive zero as a separate event.
    # --------------------------------------------------------

    recovery = bbri_recovery_metrics(
        long_df
    )

    metrics = metrics.merge(
        recovery,
        on="hospital_code",
        how="left"
    )

    for column in [
        "stock_volatility",
        "volatility_score",
        "recovery_rate",
        "mean_recovery_snapshots",
        "median_recovery_snapshots",
        "stockout_episodes",
        "recovered_episodes",
        "unresolved_stockout_episodes"
    ]:

        if column not in metrics.columns:

            metrics[
                column
            ] = np.nan

    metrics[
        "stock_volatility"
    ] = metrics[
        "stock_volatility"
    ].fillna(0)

    metrics[
        "volatility_score"
    ] = metrics[
        "volatility_score"
    ].fillna(0)

    metrics[
        "recovery_rate"
    ] = metrics[
        "recovery_rate"
    ].fillna(100)

    metrics[
        "mean_recovery_snapshots"
    ] = metrics[
        "mean_recovery_snapshots"
    ].fillna(0)

    metrics[
        "median_recovery_snapshots"
    ] = metrics[
        "median_recovery_snapshots"
    ].fillna(0)

    metrics[
        "stockout_episodes"
    ] = metrics[
        "stockout_episodes"
    ].fillna(0)

    metrics[
        "recovered_episodes"
    ] = metrics[
        "recovered_episodes"
    ].fillna(0)

    metrics[
        "unresolved_stockout_episodes"
    ] = metrics[
        "unresolved_stockout_episodes"
    ].fillna(0)

    metrics[
        "recovery_weakness_score"
    ] = (
        100
        -
        metrics[
            "recovery_rate"
        ]
    ).clip(
        lower=0,
        upper=100
    )

    # --------------------------------------------------------
    # Corrected BBRI aggregation
    # --------------------------------------------------------
    # Three conceptually distinct dimensions receive equal weights:
    #   1. shortage severity
    #   2. volatility
    #   3. recovery weakness
    #
    # Equal weighting is used because no empirically validated
    # clinical/operational weights are available in the source data.
    # The design avoids separately weighting stockout and low-stock,
    # which would double-count the same shortage dimension.
    # --------------------------------------------------------

    metrics[
        "stockout_score"
    ] = (
        metrics[
            "stockout_frequency"
        ]
        *
        100
    )

    metrics[
        "low_stock_score"
    ] = (
        metrics[
            "low_stock_frequency"
        ]
        *
        100
    )

    metrics[
        "recovery_score"
    ] = metrics[
        "recovery_weakness_score"
    ]

    metrics[
        "BBRI"
    ] = (

        metrics[
            "shortage_severity_score"
        ]
        +
        metrics[
            "volatility_score"
        ]
        +
        metrics[
            "recovery_weakness_score"
        ]

    ) / 3.0

    metrics[
        "risk_class"
    ] = np.select(

        [

            metrics["BBRI"] < 25,

            metrics["BBRI"] < 50,

            metrics["BBRI"] < 75

        ],

        [

            "Low Risk",
            "Moderate Risk",
            "High Risk"

        ],

        default="Very High Risk"

    )

    output_columns = [

        "hospital_code",
        "blood_bank_name",
        "district",
        "city",
        "area",
        "observations",
        "stockout_frequency",
        "low_stock_frequency",
        "stockout_score",
        "low_stock_score",
        "shortage_severity_score",
        "stock_volatility",
        "volatility_score",
        "stockout_episodes",
        "recovered_episodes",
        "unresolved_stockout_episodes",
        "recovery_rate",
        "mean_recovery_snapshots",
        "median_recovery_snapshots",
        "recovery_weakness_score",
        "recovery_score",
        "BBRI",
        "risk_class"

    ]

    metrics = metrics[
        output_columns
    ].copy()

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

    print(
        "BBRI methodology:",
        "equal-weight shortage severity + volatility + recovery weakness"
    )

    return metrics


# ============================================================
# OBJECTIVE 3
# Corrected District Blood Availability Risk Index (DBARI)
# ============================================================

def objective3(
    bbri,
    long_df
):

    step(
        "OBJECTIVE 3 — DBARI (REWORKED)"
    )

    # --------------------------------------------------------
    # District-level aggregation of the corrected bank-level BBRI.
    #
    # DBARI is intentionally NOT built by adding stockout and
    # low-stock components again. Those variables are already
    # represented inside BBRI's shortage-severity dimension.
    # Re-adding them would double-count shortage information.
    # --------------------------------------------------------

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

            BBRI_std=(
                "BBRI",
                lambda x:
                x.std(
                    ddof=0
                )
            ),

            blood_banks=(
                "hospital_code",
                "nunique"
            ),

            high_risk_banks=(
                "BBRI",
                lambda x:
                int(
                    (
                        x >= 50
                    ).sum()
                )
            ),

            very_high_risk_banks=(
                "BBRI",
                lambda x:
                int(
                    (
                        x >= 75
                    ).sum()
                )
            )

        )
        .reset_index()
    )

    district[
        "high_risk_bank_share"
    ] = np.where(

        district[
            "blood_banks"
        ] > 0,

        district[
            "high_risk_banks"
        ]
        /
        district[
            "blood_banks"
        ]
        *
        100,

        0

    )

    district[
        "very_high_risk_bank_share"
    ] = np.where(

        district[
            "blood_banks"
        ] > 0,

        district[
            "very_high_risk_banks"
        ]
        /
        district[
            "blood_banks"
        ]
        *
        100,

        0

    )

    # --------------------------------------------------------
    # Descriptive district stock indicators.
    # These are retained for interpretation on the website,
    # but they are NOT added again to DBARI.
    # --------------------------------------------------------

    stock = (
        long_df
        .groupby(
            "district"
        )
        .agg(

            stockout_percent=(
                "stock",
                lambda x:
                (x == 0).mean()
                * 100
            ),

            low_stock_percent=(
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

    # --------------------------------------------------------
    # Final DBARI
    # --------------------------------------------------------
    # The district score is the equal-weighted bank-level risk
    # average. This makes DBARI a hierarchical aggregation of BBRI
    # rather than a second composite that double-counts the same
    # stockout/low-stock inputs.
    # --------------------------------------------------------

    result[
        "DBARI"
    ] = result[
        "average_BBRI"
    ].clip(
        lower=0,
        upper=100
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

            "Low Risk",
            "Moderate Risk",
            "High Risk"

        ],

        default="Very High Risk"

    )

    output_columns = [

        "district",
        "blood_banks",
        "average_BBRI",
        "median_BBRI",
        "highest_BBRI",
        "BBRI_std",
        "high_risk_banks",
        "high_risk_bank_share",
        "very_high_risk_banks",
        "very_high_risk_bank_share",
        "stockout_percent",
        "low_stock_percent",
        "DBARI",
        "risk_class"

    ]

    result = result[
        output_columns
    ].copy()

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

    print(
        "DBARI methodology:",
        "district mean of corrected bank-level BBRI; shortage indicators retained descriptively"
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
        print("No latest-stock data available for redistribution.")
        return pd.DataFrame()

    if not SCIPY_AVAILABLE:
        print("scipy unavailable. Install it with: pip install scipy")
        return pd.DataFrame()

    # ------------------------------------------------------------
    # Use coordinate-valid, non-anomalous banks only.
    # The optimization is same-blood-group, safety-floor based,
    # and constrained to the selected 150 km operational radius.
    # ------------------------------------------------------------
    coord_cols = [
        "hospital_code",
        "latitude",
        "longitude"
    ]

    if geo.empty or not all(
        column in geo.columns
        for column in coord_cols
    ):
        print("Geographic coordinate data unavailable for redistribution.")
        return pd.DataFrame()

    coordinates = (
        geo[coord_cols]
        .copy()
        .drop_duplicates("hospital_code")
    )

    coordinates["hospital_code"] = coordinates["hospital_code"].map(
        normalize_code
    )

    working = latest.drop(
        columns=[
            column
            for column in ["latitude", "longitude"]
            if column in latest.columns
        ],
        errors="ignore"
    ).copy()

    working["hospital_code"] = working["hospital_code"].map(
        normalize_code
    )

    working = working.merge(
        coordinates,
        on="hospital_code",
        how="inner"
    )

    working = working[
        ~working[
            "hospital_code"
        ].isin(BAD_COORDINATE_CODES)
    ].copy()

    working["latitude"] = pd.to_numeric(
        working["latitude"],
        errors="coerce"
    )
    working["longitude"] = pd.to_numeric(
        working["longitude"],
        errors="coerce"
    )

    working = working[
        working["latitude"].notna()
        & working["longitude"].notna()
        & (working["latitude"] != 0)
        & (working["longitude"] != 0)
    ].copy()

    recommendations = []
    safety_floor = float(SAFETY_STOCK)
    shortage_penalty = 10000.0

    for group in BLOOD_GROUPS:

        group_df = working[
            [
                "hospital_code",
                "blood_bank_name",
                "district",
                "city",
                "latitude",
                "longitude",
                group
            ]
        ].copy()

        group_df["stock"] = pd.to_numeric(
            group_df[group],
            errors="coerce"
        ).fillna(0).clip(lower=0)

        group_df["surplus"] = (
            group_df["stock"] - safety_floor
        ).clip(lower=0)

        group_df["deficit"] = (
            safety_floor - group_df["stock"]
        ).clip(lower=0)

        donors = group_df[
            group_df["surplus"] > 0
        ].reset_index(drop=True)

        recipients = group_df[
            group_df["deficit"] > 0
        ].reset_index(drop=True)

        if donors.empty or recipients.empty:
            continue

        # Candidate transfer variables: donor -> recipient.
        pairs = []
        for donor_idx, donor in donors.iterrows():
            for recipient_idx, recipient in recipients.iterrows():

                distance = float(
                    haversine_km(
                        float(donor["latitude"]),
                        float(donor["longitude"]),
                        float(recipient["latitude"]),
                        float(recipient["longitude"])
                    )
                )

                # Same location is intentionally excluded from an
                # actionable transfer: it is not a meaningful physical
                # redistribution route for this objective.
                if distance <= 0:
                    continue

                if distance > REDISTRIBUTION_RADIUS_KM:
                    continue

                pairs.append({
                    "donor_idx": donor_idx,
                    "recipient_idx": recipient_idx,
                    "distance_km": distance
                })

        if not pairs:
            continue

        # Variables = every feasible transfer + one unmet-deficit
        # variable per recipient.
        transfer_count = len(pairs)
        unmet_offset = transfer_count
        variable_count = transfer_count + len(recipients)

        objective = np.zeros(variable_count, dtype=float)

        for i, pair in enumerate(pairs):
            objective[i] = pair["distance_km"]

        objective[unmet_offset:] = shortage_penalty

        # Donor capacity constraints: sum outgoing <= surplus.
        donor_constraints = []
        donor_rhs = []

        for donor_idx, donor in donors.iterrows():
            row = np.zeros(variable_count, dtype=float)
            for i, pair in enumerate(pairs):
                if pair["donor_idx"] == donor_idx:
                    row[i] = 1.0
            donor_constraints.append(row)
            donor_rhs.append(float(donor["surplus"]))

        # Recipient balance equations:
        # incoming transfer + unmet deficit = full deficit.
        recipient_constraints = []
        recipient_rhs = []

        for recipient_idx, recipient in recipients.iterrows():
            row = np.zeros(variable_count, dtype=float)
            for i, pair in enumerate(pairs):
                if pair["recipient_idx"] == recipient_idx:
                    row[i] = 1.0
            row[unmet_offset + recipient_idx] = 1.0
            recipient_constraints.append(row)
            recipient_rhs.append(float(recipient["deficit"]))

        A_ub = np.vstack(donor_constraints) if donor_constraints else None
        b_ub = np.array(donor_rhs, dtype=float) if donor_rhs else None
        A_eq = np.vstack(recipient_constraints) if recipient_constraints else None
        b_eq = np.array(recipient_rhs, dtype=float) if recipient_rhs else None

        bounds = [
            (0, None)
            for _ in range(variable_count)
        ]

        solved = linprog(
            c=objective,
            A_ub=A_ub,
            b_ub=b_ub,
            A_eq=A_eq,
            b_eq=b_eq,
            bounds=bounds,
            method="highs"
        )

        if not solved.success:
            print(
                f"Objective 10: optimization failed for {group}: "
                f"{solved.message}"
            )
            continue

        x = np.maximum(
            solved.x,
            0
        )

        for i, pair in enumerate(pairs):
            units = float(x[i])

            if units <= 1e-9:
                continue

            units_int = int(np.floor(units + 1e-9))
            if units_int <= 0:
                continue

            donor = donors.iloc[
                pair["donor_idx"]
            ]
            recipient = recipients.iloc[
                pair["recipient_idx"]
            ]

            recommendations.append({
                "blood_group": group,
                "donor_code": normalize_code(
                    donor["hospital_code"]
                ),
                "donor_name": donor["blood_bank_name"],
                "donor_district": donor["district"],
                "recipient_code": normalize_code(
                    recipient["hospital_code"]
                ),
                "recipient_name": recipient["blood_bank_name"],
                "recipient_district": recipient["district"],
                "distance_km": round(
                    pair["distance_km"],
                    6
                ),
                "optimized_units": units_int,
                "transport_km_units": round(
                    pair["distance_km"] * units_int,
                    6
                )
            })

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

    result = result.sort_values(
        [
            "blood_group",
            "distance_km",
            "donor_code",
            "recipient_code"
        ]
    ).reset_index(drop=True)

    output_folder = OUTPUT_DIR / "objective10"

    result.to_csv(
        output_folder /
        "objective10_optimized_redistribution.csv",
        index=False
    )

    result.to_csv(
        output_folder /
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
                result["optimized_units"].sum()
                if not result.empty
                else 0
            ),
            (
                result["distance_km"].mean()
                if not result.empty
                else 0
            ),
            (
                result["distance_km"].max()
                if not result.empty
                else 0
            ),
            (
                (result["distance_km"] == 0).sum()
                if not result.empty
                else 0
            )
        ]
    })

    summary.to_csv(
        output_folder /
        "objective10_redistribution_summary.csv",
        index=False
    )

    result[
        result["distance_km"] == 0
    ].to_csv(
        output_folder /
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
            result["optimized_units"].sum()
        ) if not result.empty else 0
    )

    print(
        "Average redistribution distance km:",
        round(
            float(
                result["distance_km"].mean()
            ),
            2
        ) if not result.empty else 0
    )

    print(
        "Maximum redistribution distance km:",
        round(
            float(
                result["distance_km"].max()
            ),
            2
        ) if not result.empty else 0
    )

    return result


# ============================================================
# OBJECTIVE 11
# ============================================================

def objective11(
    latest,
    long_df,
    bbri,
    geo,
    model_result
):

    step(
        "OBJECTIVE 11 — Alternatives"
    )

    if latest.empty:
        print("No latest-stock data available for Objective 11.")
        return pd.DataFrame()

    # ------------------------------------------------------------
    # Coordinate-valid population. Keep geography authoritative.
    # ------------------------------------------------------------
    if geo.empty:
        print("Geographic data unavailable for Objective 11.")
        return pd.DataFrame()

    geo_coords = geo[
        [
            "hospital_code",
            "latitude",
            "longitude"
        ]
    ].copy().drop_duplicates(
        "hospital_code"
    )

    geo_coords["hospital_code"] = geo_coords["hospital_code"].map(
        normalize_code
    )

    current = latest.drop(
        columns=[
            column
            for column in [
                "latitude",
                "longitude"
            ]
            if column in latest.columns
        ],
        errors="ignore"
    ).copy()

    current["hospital_code"] = current["hospital_code"].map(
        normalize_code
    )

    bbri_merge = bbri[
        [
            "hospital_code",
            "BBRI"
        ]
    ].copy().drop_duplicates("hospital_code")

    bbri_merge["hospital_code"] = bbri_merge["hospital_code"].map(
        normalize_code
    )

    current = current.merge(
        bbri_merge,
        on="hospital_code",
        how="left"
    )

    current = current.merge(
        geo_coords,
        on="hospital_code",
        how="inner"
    )

    current = current[
        ~current[
            "hospital_code"
        ].isin(BAD_COORDINATE_CODES)
    ].copy()

    current["latitude"] = pd.to_numeric(
        current["latitude"],
        errors="coerce"
    )
    current["longitude"] = pd.to_numeric(
        current["longitude"],
        errors="coerce"
    )

    current = current[
        current["latitude"].notna()
        & current["longitude"].notna()
        & (current["latitude"] != 0)
        & (current["longitude"] != 0)
    ].copy()

    # ------------------------------------------------------------
    # Build a true current-snapshot HGB stockout probability.
    # We use the same rolling/previous-stock feature recipe as Obj 6,
    # but keep the latest observation (which prepare_model_dataset
    # intentionally removes because it has no future target).
    # ------------------------------------------------------------
    probability_lookup = {}

    if (
        model_result is not None
        and model_result.get("hgb") is not None
        and not long_df.empty
    ):
        prediction_base = (
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

        grouped = prediction_base.groupby(
            [
                "hospital_code",
                "blood_group"
            ]
        )

        prediction_base["previous_stock"] = grouped["stock"].shift(1)
        prediction_base["stock_change"] = (
            prediction_base["stock"]
            - prediction_base["previous_stock"]
        )
        prediction_base["rolling_mean_3"] = grouped["stock"].transform(
            lambda x: x.rolling(3, min_periods=1).mean()
        )
        prediction_base["rolling_std_3"] = grouped["stock"].transform(
            lambda x: x.rolling(3, min_periods=1).std()
        ).fillna(0)
        prediction_base["previous_stockout"] = (
            prediction_base["previous_stock"]
            .fillna(-1)
            .eq(0)
            .astype(int)
        )
        prediction_base["previous_low_stock"] = (
            prediction_base["previous_stock"]
            .fillna(-1)
            .between(1, LOW_STOCK_MAX)
            .astype(int)
        )
        prediction_base["stock_vs_rolling_mean"] = (
            prediction_base["stock"]
            - prediction_base["rolling_mean_3"]
        )
        prediction_base["trend"] = (
            prediction_base["stock_change"]
            .fillna(0)
        )
        prediction_base["recovered_from_previous_stockout"] = (
            (prediction_base["previous_stockout"] == 1)
            & (prediction_base["stock"] > 0)
        ).astype(int)

        latest_feature_rows = (
            prediction_base
            .sort_values("snapshot_date")
            .groupby(
                [
                    "hospital_code",
                    "blood_group"
                ],
                as_index=False
            )
            .tail(1)
            .copy()
        )

        latest_feature_rows[MODEL_FEATURES] = (
            latest_feature_rows[MODEL_FEATURES]
            .replace(
                [
                    np.inf,
                    -np.inf
                ],
                np.nan
            )
            .fillna(0)
        )

        try:
            probabilities = model_result[
                "hgb"
            ].predict_proba(
                latest_feature_rows[MODEL_FEATURES].astype(float)
            )[:, 1]

            for row, probability in zip(
                latest_feature_rows.itertuples(index=False),
                probabilities
            ):
                probability_lookup[
                    (
                        normalize_code(row.hospital_code),
                        str(row.blood_group)
                    )
                ] = float(probability)

        except Exception as exc:
            print(
                "Objective 11 prediction warning:",
                exc
            )

    # Score weights validated in the saved Objective 11 artifact:
    # 35% availability + 25% predicted safety + 20% lower BBRI
    # + 20% proximity.
    weight_availability = 0.35
    weight_safety = 0.25
    weight_bbri = 0.20
    weight_proximity = 0.20

    # Precompute vectorized alternative-bank arrays once.
    codes = current["hospital_code"].astype(str).to_numpy()
    names = current["blood_bank_name"].astype(str).to_numpy()
    districts = current["district"].astype(str).to_numpy()
    latitudes = current["latitude"].astype(float).to_numpy()
    longitudes = current["longitude"].astype(float).to_numpy()
    bbri_values = pd.to_numeric(
        current["BBRI"],
        errors="coerce"
    ).fillna(50).to_numpy(dtype=float)

    probability_by_key = probability_lookup
    candidate_frames = []

    for origin_idx, origin in current.reset_index(drop=True).iterrows():

        origin_code = str(
            normalize_code(
                origin["hospital_code"]
            )
        )

        origin_lat = float(origin["latitude"])
        origin_lon = float(origin["longitude"])

        for group in BLOOD_GROUPS:

            origin_stock = float(
                pd.to_numeric(
                    origin[group],
                    errors="coerce"
                )
                if pd.notna(origin[group])
                else 0
            )

            if origin_stock > SAFETY_STOCK:
                continue

            alt_stock = pd.to_numeric(
                current[group],
                errors="coerce"
            ).fillna(0).to_numpy(dtype=float)

            base_mask = (
                (codes != origin_code)
                & (alt_stock > 0)
            )

            if not np.any(base_mask):
                continue

            candidate_idx = np.where(base_mask)[0]

            distances = haversine_km(
                origin_lat,
                origin_lon,
                latitudes[candidate_idx],
                longitudes[candidate_idx]
            )
            distances = np.asarray(
                distances,
                dtype=float
            )

            keep = (
                (distances > 0)
                & (distances <= REDISTRIBUTION_RADIUS_KM)
            )

            if not np.any(keep):
                continue

            candidate_idx = candidate_idx[keep]
            distances = distances[keep]
            candidate_stock = alt_stock[candidate_idx]
            candidate_bbri = bbri_values[candidate_idx]

            max_stock = max(
                float(alt_stock.max()),
                1.0
            )

            availability_score = (
                candidate_stock
                / max_stock
                * 100.0
            )

            probabilities = np.array(
                [
                    probability_by_key.get(
                        (codes[idx], group),
                        0.5
                    )
                    for idx in candidate_idx
                ],
                dtype=float
            )
            probabilities = np.clip(
                probabilities,
                0,
                1
            )

            safety_score = (
                1.0 - probabilities
            ) * 100.0

            bbri_score = np.clip(
                100.0 - candidate_bbri,
                0.0,
                100.0
            )

            proximity_score = (
                1.0
                - distances / REDISTRIBUTION_RADIUS_KM
            ) * 100.0

            decision_score = (
                weight_availability * availability_score
                + weight_safety * safety_score
                + weight_bbri * bbri_score
                + weight_proximity * proximity_score
            )

            frame = pd.DataFrame({
                "origin_code": origin_code,
                "origin_bank": str(origin["blood_bank_name"]),
                "origin_district": str(origin["district"]),
                "blood_group": group,
                "origin_stock": origin_stock,
                "alternative_code": codes[candidate_idx],
                "alternative_bank": names[candidate_idx],
                "alternative_district": districts[candidate_idx],
                "alternative_stock": candidate_stock,
                "predicted_stockout_probability": probabilities,
                "alternative_bbri": candidate_bbri,
                "distance_km": np.round(
                    distances,
                    6
                ),
                "decision_support_score": np.round(
                    decision_score,
                    6
                )
            })

            candidate_frames.append(frame)

    if candidate_frames:
        ranking = pd.concat(
            candidate_frames,
            ignore_index=True
        )
    else:
        ranking = pd.DataFrame(
            columns=[
                "origin_code",
                "origin_bank",
                "origin_district",
                "blood_group",
                "origin_stock",
                "alternative_code",
                "alternative_bank",
                "alternative_district",
                "alternative_stock",
                "predicted_stockout_probability",
                "alternative_bbri",
                "distance_km",
                "decision_support_score"
            ]
        )

    if ranking.empty:
        ranking = pd.DataFrame(
            columns=[
                "origin_code",
                "origin_bank",
                "origin_district",
                "blood_group",
                "origin_stock",
                "alternative_code",
                "alternative_bank",
                "alternative_district",
                "alternative_stock",
                "predicted_stockout_probability",
                "alternative_bbri",
                "distance_km",
                "decision_support_score",
                "rank"
            ]
        )
    else:
        ranking = ranking.sort_values(
            [
                "origin_code",
                "blood_group",
                "decision_support_score",
                "alternative_stock",
                "distance_km"
            ],
            ascending=[
                True,
                True,
                False,
                False,
                True
            ]
        ).reset_index(drop=True)

        ranking["rank"] = (
            ranking
            .groupby(
                [
                    "origin_code",
                    "blood_group"
                ]
            )
            .cumcount()
            + 1
        )

    output_folder = OUTPUT_DIR / "objective11"

    # Full candidate ranking — primary website/research file.
    ranking.to_csv(
        output_folder /
        "Maharashtra_Alternative_Blood_Bank_Ranking.csv",
        index=False
    )

    # Top-5 recommendations per origin/blood group.
    top5 = ranking[
        ranking["rank"] <= 5
    ].copy()

    top5.to_csv(
        output_folder /
        "Maharashtra_Alternative_Blood_Bank_Top5.csv",
        index=False
    )

    origin_cases = 0
    cases_with_alt = 0

    if not current.empty:
        for _, origin in current.iterrows():
            for group in BLOOD_GROUPS:
                stock = float(
                    pd.to_numeric(
                        origin[group],
                        errors="coerce"
                    )
                    if pd.notna(origin[group])
                    else 0
                )
                if stock <= SAFETY_STOCK:
                    origin_cases += 1
                    origin_code = normalize_code(
                        origin["hospital_code"]
                    )
                    has_alt = (
                        not ranking.empty
                        and
                        (
                            (
                                ranking["origin_code"].astype(str)
                                == str(origin_code)
                            )
                            &
                            (
                                ranking["blood_group"].astype(str)
                                == group
                            )
                        ).any()
                    )
                    if has_alt:
                        cases_with_alt += 1

    coverage = (
        cases_with_alt / origin_cases
        if origin_cases
        else 0
    )

    summary = pd.DataFrame({
        "Metric": [
            "Latest snapshot",
            "Banks in ranking network",
            "Operational radius",
            "Origin bank-group shortage cases (<=5 units)",
            "Cases with at least one ranked alternative",
            "Alternative coverage rate",
            "All candidate rows within 150 km",
            "Top-5 recommendation rows",
            "Score"
        ],
        "Value": [
            str(
                latest["snapshot_date"].max()
            ) if "snapshot_date" in latest.columns else "",
            int(
                current["hospital_code"].nunique()
            ),
            int(
                REDISTRIBUTION_RADIUS_KM
            ),
            int(origin_cases),
            int(cases_with_alt),
            float(coverage),
            int(len(ranking)),
            int(len(top5)),
            "35% availability + 25% predicted safety + 20% lower BBRI + 20% proximity"
        ]
    })

    summary.to_csv(
        output_folder /
        "Maharashtra_Alternative_Ranking_Summary.csv",
        index=False
    )

    print(
        "Objective 11 rows:",
        len(ranking)
    )

    print(
        "Objective 11 shortage cases:",
        origin_cases
    )

    print(
        "Objective 11 cases with alternatives:",
        cases_with_alt
    )

    print(
        "Objective 11 coverage:",
        f"{coverage * 100:.2f}%"
    )

    return ranking


# ============================================================
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
    """Write one snapshot processing event to SQLite safely."""
    conn = None
    try:
        conn = sqlite3.connect(DATABASE)
        conn.execute(
            """
            INSERT INTO update_log (
                run_time, snapshot_date, source_file, source_rows,
                valid_rows, new_rows, status, message
            )
            VALUES (datetime('now'), ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                str(snapshot_date or ""),
                str(source_file or ""),
                int(source_rows or 0),
                int(valid_rows or 0),
                int(new_rows or 0),
                str(status or ""),
                str(message or "")
            ]
        )
        conn.commit()
        return True
    except Exception as exc:
        print("WARNING: update log write failed:", exc)
        return False
    finally:
        if conn is not None:
            conn.close()


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

            write_update_log(

                "",

                path.name,

                0,

                0,

                0,

                "FAILED",

                str(exc)

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

        long_df,

        bbri,

        geo,

        model_result

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