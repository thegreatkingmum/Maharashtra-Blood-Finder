# ============================================================
# MAHARASHTRA BLOOD FINDER
# Main Flask Application
# ============================================================

from flask import Flask, render_template, request, jsonify
from pathlib import Path
import sqlite3

import pandas as pd
import numpy as np
import joblib

from sklearn.ensemble import HistGradientBoostingClassifier


# ============================================================
# APPLICATION
# ============================================================

app = Flask(__name__)

BASE_DIR = Path(__file__).resolve().parent

DATABASE = BASE_DIR / "database.db"

OUTPUT_DIR = (
    BASE_DIR
    / "data"
    / "outputs"
)

MODELS_DIR = (
    BASE_DIR
    / "models"
)


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
# LIVE MODEL CACHE
# ============================================================

LIVE_MODEL = None
LIVE_MODEL_SOURCE = None
LIVE_MODEL_SNAPSHOT_COUNT = None


# ============================================================
# DATABASE
# ============================================================

def get_db_connection():

    conn = sqlite3.connect(
        DATABASE
    )

    conn.row_factory = sqlite3.Row

    return conn


# ============================================================
# CSV LOADER
# ============================================================

def load_csv(relative_path):

    path = OUTPUT_DIR / relative_path

    if not path.exists():

        return pd.DataFrame()

    try:

        return pd.read_csv(
            path
        )

    except Exception as exc:

        print(
            f"Could not load {path}: {exc}"
        )

        return pd.DataFrame()


# ============================================================
# COLUMN FINDER
# ============================================================

def find_column(
    df,
    candidates
):

    if df.empty:

        return None

    mapping = {
        str(column).strip().lower(): column
        for column in df.columns
    }

    for candidate in candidates:

        key = (
            str(candidate)
            .strip()
            .lower()
        )

        if key in mapping:

            return mapping[key]

    return None


# ============================================================
# SAFE JSON CONVERSION
# ============================================================

def clean_nan_records(df):

    if df.empty:

        return []

    clean = df.copy()

    clean = clean.replace(
        {
            np.nan: None
        }
    )

    return clean.to_dict(
        orient="records"
    )


# ============================================================
# NORMALIZE COLUMNS
# ============================================================

def normalize_columns(df):

    if df.empty:

        return df

    rename_map = {}

    aliases = {

        "hospital code":
            "hospital_code",

        "hospitalcode":
            "hospital_code",

        "blood bank name":
            "blood_bank_name",

        "bank name":
            "blood_bank_name",

        "lat":
            "latitude",

        "lon":
            "longitude",

        "lng":
            "longitude",

        "bbri":
            "BBRI",

        "dbari":
            "DBARI",

        "gvs":
            "GVS",

        "geographic vulnerability score":
            "GVS",

        "geographic vulnerability category":
            "geographic_vulnerability",

        "risk class":
            "risk_class",

        "risk category":
            "risk_category",

        "betweenness centrality":
            "betweenness_centrality",

        "closeness centrality":
            "closeness_centrality",

        "structural criticality score":
            "network_criticality",

        "donor code":
            "donor_code",

        "donor bank":
            "donor_name",

        "donor district":
            "donor_district",

        "recipient code":
            "recipient_code",

        "recipient bank":
            "recipient_name",

        "recipient district":
            "recipient_district",

        "distance km":
            "distance_km",

        "transfer units":
            "optimized_units",

        "recommended units":
            "optimized_units",

        "blood group":
            "blood_group",

        "origin code":
            "origin_code",

        "origin bank":
            "origin_bank",

        "origin district":
            "origin_district",

        "origin stock":
            "origin_stock",

        "alternative code":
            "alternative_code",

        "alternative bank":
            "alternative_bank",

        "alternative district":
            "alternative_district",

        "alternative stock":
            "alternative_stock",

        "predicted stockout probability":
            "predicted_stockout_probability",

        "alternative bbri":
            "alternative_bbri",

        "decision support score":
            "decision_support_score"

    }

    for column in df.columns:

        normalized = (
            str(column)
            .strip()
            .lower()
        )

        if normalized in aliases:

            rename_map[column] = (
                aliases[normalized]
            )

    return df.rename(
        columns=rename_map
    )


# ============================================================
# OBJECTIVE FILE DISCOVERY
# ============================================================

def discover_csv(
    objective_folder,
    required_columns
):

    folder = (
        OUTPUT_DIR
        / objective_folder
    )

    if not folder.exists():

        return pd.DataFrame()

    for path in sorted(
        folder.glob("*.csv")
    ):

        try:

            df = pd.read_csv(
                path
            )

            df = normalize_columns(
                df
            )

            ok = True

            for column in required_columns:

                if find_column(
                    df,
                    [column]
                ) is None:

                    ok = False
                    break

            if ok:

                print(
                    f"Discovered "
                    f"{objective_folder}: "
                    f"{path.name}"
                )

                return df

        except Exception as exc:

            print(
                f"Could not inspect "
                f"{path.name}: {exc}"
            )

    return pd.DataFrame()


# ============================================================
# BBRI
# ============================================================

def load_bbri():

    candidate_files = [

        "objective2/"
        "objective2_BBRI_corrected.csv",

        "objective2/"
        "objective2_BBRI.csv",

        "objective2_BBRI_corrected.csv",

        "objective2_BBRI.csv"

    ]

    for relative_path in candidate_files:

        df = load_csv(
            relative_path
        )

        df = normalize_columns(
            df
        )

        if (
            not df.empty
            and
            find_column(
                df,
                ["BBRI"]
            ) is not None
        ):

            return df

    return discover_csv(
        "objective2",
        ["BBRI"]
    )


# ============================================================
# OBJECTIVE 11
# ALTERNATIVE RECOMMENDATION DATA
# ============================================================

def load_objective11():

    candidate_files = [

        "objective11/"
        "Maharashtra_Alternative_Blood_Bank_Ranking.csv",

        "objective11/"
        "objective11_alternative_recommendations.csv",

        "objective11/"
        "objective11_alternative_ranking.csv",

        "objective11/"
        "Maharashtra_Alternative_Ranking.csv",

        "Maharashtra_Alternative_Blood_Bank_Ranking.csv",

        "Maharashtra_Alternative_Ranking.csv"

    ]

    for relative_path in candidate_files:

        df = load_csv(
            relative_path
        )

        df = normalize_columns(
            df
        )

        if (
            not df.empty
            and
            find_column(
                df,
                ["origin_code"]
            ) is not None
            and
            find_column(
                df,
                ["alternative_code"]
            ) is not None
            and
            find_column(
                df,
                ["blood_group"]
            ) is not None
        ):

            print(
                "Objective 11 loaded:",
                relative_path,
                "rows:",
                len(df)
            )

            return df

    discovered = discover_csv(
        "objective11",
        [
            "origin_code",
            "alternative_code",
            "blood_group"
        ]
    )

    if not discovered.empty:

        return discovered

    print(
        "Objective 11 recommendation file not found."
    )

    return pd.DataFrame()


# ============================================================
# PAGE ROUTES
# ============================================================

@app.route("/")
def index():

    return render_template(
        "index.html"
    )


@app.route("/finder")
def finder():

    return render_template(
        "finder.html"
    )


@app.route("/insights")
def insights():

    return render_template(
        "insights.html"
    )


@app.route("/risk")
def risk():

    return render_template(
        "risk.html"
    )


@app.route("/gis")
def gis():

    return render_template(
        "gis.html"
    )


# Temporary compatibility route for existing template links.
# Keep this route for the demo; remove it later after all research links are removed.
@app.route("/research")
def research():

    return render_template(
        "research.html"
    )


# ============================================================
# HEALTH
# ============================================================

@app.route("/api/health")
def health():

    return jsonify({

        "status":
            "ok",

        "application":
            "Maharashtra Blood Finder"

    })


# ============================================================
# LOCATIONS
# ============================================================

@app.route("/api/locations")
def locations():

    conn = get_db_connection()

    try:

        cities = conn.execute(
            """
            SELECT DISTINCT city
            FROM blood_banks
            WHERE city IS NOT NULL
              AND TRIM(city) != ''
            ORDER BY city
            """
        ).fetchall()

        districts = conn.execute(
            """
            SELECT DISTINCT district
            FROM blood_banks
            WHERE district IS NOT NULL
              AND TRIM(district) != ''
            ORDER BY district
            """
        ).fetchall()

        areas = conn.execute(
            """
            SELECT DISTINCT area
            FROM blood_banks
            WHERE area IS NOT NULL
              AND TRIM(area) != ''
            ORDER BY area
            """
        ).fetchall()

        return jsonify({

            "cities": [
                row["city"]
                for row in cities
            ],

            "districts": [
                row["district"]
                for row in districts
            ],

            "areas": [
                row["area"]
                for row in areas
            ]

        })

    finally:

        conn.close()


# ============================================================
# FIND BLOOD
# ============================================================

@app.route(
    "/api/find-blood",
    methods=["GET"]
)
def find_blood():

    area = request.args.get(
        "area",
        ""
    ).strip()

    city = request.args.get(
        "city",
        ""
    ).strip()

    district = request.args.get(
        "district",
        ""
    ).strip()

    blood_group = request.args.get(
        "blood_group",
        "O+"
    ).strip()

    try:

        required_units = int(
            request.args.get(
                "required_units",
                1
            )
        )

    except (
        ValueError,
        TypeError
    ):

        required_units = 1

    if blood_group not in BLOOD_GROUPS:

        return jsonify({

            "success":
                False,

            "message":
                "Invalid blood group."

        }), 400

    if required_units < 1:

        return jsonify({

            "success":
                False,

            "message":
                "Required units must be at least 1."

        }), 400

    conn = get_db_connection()

    try:

        rows = []

        search_level = None

        if area:

            rows = conn.execute(
                """
                SELECT *
                FROM blood_banks
                WHERE LOWER(TRIM(area))
                    = LOWER(TRIM(?))
                """,
                (area,)
            ).fetchall()

            if rows:

                search_level = "area"

        if not rows and city:

            rows = conn.execute(
                """
                SELECT *
                FROM blood_banks
                WHERE LOWER(TRIM(city))
                    = LOWER(TRIM(?))
                """,
                (city,)
            ).fetchall()

            if rows:

                search_level = "city"

        if not rows and district:

            rows = conn.execute(
                """
                SELECT *
                FROM blood_banks
                WHERE LOWER(TRIM(district))
                    = LOWER(TRIM(?))
                """,
                (district,)
            ).fetchall()

            if rows:

                search_level = "district"

        if not rows:

            return jsonify({

                "success":
                    True,

                "search_level":
                    "none",

                "results":
                    []

            })

        results = []

        for row in rows:

            try:

                stock = float(
                    row[blood_group]
                )

            except (
                ValueError,
                TypeError,
                KeyError
            ):

                continue

            if stock < required_units:

                continue

            results.append({

                "hospital_code":
                    row["hospital_code"],

                "blood_bank_name":
                    row["blood_bank_name"],

                "address":
                    row["address"],

                "contact":
                    row["contact"],

                "hospital_type":
                    row["hospital_type"],

                "district":
                    row["district"],

                "area":
                    row["area"],

                "city":
                    row["city"],

                "latitude":
                    row["latitude"],

                "longitude":
                    row["longitude"],

                "stock":
                    stock,

                "blood_group":
                    blood_group

            })

        results.sort(
            key=lambda item:
                item["stock"],
            reverse=True
        )

        return jsonify({

            "success":
                True,

            "search_level":
                search_level,

            "blood_group":
                blood_group,

            "required_units":
                required_units,

            "results":
                results

        })

    finally:

        conn.close()


# ============================================================
# GENERIC CSV API
# ============================================================

def objective_csv_api(
    relative_path,
    message
):

    df = load_csv(
        relative_path
    )

    df = normalize_columns(
        df
    )

    if df.empty:

        return jsonify({

            "success":
                False,

            "message":
                message

        })

    return jsonify({

        "success":
            True,

        "columns":
            list(df.columns),

        "data":
            clean_nan_records(df)

    })


# ============================================================
# INSIGHTS
# ============================================================

@app.route(
    "/api/insights/overview"
)
def insights_overview():

    conn = get_db_connection()

    try:

        bank_count = conn.execute(
            """
            SELECT COUNT(*)
            FROM blood_banks
            """
        ).fetchone()[0]

        district_count = conn.execute(
            """
            SELECT COUNT(
                DISTINCT district
            )
            FROM blood_banks
            """
        ).fetchone()[0]

        snapshot_count = conn.execute(
            """
            SELECT COUNT(
                DISTINCT snapshot_date
            )
            FROM stock_history
            """
        ).fetchone()[0]

        latest_snapshot = conn.execute(
            """
            SELECT MAX(snapshot_date)
            FROM stock_history
            """
        ).fetchone()[0]

    finally:

        conn.close()

    return jsonify({

        "current_banks":
            bank_count,

        "districts":
            district_count,

        "snapshots":
            snapshot_count,

        "latest_snapshot":
            latest_snapshot

    })


@app.route(
    "/api/insights/availability"
)
def insights_availability():

    return objective_csv_api(

        "objective1/"
        "objective1_maharashtra_availability.csv",

        "Objective 1 data not found."

    )


@app.route(
    "/api/insights/district-stockout"
)
def insights_district_stockout():

    return objective_csv_api(

        "objective1/"
        "objective1_district_availability.csv",

        "District data not found."

    )


@app.route(
    "/api/insights/hotspots"
)
def insights_hotspots():

    return objective_csv_api(

        "objective4/"
        "objective4_district_bloodgroup_hotspots.csv",

        "Objective 4 data not found."

    )


@app.route(
    "/api/insights/blood-group-comparison"
)
def insights_blood_group_comparison():

    return objective_csv_api(

        "objective8/"
        "objective8_blood_group_comparison.csv",

        "Objective 8 data not found."

    )


# ============================================================
# BBRI
# ============================================================

@app.route(
    "/api/risk/bbri"
)
def risk_bbri():

    df = load_bbri()

    if df.empty:

        return jsonify({

            "success":
                False,

            "message":
                "BBRI data not found."

        })

    return jsonify({

        "success":
            True,

        "data":
            clean_nan_records(df)

    })


@app.route(
    "/api/risk/bbri-summary"
)
def bbri_summary():

    df = load_bbri()

    if df.empty:

        return jsonify({

            "success":
                False,

            "message":
                "BBRI data not found."

        })

    bbri_col = find_column(
        df,
        [
            "BBRI"
        ]
    )

    risk_col = find_column(
        df,
        [
            "risk_class",
            "risk_category"
        ]
    )

    values = pd.to_numeric(
        df[bbri_col],
        errors="coerce"
    )

    summary = {

        "banks":
            (
                int(
                    df[
                        "hospital_code"
                    ].nunique()
                )
                if
                "hospital_code"
                in df.columns
                else
                int(
                    values.notna().sum()
                )
            ),

        "average_bbri":
            round(
                float(
                    values.mean()
                ),
                2
            ),

        "median_bbri":
            round(
                float(
                    values.median()
                ),
                2
            ),

        "highest_bbri":
            round(
                float(
                    values.max()
                ),
                2
            )

    }

    if risk_col:

        classes = (
            df[risk_col]
            .astype(str)
            .str.lower()
        )

        summary[
            "high_risk_count"
        ] = int(
            classes
            .str.contains(
                "high"
            )
            .sum()
        )

    else:

        summary[
            "high_risk_count"
        ] = 0

    return jsonify({

        "success":
            True,

        "summary":
            summary,

        **summary

    })


# ============================================================
# RISK BANKS
# ============================================================

@app.route(
    "/api/risk/banks"
)
def risk_banks():

    conn = get_db_connection()

    try:

        rows = conn.execute(
            """
            SELECT
                hospital_code,
                blood_bank_name,
                district,
                city
            FROM blood_banks
            ORDER BY
                district,
                city,
                blood_bank_name
            """
        ).fetchall()

        result = [
            dict(row)
            for row in rows
        ]

        return jsonify({

            "success":
                True,

            "banks":
                result,

            "data":
                result

        })

    finally:

        conn.close()


# ============================================================
# MODEL COMPARISON
# ============================================================

@app.route(
    "/api/risk/model-comparison"
)
def risk_model_comparison():

    return objective_csv_api(

        "objective6/"
        "objective6_model_comparison.csv",

        "Model comparison data not found."

    )


@app.route(
    "/api/risk/feature-importance"
)
def risk_feature_importance():

    return objective_csv_api(

        "objective7/"
        "objective7_feature_importance.csv",

        "Feature importance data not found."

    )


@app.route(
    "/api/risk/explanations"
)
def risk_explanations():

    return objective_csv_api(

        "objective7/"
        "objective7_stockout_explanations.csv",

        "Explanation data not found."

    )


@app.route(
    "/api/risk/saved-predictions"
)
def saved_predictions():

    return objective_csv_api(

        "objective6/"
        "objective6_stockout_predictions.csv",

        "Saved prediction file not found."

    )


# ============================================================
# OBJECTIVE 6 — HGB
# ============================================================

def build_history_long():

    conn = get_db_connection()

    try:

        history = pd.read_sql_query(
            """
            SELECT
                id,
                snapshot_date,
                hospital_code,
                "A+",
                "A-",
                "B+",
                "B-",
                "O+",
                "O-",
                "AB+",
                "AB-"
            FROM stock_history
            ORDER BY
                snapshot_date ASC,
                id ASC
            """,
            conn
        )

    finally:

        conn.close()

    if history.empty:

        return pd.DataFrame()

    parts = []

    for group in BLOOD_GROUPS:

        temp = history[
            [
                "id",
                "snapshot_date",
                "hospital_code",
                group
            ]
        ].copy()

        temp["blood_group"] = group

        temp["stock"] = pd.to_numeric(
            temp[group],
            errors="coerce"
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

    return (
        result
        .sort_values(
            [
                "hospital_code",
                "blood_group",
                "snapshot_date",
                "id"
            ]
        )
        .reset_index(
            drop=True
        )
    )


def add_model_features(
    df
):

    result = df.copy()

    grouped = result.groupby(
        [
            "hospital_code",
            "blood_group"
        ]
    )

    result[
        "previous_stock"
    ] = grouped[
        "stock"
    ].shift(1)

    result[
        "stock_change"
    ] = (
        result["stock"]
        -
        result["previous_stock"]
    )

    result[
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

    result[
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

    result[
        "previous_stockout"
    ] = (
        result[
            "previous_stock"
        ]
        .fillna(-1)
        .eq(0)
        .astype(int)
    )

    result[
        "previous_low_stock"
    ] = (
        result[
            "previous_stock"
        ]
        .fillna(-1)
        .between(
            1,
            5
        )
        .astype(int)
    )

    result[
        "stock_vs_rolling_mean"
    ] = (
        result["stock"]
        -
        result[
            "rolling_mean_3"
        ]
    )

    result[
        "trend"
    ] = (
        result[
            "stock_change"
        ]
        .fillna(0)
    )

    result[
        "recovered_from_previous_stockout"
    ] = (
        (
            result[
                "previous_stockout"
            ] == 1
        )
        &
        (
            result[
                "stock"
            ] > 0
        )
    ).astype(int)

    result[
        "next_stock"
    ] = grouped[
        "stock"
    ].shift(-1)

    result = result[
        result[
            "next_stock"
        ].notna()
    ].copy()

    result[
        "target_stockout"
    ] = (
        result[
            "next_stock"
        ]
        .eq(0)
        .astype(int)
    )

    result[
        MODEL_FEATURES
    ] = (
        result[
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

    return result


def train_live_hgb():

    long_df = build_history_long()

    feature_df = add_model_features(
        long_df
    )

    X = feature_df[
        MODEL_FEATURES
    ].astype(float)

    y = feature_df[
        "target_stockout"
    ].astype(int)

    model = HistGradientBoostingClassifier(
        random_state=42,
        max_iter=200,
        learning_rate=0.05,
        max_leaf_nodes=31
    )

    model.fit(
        X,
        y
    )

    return model


def load_hgb_model():

    global LIVE_MODEL
    global LIVE_MODEL_SOURCE
    global LIVE_MODEL_SNAPSHOT_COUNT

    conn = get_db_connection()

    try:

        snapshot_count = conn.execute(
            """
            SELECT COUNT(
                DISTINCT snapshot_date
            )
            FROM stock_history
            """
        ).fetchone()[0]

    finally:

        conn.close()

    if (
        LIVE_MODEL is not None
        and
        LIVE_MODEL_SNAPSHOT_COUNT
        ==
        snapshot_count
    ):

        return LIVE_MODEL

    model_path = (
        MODELS_DIR
        /
        "histgradientboosting_stockout_model.pkl"
    )

    model = None

    if model_path.exists():

        try:

            model = joblib.load(
                model_path
            )

            LIVE_MODEL_SOURCE = (
                "saved_project_model"
            )

        except Exception as exc:

            print(
                "Saved model failed:",
                exc
            )

    if model is None:

        model = train_live_hgb()

        LIVE_MODEL_SOURCE = (
            "live_sqlite_retrained"
        )

    LIVE_MODEL = model

    LIVE_MODEL_SNAPSHOT_COUNT = (
        snapshot_count
    )

    return model


def build_live_features(
    hospital_code,
    blood_group
):

    conn = get_db_connection()

    try:

        query = f"""
            SELECT
                id,
                snapshot_date,
                "{blood_group}" AS stock
            FROM stock_history
            WHERE CAST(
                hospital_code AS TEXT
            ) = ?
            ORDER BY
                snapshot_date ASC,
                id ASC
        """

        history = pd.read_sql_query(
            query,
            conn,
            params=[
                str(hospital_code)
            ]
        )

    finally:

        conn.close()

    if history.empty:

        return None

    history["stock"] = pd.to_numeric(
        history["stock"],
        errors="coerce"
    ).fillna(0)

    history[
        "previous_stock"
    ] = history[
        "stock"
    ].shift(1).fillna(0)

    history[
        "stock_change"
    ] = (
        history["stock"]
        -
        history["previous_stock"]
    )

    history[
        "rolling_mean_3"
    ] = (
        history["stock"]
        .rolling(
            3,
            min_periods=1
        )
        .mean()
    )

    history[
        "rolling_std_3"
    ] = (
        history["stock"]
        .rolling(
            3,
            min_periods=1
        )
        .std()
        .fillna(0)
    )

    history[
        "previous_stockout"
    ] = (
        history[
            "previous_stock"
        ]
        .eq(0)
        .astype(int)
    )

    history[
        "previous_low_stock"
    ] = (
        history[
            "previous_stock"
        ]
        .between(
            1,
            5
        )
        .astype(int)
    )

    history[
        "stock_vs_rolling_mean"
    ] = (
        history["stock"]
        -
        history[
            "rolling_mean_3"
        ]
    )

    history[
        "trend"
    ] = history[
        "stock_change"
    ].fillna(0)

    history[
        "recovered_from_previous_stockout"
    ] = (
        (
            history[
                "previous_stockout"
            ] == 1
        )
        &
        (
            history[
                "stock"
            ] > 0
        )
    ).astype(int)

    latest = history.iloc[-1]

    features = {

        feature:
            float(
                latest.get(
                    feature,
                    0
                )
            )

        for feature in MODEL_FEATURES

    }

    return {

        "features":
            features,

        "observations":
            len(history),

        "latest_snapshot":
            str(
                history[
                    "snapshot_date"
                ].iloc[-1]
            )

    }


@app.route(
    "/api/risk/predict",
    methods=[
        "GET",
        "POST"
    ]
)
def risk_predict():

    payload = (
        request.get_json(
            silent=True
        )
        or {}
    )

    hospital_code = str(
        (
            payload.get(
                "hospital_code"
            )
            if payload
            else request.args.get(
                "hospital_code",
                ""
            )
        )
        or ""
    ).strip()

    blood_group = str(
        (
            payload.get(
                "blood_group"
            )
            if payload
            else request.args.get(
                "blood_group",
                ""
            )
        )
        or ""
    ).strip()

    if (
        not hospital_code
        or
        blood_group
        not in BLOOD_GROUPS
    ):

        return jsonify({

            "success":
                False,

            "message":
                "Valid hospital code and blood group are required."

        }), 400

    live = build_live_features(
        hospital_code,
        blood_group
    )

    if live is None:

        return jsonify({

            "success":
                False,

            "message":
                "No historical data found."

        }), 404

    model = load_hgb_model()

    X = pd.DataFrame(
        [
            live["features"]
        ],
        columns=MODEL_FEATURES
    )

    X = (
        X
        .replace(
            [
                np.inf,
                -np.inf
            ],
            np.nan
        )
        .fillna(0)
        .astype(float)
    )

    probability = float(
        model.predict_proba(
            X
        )[0][1]
    )

    prediction = int(
        model.predict(
            X
        )[0]
    )

    if probability >= 0.80:

        risk_level = "High"

    elif probability >= 0.50:

        risk_level = "Moderate"

    else:

        risk_level = "Low"

    reasons = []

    if live["features"][
        "stock"
    ] <= 5:

        reasons.append(
            "Current stock is in the low-stock range."
        )

    if live["features"][
        "previous_stockout"
    ] == 1:

        reasons.append(
            "The previous observation had a stockout."
        )

    if live["features"][
        "stock_change"
    ] < 0:

        reasons.append(
            "Stock is declining."
        )

    if live["features"][
        "stock_vs_rolling_mean"
    ] < 0:

        reasons.append(
            "Current stock is below the recent rolling average."
        )

    if not reasons:

        reasons.append(
            "Recent stock behaviour does not show a strong warning signal."
        )

    return jsonify({

        "success":
            True,

        "hospital_code":
            hospital_code,

        "blood_group":
            blood_group,

        "prediction":
            prediction,

        "stockout_probability":
            round(
                probability * 100,
                2
            ),

        "probability":
            probability,

        "risk_level":
            risk_level,

        "risk":
            risk_level,

        "observations":
            live[
                "observations"
            ],

        "latest_snapshot":
            live[
                "latest_snapshot"
            ],

        "features":
            live[
                "features"
            ],

        "reasons":
            reasons,

        "model":
            "HistGradientBoosting",

        "model_source":
            LIVE_MODEL_SOURCE

    })


# ============================================================
# OBJECTIVE 3 — DBARI
# ============================================================

def load_dbari():

    candidates = [

        "objective3/"
        "objective3_DBARI.csv",

        "objective3/"
        "Maharashtra_District_DBARI.csv"

    ]

    for path in candidates:

        df = load_csv(
            path
        )

        df = normalize_columns(
            df
        )

        if (
            not df.empty
            and
            find_column(
                df,
                ["DBARI"]
            ) is not None
        ):

            return df

    return discover_csv(
        "objective3",
        ["DBARI"]
    )


@app.route(
    "/api/gis/dbari"
)
def gis_dbari():

    df = load_dbari()

    if df.empty:

        return jsonify({

            "success":
                False,

            "message":
                "DBARI file not found."

        })

    return jsonify({

        "success":
            True,

        "data":
            clean_nan_records(df)

    })


# ============================================================
# OBJECTIVE 5 — GEOGRAPHIC VULNERABILITY
# ============================================================

def load_geographic_vulnerability():

    candidates = [

        "objective5/"
        "objective5_geographic_vulnerability_final.csv",

        "objective5/"
        "objective5_geographic_vulnerability.csv"

    ]

    for path in candidates:

        df = load_csv(
            path
        )

        df = normalize_columns(
            df
        )

        if (
            not df.empty
            and
            find_column(
                df,
                ["GVS"]
            ) is not None
        ):

            return df

    return discover_csv(
        "objective5",
        ["GVS"]
    )


@app.route(
    "/api/gis/vulnerability"
)
def gis_vulnerability():

    df = load_geographic_vulnerability()

    if df.empty:

        return jsonify({

            "success":
                False,

            "message":
                "Geographic vulnerability file not found."

        })

    if "latitude" in df.columns:

        df["latitude"] = pd.to_numeric(
            df["latitude"],
            errors="coerce"
        )

    if "longitude" in df.columns:

        df["longitude"] = pd.to_numeric(
            df["longitude"],
            errors="coerce"
        )

    df = df[
        df["latitude"].notna()
        &
        df["longitude"].notna()
        &
        (df["latitude"] != 0)
        &
        (df["longitude"] != 0)
    ].copy()

    return jsonify({

        "success":
            True,

        "count":
            len(df),

        "data":
            clean_nan_records(df)

    })


# ============================================================
# OBJECTIVE 9 — NETWORK
# ============================================================

def load_network():

    return discover_csv(
        "objective9",
        [
            "betweenness_centrality"
        ]
    )


@app.route(
    "/api/gis/network"
)
def gis_network():

    df = load_network()

    if df.empty:

        return jsonify({

            "success":
                False,

            "message":
                "Network file not found."

        })

    if "network_criticality" in df.columns:

        critical = (
            df
            .sort_values(
                "network_criticality",
                ascending=False
            )
            .head(20)
        )

    else:

        critical = (
            df
            .sort_values(
                "betweenness_centrality",
                ascending=False
            )
            .head(20)
        )

    return jsonify({

        "success":
            True,

        "node_count":
            len(df),

        "centrality":
            clean_nan_records(
                df
            ),

        "critical":
            clean_nan_records(
                critical
            )

    })


# ============================================================
# OBJECTIVE 10 — REDISTRIBUTION
# ============================================================

def load_redistribution():

    candidates = [

        "objective10/"
        "objective10_optimized_redistribution.csv",

        "objective10/"
        "objective10_feasible_transfers_network_aware.csv"

    ]

    for path in candidates:

        df = load_csv(
            path
        )

        df = normalize_columns(
            df
        )

        if not df.empty:

            return df

    return discover_csv(
        "objective10",
        [
            "donor_code",
            "recipient_code",
            "distance_km"
        ]
    )


@app.route(
    "/api/gis/redistribution"
)
def gis_redistribution():

    df = load_redistribution()

    if df.empty:

        return jsonify({

            "success":
                False,

            "message":
                "Redistribution file not found."

        })

    return jsonify({

        "success":
            True,

        "count":
            len(df),

        "data":
            clean_nan_records(
                df
            )

    })


# ============================================================
# OBJECTIVE 11 — SUMMARY
# ============================================================

@app.route(
    "/api/objective11/summary"
)
def objective11_summary():

    df = load_objective11()

    if df.empty:

        return jsonify({

            "success":
                False,

            "message":
                "Objective 11 ranking file not found."

        })

    origin_col = find_column(
        df,
        ["origin_code"]
    )

    alt_col = find_column(
        df,
        ["alternative_code"]
    )

    group_col = find_column(
        df,
        ["blood_group"]
    )

    score_col = find_column(
        df,
        ["decision_support_score"]
    )

    rank_col = find_column(
        df,
        ["rank"]
    )

    summary = {

        "recommendation_rows":
            len(df),

        "origin_banks":
            (
                int(
                    df[
                        origin_col
                    ].nunique()
                )
                if origin_col
                else 0
            ),

        "alternative_banks":
            (
                int(
                    df[
                        alt_col
                    ].nunique()
                )
                if alt_col
                else 0
            ),

        "blood_groups":
            (
                int(
                    df[
                        group_col
                    ].nunique()
                )
                if group_col
                else 0
            ),

        "top_ranked_rows":
            (
                int(
                    (
                        pd.to_numeric(
                            df[rank_col],
                            errors="coerce"
                        )
                        <= 5
                    ).sum()
                )
                if rank_col
                else 0
            )

    }

    if score_col:

        scores = pd.to_numeric(
            df[score_col],
            errors="coerce"
        )

        summary[
            "average_decision_score"
        ] = round(
            float(
                scores.mean()
            ),
            2
        )

    return jsonify({

        "success":
            True,

        "summary":
            summary

    })


# ============================================================
# OBJECTIVE 11 — RECOMMENDATIONS
# ============================================================

@app.route(
    "/api/objective11/recommendations"
)
def objective11_recommendations():

    origin_code = request.args.get(
        "origin_code",
        ""
    ).strip()

    blood_group = request.args.get(
        "blood_group",
        ""
    ).strip()

    try:

        limit = int(
            request.args.get(
                "limit",
                5
            )
        )

    except (
        ValueError,
        TypeError
    ):

        limit = 5

    limit = max(
        1,
        min(
            limit,
            20
        )
    )

    df = load_objective11()

    if df.empty:

        return jsonify({

            "success":
                False,

            "message":
                "Objective 11 ranking file not found."

        })

    origin_col = find_column(
        df,
        ["origin_code"]
    )

    group_col = find_column(
        df,
        ["blood_group"]
    )

    if origin_code:

        df = df[
            df[
                origin_col
            ]
            .astype(str)
            .str.strip()
            ==
            str(origin_code)
            .strip()
        ].copy()

    if blood_group:

        df = df[
            df[
                group_col
            ]
            .astype(str)
            .str.strip()
            .str.upper()
            ==
            blood_group.upper()
        ].copy()

    if df.empty:

        return jsonify({

            "success":
                True,

            "count":
                0,

            "data":
                []

        })

    rank_col = find_column(
        df,
        ["rank"]
    )

    score_col = find_column(
        df,
        ["decision_support_score"]
    )

    if rank_col:

        df["_rank_numeric"] = pd.to_numeric(
            df[rank_col],
            errors="coerce"
        )

    else:

        df["_rank_numeric"] = 999999

    if score_col:

        df["_score_numeric"] = pd.to_numeric(
            df[score_col],
            errors="coerce"
        )

    else:

        df["_score_numeric"] = 0

    df = (
        df
        .sort_values(
            [
                "_rank_numeric",
                "_score_numeric"
            ],
            ascending=[
                True,
                False
            ]
        )
        .head(
            limit
        )
        .drop(
            columns=[
                "_rank_numeric",
                "_score_numeric"
            ],
            errors="ignore"
        )
    )

    return jsonify({

        "success":
            True,

        "count":
            len(df),

        "data":
            clean_nan_records(
                df
            )

    })


# ============================================================
# START
# ============================================================

if __name__ == "__main__":

    app.run(
        debug=True,
        host="127.0.0.1",
        port=5000
    )