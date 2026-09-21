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


# ============================================================
# MUMBAI AREA SEARCH CATALOG
# ============================================================
#
# User-facing locality areas for Mumbai City and Mumbai
# Suburban. These are used only by the Find Blood page.
#
# Representative coordinates are used only for an approximate
# straight-line distance when an area has no blood bank of its
# own. They do not modify research/GIS calculations.
# ============================================================

MUMBAI_AREA_CATALOG = {'Mumbai': {'Colaba': (18.9067, 72.8147),
  'Cuffe Parade': (18.9145, 72.822),
  'Fort': (18.932, 72.8347),
  'Churchgate': (18.9353, 72.827),
  'Ballard Estate': (18.9358, 72.8398),
  'Nariman Point': (18.9255, 72.8242),
  'Marine Lines': (18.944, 72.8245),
  'Kalbadevi': (18.9495, 72.8265),
  'Bhuleshwar': (18.9523, 72.83),
  'Mandvi': (18.9522, 72.838),
  'Masjid': (18.9561, 72.8392),
  'Dongri': (18.9574, 72.838),
  'Girgaon': (18.954, 72.8145),
  'Charni Road': (18.9525, 72.8182),
  'Grant Road': (18.964, 72.815),
  'Khetwadi': (18.9605, 72.8188),
  'Malabar Hill': (18.9549, 72.8037),
  'Cumballa Hill': (18.9647, 72.8084),
  'Tardeo': (18.9715, 72.8175),
  'Mumbai Central': (18.9698, 72.8205),
  'Mahalaxmi': (18.982, 72.812),
  'Breach Candy': (18.9727, 72.8044),
  'Agripada': (18.9715, 72.823),
  'Nagpada': (18.965, 72.824),
  'Byculla': (18.976, 72.8338),
  'Mazgaon': (18.9685, 72.843),
  'Dockyard Road': (18.969, 72.8465),
  'Parel': (18.9993, 72.8406),
  'Lalbaug': (18.9986, 72.837),
  'Lower Parel': (18.998, 72.825),
  'Dadar': (19.0233, 72.8377),
  'Prabhadevi': (19.0135, 72.825),
  'Worli': (19.017, 72.817),
  'Mahim': (19.0335, 72.8383),
  'Matunga': (19.0278, 72.855),
  'Sion': (19.0365, 72.8687),
  'Dharavi': (19.0445, 72.855),
  'Wadala': (19.0165, 72.858),
  'Antop Hill': (19.0268, 72.8725),
  'Sewri': (19.0005, 72.8545),
  'Shivdi': (19.003, 72.859),
  'Cotton Green': (18.9948, 72.8515),
  'Chinchpokli': (18.9785, 72.83),
  'Jacob Circle': (18.976, 72.824),
  'Reay Road': (18.981, 72.857),
  'Sandhurst Road': (18.9538, 72.845),
  'Princess Dock': (18.949, 72.85),
  'Chinch Bunder': (18.9525, 72.844)},
 'Mumbai Suburban': {'Bandra': (19.0575, 72.8336),
  'Bandra East': (19.0605, 72.849),
  'Bandra West': (19.06, 72.836),
  'Bandra Kurla Complex': (19.068, 72.869),
  'Khar': (19.0695, 72.836),
  'Khar East': (19.071, 72.846),
  'Khar West': (19.0695, 72.832),
  'Santacruz': (19.081, 72.842),
  'Santacruz East': (19.083, 72.854),
  'Santacruz West': (19.08, 72.839),
  'Vile Parle': (19.1, 72.843),
  'Vile Parle East': (19.1005, 72.852),
  'Vile Parle West': (19.098, 72.836),
  'Juhu': (19.1075, 72.8265),
  'Vakola': (19.0855, 72.861),
  'Kalina': (19.082, 72.865),
  'Andheri': (19.1195, 72.846),
  'Andheri East': (19.1155, 72.865),
  'Andheri West': (19.1195, 72.844),
  'Chakala': (19.116, 72.8615),
  'Marol': (19.1175, 72.879),
  'MIDC': (19.1115, 72.8735),
  'Saki Naka': (19.1015, 72.888),
  'Jogeshwari': (19.137, 72.848),
  'Jogeshwari East': (19.136, 72.861),
  'Jogeshwari West': (19.137, 72.842),
  'Goregaon': (19.166, 72.852),
  'Goregaon East': (19.166, 72.865),
  'Goregaon West': (19.1595, 72.8403),
  'Aarey': (19.145, 72.869),
  'Malad': (19.187, 72.848),
  'Malad East': (19.186, 72.858),
  'Malad West': (19.187, 72.838),
  'Malvani': (19.198, 72.822),
  'Marve': (19.195, 72.798),
  'Madh Island': (19.135, 72.795),
  'Kandivali': (19.204, 72.839),
  'Kandivali East': (19.205, 72.861),
  'Kandivali West': (19.205, 72.837),
  'Charkop': (19.22, 72.82),
  'Poisar': (19.205, 72.845),
  'Kurar': (19.192, 72.872),
  'Borivali': (19.23, 72.8525),
  'Borivali East': (19.23, 72.864),
  'Borivali West': (19.23, 72.843),
  'Dahisar': (19.25, 72.858),
  'Dahisar East': (19.25, 72.866),
  'Dahisar West': (19.25, 72.851),
  'Gorai': (19.245, 72.8),
  'Manori': (19.2, 72.78),
  'Kurla': (19.072, 72.879),
  'Kurla East': (19.073, 72.888),
  'Kurla West': (19.071, 72.872),
  'Chandivali': (19.114, 72.9),
  'Powai': (19.118, 72.907),
  'Ghatkopar': (19.085, 72.908),
  'Ghatkopar East': (19.0825, 72.915),
  'Ghatkopar West': (19.086, 72.9),
  'Vikhroli': (19.1115, 72.927),
  'Vikhroli East': (19.112, 72.938),
  'Vikhroli West': (19.111, 72.918),
  'Kanjurmarg': (19.129, 72.94),
  'Kanjurmarg East': (19.131, 72.948),
  'Kanjurmarg West': (19.128, 72.931),
  'Bhandup': (19.146, 72.937),
  'Bhandup East': (19.145, 72.945),
  'Bhandup West': (19.146, 72.93),
  'Nahur': (19.155, 72.946),
  'Mulund': (19.1726, 72.956),
  'Mulund East': (19.1715, 72.965),
  'Mulund West': (19.1725, 72.947),
  'Chembur': (19.062, 72.899),
  'Tilak Nagar': (19.067, 72.899),
  'Govandi': (19.055, 72.915),
  'Govandi East': (19.054, 72.92),
  'Govandi West': (19.057, 72.909),
  'Mankhurd': (19.048, 72.933),
  'Deonar': (19.059, 72.916),
  'Trombay': (19.016, 72.957),
  'Anushakti Nagar': (19.0425, 72.9145)}}


MUMBAI_AREA_ALIASES = {'Mumbai': {'girgoan': 'Girgaon',
  'dadar west': 'Dadar',
  'vileparle': 'Vile Parle',
  'mumbai cst': 'Fort',
  'chinch bunder': 'Chinch Bunder'},
 'Mumbai Suburban': {'vileparle': 'Vile Parle',
  'bandra reclamation': 'Bandra West',
  'bandra reclaimation': 'Bandra West',
  'rajawadi hospital': 'Ghatkopar East',
  'janta market': 'Bhandup West',
  'janta markat': 'Bhandup West',
  'andher': 'Andheri East',
  'mumabi': 'Andheri East',
  'cts no': 'Jogeshwari West',
  'goregaon': 'Goregaon',
  'kurla': 'Kurla'}}


def normalize_search_text(value):

    return (
        str(value or "")
        .strip()
        .casefold()
    )


def canonical_mumbai_area(
    district,
    area
):

    if not area:

        return None

    district_key = (
        "Mumbai Suburban"
        if normalize_search_text(
            district
        )
        == "mumbai suburban"
        else "Mumbai"
    )

    text = normalize_search_text(
        area
    )

    alias_map = (
        MUMBAI_AREA_ALIASES.get(
            district_key,
            {}
        )
    )

    if text in alias_map:

        return alias_map[text]

    for canonical_name in (
        MUMBAI_AREA_CATALOG[
            district_key
        ]
    ):

        if (
            normalize_search_text(
                canonical_name
            )
            ==
            text
        ):

            return canonical_name

    return None


def mumbai_area_coordinates(
    district,
    canonical_area
):

    if not canonical_area:

        return None

    district_key = (
        "Mumbai Suburban"
        if normalize_search_text(
            district
        )
        == "mumbai suburban"
        else "Mumbai"
    )

    return (
        MUMBAI_AREA_CATALOG[
            district_key
        ].get(
            canonical_area
        )
    )


def haversine_km(
    lat1,
    lon1,
    lat2,
    lon2
):

    radius = 6371.0

    lat1 = np.radians(lat1)
    lon1 = np.radians(lon1)
    lat2 = np.radians(lat2)
    lon2 = np.radians(lon2)

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = (
        np.sin(dlat / 2.0) ** 2
        +
        np.cos(lat1)
        *
        np.cos(lat2)
        *
        np.sin(dlon / 2.0) ** 2
    )

    return float(
        2.0
        *
        radius
        *
        np.arctan2(
            np.sqrt(a),
            np.sqrt(
                np.maximum(
                    0.0,
                    1.0 - a
                )
            )
        )
    )


def bank_area_matches(
    bank_area,
    canonical_area
):

    if (
        not bank_area
        or
        not canonical_area
    ):

        return False

    text = normalize_search_text(
        bank_area
    )

    canonical = normalize_search_text(
        canonical_area
    )

    if text == canonical:

        return True

    known_aliases = {

        "dadar west":
            "dadar",

        "girgoan":
            "girgaon",

        "mumbai cst":
            "fort",

        "bandra reclaimation":
            "bandra west",

        "bandra reclamation":
            "bandra west",

        "rajawadi hospit":
            "ghatkopar east",

        "janta markat":
            "bhandup west",

        "janta market":
            "bhandup west",

        "andher":
            "andheri east",

        "mumabi":
            "andheri east",

        "cts no":
            "jogeshwari west",

        "opp. kem hospital":
            "parel",

        "l.t.":
            "fort",

        "dr.g.deshmukh":
            "cumballa hill",

    }

    return (
        known_aliases.get(
            text
        )
        ==
        canonical
    )


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

        # ----------------------------------------------------
        # REAL DISTRICT -> CITY -> AREA HIERARCHY
        # Kept for all districts outside Mumbai City and
        # Mumbai Suburban.
        # ----------------------------------------------------

        hierarchy_rows = conn.execute(
            """
            SELECT DISTINCT
                district,
                city,
                area
            FROM blood_banks
            WHERE district IS NOT NULL
              AND TRIM(district) != ''
              AND city IS NOT NULL
              AND TRIM(city) != ''
              AND area IS NOT NULL
              AND TRIM(area) != ''
            ORDER BY
                district,
                city,
                area
            """
        ).fetchall()

        hierarchy = {}

        for row in hierarchy_rows:

            district_name = str(
                row["district"]
            ).strip()

            city_name = str(
                row["city"]
            ).strip()

            area_name = str(
                row["area"]
            ).strip()

            if (
                not district_name
                or
                not city_name
                or
                not area_name
            ):

                continue

            hierarchy.setdefault(
                district_name,
                {}
            )

            hierarchy[
                district_name
            ].setdefault(
                city_name,
                set()
            )

            hierarchy[
                district_name
            ][
                city_name
            ].add(
                area_name
            )

        hierarchy_output = []

        for district_name in sorted(
            hierarchy.keys(),
            key=lambda value:
                value.casefold()
        ):

            city_output = []

            for city_name in sorted(
                hierarchy[
                    district_name
                ].keys(),
                key=lambda value:
                    value.casefold()
            ):

                city_output.append({

                    "city":
                        city_name,

                    "areas":
                        sorted(
                            hierarchy[
                                district_name
                            ][
                                city_name
                            ],
                            key=lambda value:
                                value.casefold()
                        )

                })

            hierarchy_output.append({

                "district":
                    district_name,

                "cities":
                    city_output

            })

        cityless_districts = [
            "Mumbai",
            "Mumbai Suburban"
        ]

        # ----------------------------------------------------
        # SPECIAL MUMBAI AREA CATALOG
        #
        # For Mumbai City and Mumbai Suburban the user goes
        # directly:
        #
        # District -> Area
        #
        # The area field remains a plain text field with no
        # dropdown/autocomplete. The catalog is returned only
        # so the frontend can validate the typed locality.
        # ----------------------------------------------------

        valid_areas_by_district = {

            "Mumbai":
                sorted(
                    MUMBAI_AREA_CATALOG[
                        "Mumbai"
                    ].keys(),
                    key=lambda value:
                        value.casefold()
                ),

            "Mumbai Suburban":
                sorted(
                    MUMBAI_AREA_CATALOG[
                        "Mumbai Suburban"
                    ].keys(),
                    key=lambda value:
                        value.casefold()
                )

        }

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
            ],

            "hierarchy":
                hierarchy_output,

            "cityless_districts":
                cityless_districts,

            "valid_areas_by_district":
                valid_areas_by_district

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

        # ====================================================
        # SPECIAL RULE:
        # Mumbai City + Mumbai Suburban
        #
        # These two districts use:
        #
        #     District -> Area
        #
        # and only a curated, valid locality catalog is
        # accepted.
        #
        # If a valid area has no blood bank of its own,
        # return the closest registered Mumbai blood bank.
        #
        # IMPORTANT:
        # The closest-bank fallback happens ONLY when there
        # is no bank registered in the requested area.
        # If a bank exists in the area but lacks the requested
        # stock, we do NOT silently substitute another area.
        # ====================================================

        district_key = (
            "Mumbai Suburban"
            if normalize_search_text(
                district
            )
            == "mumbai suburban"
            else (
                "Mumbai"
                if normalize_search_text(
                    district
                )
                == "mumbai"
                else district
            )
        )

        is_mumbai_special = (
            district_key
            in {
                "Mumbai",
                "Mumbai Suburban"
            }
        )

        canonical_area = None

        if (
            is_mumbai_special
            and
            area
        ):

            canonical_area = canonical_mumbai_area(
                district_key,
                area
            )

            if canonical_area is None:

                valid_areas = sorted(
                    MUMBAI_AREA_CATALOG[
                        district_key
                    ].keys(),
                    key=lambda value:
                        value.casefold()
                )

                return jsonify({

                    "success":
                        False,

                    "error":
                        "invalid_area",

                    "message":
                        "Please enter a valid area for "
                        f"{district_key}.",

                    "district":
                        district_key,

                    "valid_areas":
                        valid_areas

                }), 400

        # ====================================================
        # SPECIAL MUMBAI AREA SEARCH
        # ====================================================

        if (
            is_mumbai_special
            and
            canonical_area
        ):

            # ------------------------------------------------
            # Get all blood banks that belong to Mumbai City
            # / Mumbai Suburban in the current database.
            #
            # We intentionally use both district and city
            # because the current source data contains some
            # Mumbai-suburban facilities recorded under the
            # broader "Mumbai" district label.
            # ------------------------------------------------

            region_rows = conn.execute(
                """
                SELECT *
                FROM blood_banks
                WHERE
                    LOWER(TRIM(district))
                        IN (
                            'mumbai',
                            'mumbai suburban'
                        )
                    OR
                    LOWER(TRIM(city))
                        = 'mumbai'
                ORDER BY
                    blood_bank_name
                """
            ).fetchall()

            # ------------------------------------------------
            # 1. FIRST: determine whether a bank is actually
            # registered in the requested area.
            #
            # This uses canonical area names and a small set
            # of known legacy aliases such as:
            #   Dadar West -> Dadar
            #   Girgoan -> Girgaon
            # ------------------------------------------------

            local_rows = [

                row

                for row in region_rows

                if bank_area_matches(
                    row["area"],
                    canonical_area
                )

            ]

            # ------------------------------------------------
            # The area HAS a blood bank.
            #
            # Therefore we only show banks from that area
            # that satisfy the requested blood-stock amount.
            #
            # No nearest-area substitution is performed.
            # ------------------------------------------------

            if local_rows:

                results = []

                for row in local_rows:

                    try:

                        stock = float(
                            row[
                                blood_group
                            ]
                        )

                    except (
                        ValueError,
                        TypeError,
                        KeyError
                    ):

                        continue

                    if (
                        stock
                        <
                        required_units
                    ):

                        continue

                    results.append({

                        "hospital_code":
                            row[
                                "hospital_code"
                            ],

                        "blood_bank_name":
                            row[
                                "blood_bank_name"
                            ],

                        "address":
                            row[
                                "address"
                            ],

                        "contact":
                            row[
                                "contact"
                            ],

                        "hospital_type":
                            row[
                                "hospital_type"
                            ],

                        "district":
                            row[
                                "district"
                            ],

                        "area":
                            canonical_area,

                        "city":
                            row[
                                "city"
                            ],

                        "latitude":
                            row[
                                "latitude"
                            ],

                        "longitude":
                            row[
                                "longitude"
                            ],

                        "stock":
                            stock,

                        "blood_group":
                            blood_group,

                        "is_area_fallback":
                            False,

                        "distance_km":
                            None

                    })

                results.sort(
                    key=lambda item:
                        item[
                            "stock"
                        ],
                    reverse=True
                )

                return jsonify({

                    "success":
                        True,

                    "search_level":
                        "area",

                    "blood_group":
                        blood_group,

                    "required_units":
                        required_units,

                    "selected_area":
                        canonical_area,

                    "area_has_blood_bank":
                        True,

                    "area_fallback":
                        False,

                    "results":
                        results

                })

            # ------------------------------------------------
            # 2. NO BANK IN REQUESTED AREA
            #
            # Find the closest registered Mumbai blood bank.
            # ------------------------------------------------

            area_coordinates = (
                mumbai_area_coordinates(
                    district_key,
                    canonical_area
                )
            )

            if area_coordinates is None:

                return jsonify({

                    "success":
                        True,

                    "search_level":
                        "area",

                    "blood_group":
                        blood_group,

                    "required_units":
                        required_units,

                    "selected_area":
                        canonical_area,

                    "area_has_blood_bank":
                        False,

                    "area_fallback":
                        False,

                    "results":
                        [],

                    "message":
                        "No representative location "
                        "is available for this area."

                })

            area_lat, area_lon = (
                area_coordinates
            )

            candidates = []

            for row in region_rows:

                latitude = row[
                    "latitude"
                ]

                longitude = row[
                    "longitude"
                ]

                if (
                    latitude is None
                    or
                    longitude is None
                ):

                    continue

                try:

                    bank_lat = float(
                        latitude
                    )

                    bank_lon = float(
                        longitude
                    )

                except (
                    ValueError,
                    TypeError
                ):

                    continue

                distance = haversine_km(
                    area_lat,
                    area_lon,
                    bank_lat,
                    bank_lon
                )

                try:

                    stock = float(
                        row[
                            blood_group
                        ]
                    )

                except (
                    ValueError,
                    TypeError,
                    KeyError
                ):

                    stock = 0.0

                candidates.append({

                    "row":
                        row,

                    "distance_km":
                        distance,

                    "stock":
                        stock,

                    "stock_sufficient":
                        (
                            stock
                            >=
                            required_units
                        )

                })

            # ------------------------------------------------
            # Prefer the nearest bank that actually has the
            # requested number of units.
            # ------------------------------------------------

            sufficient = [

                item
                for item in candidates
                if item[
                    "stock_sufficient"
                ]

            ]

            if sufficient:

                sufficient.sort(
                    key=lambda item:
                        item[
                            "distance_km"
                        ]
                )

                selected = [
                    sufficient[0]
                ]

                fallback_stock_available = True

            elif candidates:

                candidates.sort(
                    key=lambda item:
                        item[
                            "distance_km"
                        ]
                )

                selected = [
                    candidates[0]
                ]

                fallback_stock_available = False

            else:

                return jsonify({

                    "success":
                        True,

                    "search_level":
                        "area-fallback",

                    "blood_group":
                        blood_group,

                    "required_units":
                        required_units,

                    "selected_area":
                        canonical_area,

                    "area_has_blood_bank":
                        False,

                    "area_fallback":
                        True,

                    "results":
                        [],

                    "message":
                        "No Mumbai blood-bank coordinate "
                        "was available for a nearest-bank search."

                })

            results = []

            for item in selected:

                row = item[
                    "row"
                ]

                results.append({

                    "hospital_code":
                        row[
                            "hospital_code"
                        ],

                    "blood_bank_name":
                        row[
                            "blood_bank_name"
                        ],

                    "address":
                        row[
                            "address"
                        ],

                    "contact":
                        row[
                            "contact"
                        ],

                    "hospital_type":
                        row[
                            "hospital_type"
                        ],

                    "district":
                        row[
                            "district"
                        ],

                    "area":
                        row[
                            "area"
                        ],

                    "city":
                        row[
                            "city"
                        ],

                    "latitude":
                        row[
                            "latitude"
                        ],

                    "longitude":
                        row[
                            "longitude"
                        ],

                    "stock":
                        item[
                            "stock"
                        ],

                    "blood_group":
                        blood_group,

                    "is_area_fallback":
                        True,

                    "distance_km":
                        round(
                            item[
                                "distance_km"
                            ],
                            2
                        ),

                    "stock_sufficient":
                        item[
                            "stock_sufficient"
                        ]

                })

            if fallback_stock_available:

                message = (
                    f"No blood bank is registered in "
                    f"{canonical_area}. "
                    f"Showing the closest registered "
                    f"blood bank with {blood_group} "
                    f"stock."
                )

            else:

                message = (
                    f"No blood bank is registered in "
                    f"{canonical_area}, and no nearby "
                    f"Mumbai blood bank currently has "
                    f"{required_units} or more units of "
                    f"{blood_group}. Showing the closest "
                    f"registered blood bank and its current stock."
                )

            return jsonify({

                "success":
                    True,

                "search_level":
                    "area-fallback",

                "blood_group":
                    blood_group,

                "required_units":
                    required_units,

                "selected_area":
                    canonical_area,

                "area_has_blood_bank":
                    False,

                "area_fallback":
                    True,

                "fallback_stock_available":
                    fallback_stock_available,

                "message":
                    message,

                "results":
                    results

            })

        # ====================================================
        # ORIGINAL MAHARASHTRA-WIDE SEARCH
        #
        # For all other districts, preserve the existing
        # behaviour.
        # ====================================================

        filters = []
        parameters = []

        if district:

            filters.append(
                "LOWER(TRIM(district)) = LOWER(TRIM(?))"
            )

            parameters.append(
                district
            )

        if city:

            filters.append(
                "LOWER(TRIM(city)) = LOWER(TRIM(?))"
            )

            parameters.append(
                city
            )

        if area:

            filters.append(
                "LOWER(TRIM(area)) = LOWER(TRIM(?))"
            )

            parameters.append(
                area
            )

        query = """
            SELECT *
            FROM blood_banks
        """

        if filters:

            query += (
                " WHERE "
                +
                " AND ".join(
                    filters
                )
            )

        query += """
            ORDER BY
                district,
                city,
                area,
                blood_bank_name
        """

        rows = conn.execute(
            query,
            parameters
        ).fetchall()

        if (
            district
            and
            city
            and
            area
        ):

            search_level = (
                "district-city-area"
            )

        elif (
            district
            and
            area
        ):

            search_level = (
                "district-area"
            )

        elif (
            district
            and
            city
        ):

            search_level = (
                "district-city"
            )

        elif area:

            search_level = "area"

        elif city:

            search_level = "city"

        elif district:

            search_level = "district"

        else:

            search_level = "all"

        results = []

        for row in rows:

            try:

                stock = float(
                    row[
                        blood_group
                    ]
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
                    row[
                        "hospital_code"
                    ],

                "blood_bank_name":
                    row[
                        "blood_bank_name"
                    ],

                "address":
                    row[
                        "address"
                    ],

                "contact":
                    row[
                        "contact"
                    ],

                "hospital_type":
                    row[
                        "hospital_type"
                    ],

                "district":
                    row[
                        "district"
                    ],

                "area":
                    row[
                        "area"
                    ],

                "city":
                    row[
                        "city"
                    ],

                "latitude":
                    row[
                        "latitude"
                    ],

                "longitude":
                    row[
                        "longitude"
                    ],

                "stock":
                    stock,

                "blood_group":
                    blood_group,

                "is_area_fallback":
                    False,

                "distance_km":
                    None

            })

        results.sort(
            key=lambda item:
                item[
                    "stock"
                ],
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

            "area_fallback":
                False,

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