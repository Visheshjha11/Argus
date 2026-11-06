# predict.py

import os
import sys
import textwrap
import requests


from datetime import datetime, timedelta, timezone
from dotenv import load_dotenv

from argus_core import (
    RAW_COLUMNS,
    build_reasoning,
    featurize,
    load_bundle,
    shapley_attribution,
)

import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_FILE = "argus_model.pkl"
NASA_API_URL = "https://api.nasa.gov/neo/rest/v1/feed"

MAX_DISPLAY = 10

# Calibrated probability at which the ML layer calls an object hazardous.
# Matches the lower edge of the MEDIUM band in the README.
ML_FLAG_THRESHOLD = 0.40

WRAP_WIDTH = 78

# Emoji status markers crash on cp1252 consoles (default on Windows).
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


# ============================================================
# LOAD API KEY
# ============================================================

load_dotenv()

API_KEY = os.getenv("NASA_API_KEY")


def require_api_key():
    if not API_KEY:
        raise ValueError(
            "NASA_API_KEY is not set in the .env file."
        )
# ============================================================
# FETCH LATEST AVAILABLE NASA DATA
# ============================================================

def fetch_latest_asteroids():

    # One request covering the last 7 days instead of up to 7 sequential
    # requests; then take the most recent day that has data.

    today = datetime.now(timezone.utc).date()

    params = {
        "start_date": (today - timedelta(days=6)).strftime("%Y-%m-%d"),
        "end_date": today.strftime("%Y-%m-%d"),
        "api_key": API_KEY
    }

    response = requests.get(
        NASA_API_URL,
        params=params,
        timeout=30
    )

    response.raise_for_status()

    neo_data = response.json().get(
        "near_earth_objects",
        {}
    )

    for date_string in sorted(neo_data, reverse=True):

        if neo_data[date_string]:
            return date_string, neo_data[date_string]

    raise RuntimeError(
        "No NASA asteroid data found in the last 7 days."
    )


# ============================================================
# EXTRACT FEATURES
# ============================================================

def extract_features(asteroid):

    # Canonical units (see argus_core.RAW_COLUMNS): metres, km/h, km.

    approach = asteroid["close_approach_data"][0]

    diameter_min = (
        asteroid["estimated_diameter"]
        ["meters"]
        ["estimated_diameter_min"]
    )

    diameter_max = (

        asteroid["estimated_diameter"]
        ["meters"]
        ["estimated_diameter_max"]

    )



    return {
        "diameter_m": (diameter_min + diameter_max) / 2,
        "velocity_kmh": float(
            approach["relative_velocity"]["kilometers_per_hour"]
        ),
        "miss_km": float(
            approach["miss_distance"]["kilometers"]
        ),
        "magnitude": float(
            asteroid["absolute_magnitude_h"]
        ),
        "diameter_range_m": (diameter_min, diameter_max),
    }


# ============================================================
# PREDICT ONE ASTEROID
# ============================================================

def predict_asteroid(asteroid, bundle):

    values = extract_features(asteroid)

    features = featurize(
        pd.DataFrame([{c: values[c] for c in RAW_COLUMNS}])
    )

    # --------------------------------------------------------
    # ML PREDICTION (calibrated probability)
    # --------------------------------------------------------

    probabilities = bundle["model"].predict_proba(features)[0]

    safe_probability = probabilities[0]

    hazardous_probability = probabilities[1]

    prediction = int(hazardous_probability >= ML_FLAG_THRESHOLD)

    ml_risk_score = hazardous_probability * 100

    # --------------------------------------------------------
    # NASA CLASSIFICATION
    # --------------------------------------------------------

    nasa_hazardous = asteroid[
        "is_potentially_hazardous_asteroid"
    ]

    # --------------------------------------------------------
    # ARGUS FINAL ASSESSMENT
    #
    # NASA classification is the baseline.
    # ML probability is treated as an additional signal.
    # --------------------------------------------------------

    if nasa_hazardous:

        if ml_risk_score >= 75:
            threat_level = "HIGH"

        elif ml_risk_score >= 40:
            threat_level = "MEDIUM"

        else:
            threat_level = "LOW"

        final_prediction = "POTENTIALLY HAZARDOUS"

    else:

        # NASA says it is NOT potentially hazardous.
        # Therefore ARGUS does not override NASA solely
        # because of the ML probability.

        if ml_risk_score >= 75:

            threat_level = "REVIEW"

            final_prediction = (
                "SAFE - ELEVATED ML SIGNAL"
            )

        elif ml_risk_score >= 40:

            threat_level = "LOW"

            final_prediction = (
                "LOW HAZARD - ELEVATED ML SIGNAL"
            )

        else:

            threat_level = "LOW"

            final_prediction = "LOW HAZARD"

    return {

        "id": asteroid["id"],

        "name": asteroid["name"],

        "diameter": values["diameter_m"],

        "velocity": values["velocity_kmh"],

        "miss_distance": values["miss_km"],

        "magnitude": values["magnitude"],

        "diameter_range_m": values["diameter_range_m"],

        "ml_risk_score": ml_risk_score,

        "threat_level": threat_level,

        "prediction": prediction,

        "safe_probability": safe_probability,

        "hazardous_probability": hazardous_probability,

        "nasa_hazardous": nasa_hazardous,

        "final_prediction": final_prediction
    }


# ============================================================
# EXPLAIN ONE ASTEROID
# ============================================================

def explain_result(result, bundle):

    raw = {
        "diameter_m": result["diameter"],
        "velocity_kmh": result["velocity"],
        "miss_km": result["miss_distance"],
        "magnitude": result["magnitude"],
    }

    attribution = shapley_attribution(
        bundle["model"],
        raw,
        bundle["background"]
    )

    return build_reasoning(
        raw,
        attribution,
        bundle["reference"],
        nasa_hazardous=result["nasa_hazardous"],
        ml_risk_score=result["ml_risk_score"],
        diameter_range_m=result["diameter_range_m"]
    )


def print_bullets(title, items, indent=2):

    print(f"\n{' ' * indent}{title}")

    prefix = " " * (indent + 2) + "- "
    follow = " " * (indent + 4)

    for item in items:
        print(
            textwrap.fill(
                item,
                width=WRAP_WIDTH,
                initial_indent=prefix,
                subsequent_indent=follow
            )
        )


# ============================================================
# DISPLAY SINGLE ASTEROID
# ============================================================

def display_single(result, latest_date, total, bundle):

    reasoning = explain_result(result, bundle)

    print("\n==============================================")
    print("            ARGUS AI PREDICTION")
    print("==============================================")

    print(
        f"\nNASA DATA DATE"
        f"\n{latest_date}"
    )

    print(
        f"\nTOTAL ASTEROIDS DETECTED"
        f"\n{total}"
    )

    print(
        f"\nASTEROID"
        f"\n{result['name']}"
    )

    print("\n----------------------------------------------")

    print(
        f"\nML RISK SCORE"
        f"\n{result['ml_risk_score']:.2f} / 100"
    )

    print(
        f"\nTHREAT LEVEL"
        f"\n{result['threat_level']}"
    )

    print(
        f"\nDiameter"
        f"\n{result['diameter']:.2f} m"
    )

    print(
        f"\nVelocity"
        f"\n{result['velocity']:,.2f} km/h"
    )

    print(
        f"\nMiss Distance"
        f"\n{result['miss_distance']:,.2f} km"
    )

    print(
        f"\nMagnitude"
        f"\n{result['magnitude']:.2f}"
    )

    print("\n----------------------------------------------")

    print(
        f"\nHazardous Probability"
        f"\n{result['hazardous_probability'] * 100:.2f}%"
    )

    print(
        f"\nSafe Probability"
        f"\n{result['safe_probability'] * 100:.2f}%"
    )

    print("\n----------------------------------------------")

    print("\nNASA CLASSIFICATION")

    if result["nasa_hazardous"]:
        print("⚠️ POTENTIALLY HAZARDOUS")
    else:
        print("✅ NOT POTENTIALLY HAZARDOUS")

    print("\nARGUS ML PREDICTION")

    if result["prediction"] == 1:
        print("⚠️ POTENTIALLY HAZARDOUS")
    else:
        print("✅ LOW HAZARD")

    print("\nARGUS FINAL ASSESSMENT")

    if result["final_prediction"] == "POTENTIALLY HAZARDOUS":

        print("🔴 POTENTIALLY HAZARDOUS")

    elif "ELEVATED ML" in result["final_prediction"]:

        print("🟡 ELEVATED ML SIGNAL")

    else:

        print("🟢 LOW HAZARD")

    # --------------------------------------------------------
    # REASONING
    # --------------------------------------------------------

    print("\n----------------------------------------------")
    print("\nWHY THIS ASSESSMENT")

    print_bullets(
        "NASA hazard criteria (H <= 22 and MOID <= 0.05 AU)",
        reasoning["criteria"]
    )

    print_bullets(
        "Impact severity context",
        reasoning["severity"]
    )

    print_bullets(
        "What drove the ML score (exact Shapley attribution)",
        reasoning["drivers"]
    )

    print_bullets(
        "Verdict",
        reasoning["verdict"]
    )

    # --------------------------------------------------------
    # MODEL CARD
    # --------------------------------------------------------

    holdout = bundle["metrics"]["holdout"]
    meta = bundle["meta"]

    print("\n----------------------------------------------")
    print("\nMODEL CARD")
    print(
        f"\n  Random Forest, isotonic-calibrated | "
        f"trained on {meta['asteroids']:,} asteroids "
        f"({meta['rows']:,} approaches)"
    )
    print(
        f"  Held-out ROC-AUC {holdout['roc_auc']:.3f} | "
        f"PR-AUC {holdout['pr_auc']:.3f} | "
        f"calibration error {holdout['ece']:.3f}"
    )
    print(
        "  Limitation: MOID and orbital elements are not model inputs, so "
        "objects\n  large enough to qualify but passing far from Earth "
        "cannot be separated reliably."
    )

    print("\n==============================================")


# ============================================================
# DISPLAY MULTIPLE ASTEROIDS
# ============================================================

def display_multiple(results, latest_date, total, bundle):

    results_to_display = results[:MAX_DISPLAY]

    print("\n==============================================")
    print("                 ARGUS AI")
    print("==============================================")

    print(
        f"\nNASA DATA DATE: {latest_date}"
    )

    print(
        f"TOTAL ASTEROIDS DETECTED: {total}"
    )

    print(
        f"DISPLAYING: {len(results_to_display)}"
    )

    print("==============================================")

    print("\nTOP RISK ASTEROIDS")
    print("----------------------------------------------")

    for index, result in enumerate(
        results_to_display,
        start=1
    ):

        reasoning = explain_result(result, bundle)

        print(
            f"\n#{index} {result['name']}"
        )

        print(
            f"    ML Risk Score    : "
            f"{result['ml_risk_score']:.2f}/100"
        )

        print(
            f"    Threat Level     : "
            f"{result['threat_level']}"
        )

        print(
            f"    Diameter         : "
            f"{result['diameter']:.2f} m"
        )

        print(
            f"    Velocity         : "
            f"{result['velocity']:,.2f} km/h"
        )

        print(
            f"    Miss Distance    : "
            f"{result['miss_distance']:,.2f} km"
        )

        print(
            f"    NASA Hazardous   : "
            f"{'YES' if result['nasa_hazardous'] else 'NO'}"
        )

        print(
            f"    ARGUS Assessment : "
            f"{result['final_prediction']}"
        )

        print_bullets(
            "Why:",
            [
                reasoning["criteria"][-1],
                reasoning["drivers"][1],
                reasoning["verdict"][0],
            ],
            indent=4
        )

    print("\n==============================================")


# ============================================================
# USER MENU
# ============================================================

def get_user_choice():

    print("\n==============================================")
    print("              ARGUS AI")
    print("==============================================")

    print("\nWhat do you want to check?")

    print("\n1. Show all latest asteroids")
    print("2. Show highest-risk asteroid only")

    while True:

        choice = input(
            "\nEnter your choice (1 or 2): "
        ).strip()

        if choice in ["1", "2"]:
            return choice

        print(
            "Invalid choice. Please enter 1 or 2."
        )


# ============================================================
# MAIN
# ============================================================

def main():

    require_api_key()

    bundle = load_bundle(MODEL_FILE)

    print(
        "\nFetching latest NASA asteroid data..."
    )

    latest_date, asteroids = (
        fetch_latest_asteroids()
    )

    total_asteroids = len(asteroids)

    # --------------------------------------------------------
    # PREDICT EVERY ASTEROID
    # --------------------------------------------------------

    results = []

    for asteroid in asteroids:

        try:

            result = predict_asteroid(
                asteroid,
                bundle
            )

            results.append(result)

        except (KeyError, IndexError, TypeError, ValueError) as error:

            print(
                f"\nSkipping "
                f"{asteroid.get('name', 'Unknown')}: "
                f"{error!r}"
            )

    if not results:

        print(
            "\nNo valid asteroid data available."
        )

        return

    # --------------------------------------------------------
    # SORT BY ML RISK
    # --------------------------------------------------------

    results.sort(
        key=lambda x: x["ml_risk_score"],
        reverse=True
    )

    # --------------------------------------------------------
    # USER CHOICE
    # --------------------------------------------------------

    choice = get_user_choice()

    # --------------------------------------------------------
    # OPTION 1
    # --------------------------------------------------------

    if choice == "1":

        display_multiple(
            results,
            latest_date,
            total_asteroids,
            bundle
        )

    # --------------------------------------------------------
    # OPTION 2
    # --------------------------------------------------------

    elif choice == "2":

        display_single(
            results[0],
            latest_date,
            total_asteroids,
            bundle
        )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()
