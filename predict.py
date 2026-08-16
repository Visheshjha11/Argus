# predict.py

import os
import requests
import joblib
import pandas as pd

from datetime import datetime, timedelta
from dotenv import load_dotenv


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_FILE = "argus_model.pkl"
NASA_API_URL = "https://api.nasa.gov/neo/rest/v1/feed"

MAX_DISPLAY = 10


# ============================================================
# LOAD API KEY
# ============================================================

load_dotenv()

API_KEY = os.getenv("NASA_API_KEY")

if not API_KEY:
    raise ValueError(
        "NASA_API_KEY is not set in the .env file."
    )


# ============================================================
# LOAD TRAINED MODEL
# ============================================================

model = joblib.load(MODEL_FILE)


# ============================================================
# FETCH LATEST AVAILABLE NASA DATA
# ============================================================

def fetch_latest_asteroids():

    today = datetime.utcnow().date()

    for days_back in range(7):

        target_date = today - timedelta(days=days_back)

        date_string = target_date.strftime("%Y-%m-%d")

        params = {
            "start_date": date_string,
            "end_date": date_string,
            "api_key": API_KEY
        }

        response = requests.get(
            NASA_API_URL,
            params=params,
            timeout=30
        )

        response.raise_for_status()

        data = response.json()

        neo_data = data.get(
            "near_earth_objects",
            {}
        )

        if date_string in neo_data:

            asteroids = neo_data[date_string]

            if asteroids:
                return date_string, asteroids

    raise RuntimeError(
        "No NASA asteroid data found in the last 7 days."
    )


# ============================================================
# EXTRACT FEATURES
# ============================================================

def extract_features(asteroid):

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

    diameter = (
        diameter_min + diameter_max
    ) / 2

    velocity = float(
        approach["relative_velocity"]
        ["kilometers_per_hour"]
    )

    miss_distance = float(
        approach["miss_distance"]
        ["kilometers"]
    )

    magnitude = float(
        asteroid["absolute_magnitude_h"]
    )

    return {
        "diameter": diameter,
        "velocity": velocity,
        "miss_distance": miss_distance,
        "magnitude": magnitude
    }


# ============================================================
# PREDICT ONE ASTEROID
# ============================================================

def predict_asteroid(asteroid):

    values = extract_features(asteroid)

    features = pd.DataFrame({
        "diameter": [values["diameter"]],
        "velocity": [values["velocity"]],
        "miss_distance": [values["miss_distance"]],
        "magnitude": [values["magnitude"]]
    })

    # --------------------------------------------------------
    # ML PREDICTION
    # --------------------------------------------------------

    prediction = model.predict(features)[0]

    probabilities = model.predict_proba(features)[0]

    safe_probability = probabilities[0]

    hazardous_probability = probabilities[1]

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

        "diameter": values["diameter"],

        "velocity": values["velocity"],

        "miss_distance": values["miss_distance"],

        "magnitude": values["magnitude"],

        "ml_risk_score": ml_risk_score,

        "threat_level": threat_level,

        "prediction": prediction,

        "safe_probability": safe_probability,

        "hazardous_probability": hazardous_probability,

        "nasa_hazardous": nasa_hazardous,

        "final_prediction": final_prediction
    }


# ============================================================
# DISPLAY SINGLE ASTEROID
# ============================================================

def display_single(result, latest_date, total):

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

    print("\n==============================================")


# ============================================================
# DISPLAY MULTIPLE ASTEROIDS
# ============================================================

def display_multiple(results, latest_date, total):

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
                asteroid
            )

            results.append(result)

        except Exception as error:

            print(
                f"\nSkipping "
                f"{asteroid.get('name', 'Unknown')}: "
                f"{error}"
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
            total_asteroids
        )

    # --------------------------------------------------------
    # OPTION 2
    # --------------------------------------------------------

    elif choice == "2":

        display_single(
            results[0],
            latest_date,
            total_asteroids
        )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()