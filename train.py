# train.py

import os
import requests
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("NASA_API_KEY")
# ============================================================
# CONFIGURATION
# ============================================================

if not API_KEY:
    raise ValueError(
        "NASA_API_KEY environment variable is not set."
    )

START_DATE = "2026-08-01"
END_DATE = "2026-08-07"

MODEL_FILE = "argus_model.pkl"
DATASET_FILE = "nasa_neo_dataset.csv"


# ============================================================
# 1. FETCH NASA NEO DATA
# ============================================================

def fetch_neo_data():

    url = "https://api.nasa.gov/neo/rest/v1/feed"

    params = {
        "start_date": START_DATE,
        "end_date": END_DATE,
        "api_key": API_KEY
    }

    response = requests.get(url, params=params, timeout=30)

    response.raise_for_status()

    data = response.json()

    return data["near_earth_objects"]


# ============================================================
# 2. CONVERT NASA RESPONSE INTO DATAFRAME
# ============================================================

def create_dataset(neo_data):

    rows = []

    for date, asteroids in neo_data.items():

        for asteroid in asteroids:

            # Some objects may not have close approach data
            if not asteroid["close_approach_data"]:
                continue

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

            diameter = (diameter_min + diameter_max) / 2

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

            hazardous = int(
                asteroid["is_potentially_hazardous_asteroid"]
            )

            rows.append({

                "diameter": diameter,

                "velocity": velocity,

                "miss_distance": miss_distance,

                "magnitude": magnitude,

                "hazardous": hazardous
            })

    return pd.DataFrame(rows)


# ============================================================
# 3. TRAIN MODEL
# ============================================================

def train_model(df):

    features = [
        "diameter",
        "velocity",
        "miss_distance",
        "magnitude"
    ]

    X = df[features]

    y = df["hazardous"]

    # 80% training
    # 20% testing
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=8,
        min_samples_split=4,
        random_state=42,
        class_weight="balanced"
    )

    model.fit(X_train, y_train)

    # ========================================================
    # EVALUATION
    # ========================================================

    predictions = model.predict(X_test)

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    print("\n==============================")
    print("ARGUS AI MODEL EVALUATION")
    print("==============================")

    print(
        f"\nAccuracy: {accuracy * 100:.2f}%"
    )

    print("\nClassification Report:")
    print(
        classification_report(
            y_test,
            predictions,
            target_names=[
                "Safe",
                "Hazardous"
            ],
            zero_division=0
        )
    )

    print("\nConfusion Matrix:")
    print(
        confusion_matrix(
            y_test,
            predictions
        )
    )

    return model


# ============================================================
# 4. MAIN
# ============================================================

def main():

    print("Fetching NASA NEO data...")

    neo_data = fetch_neo_data()

    print("Creating dataset...")

    df = create_dataset(neo_data)

    if len(df) < 20:
        raise ValueError(
            "Not enough asteroid data for training."
        )

    print(
        f"Dataset size: {len(df)} asteroids"
    )

    print("\nClass distribution:")
    print(
        df["hazardous"].value_counts()
    )

    # Save dataset
    df.to_csv(
        DATASET_FILE,
        index=False
    )

    print(
        f"\nDataset saved as {DATASET_FILE}"
    )

    # Train
    model = train_model(df)
    

    # Save model
    joblib.dump(
        model,
        MODEL_FILE
    )

    print(
        f"\nModel saved as {MODEL_FILE}"
    )

    print("\nTraining complete.")


if __name__ == "__main__":
    main()