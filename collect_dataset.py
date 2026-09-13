import os
import time
import requests
import pandas as pd

from datetime import datetime, timedelta
from dotenv import load_dotenv


# ============================================================
# CONFIGURATION
# ============================================================

load_dotenv()

NASA_API_KEY = os.getenv("NASA_API_KEY")

NASA_API_URL = "https://api.nasa.gov/neo/rest/v1/feed"

OUTPUT_FILE = "nasa_neo_dataset.csv"

TARGET_ROWS = 100000

# Historical period
START_DATE = datetime(2000, 1, 1)
END_DATE = datetime.today()

# NASA allows maximum 7 days for Feed requests
WINDOW_DAYS = 7


# ============================================================
# VALIDATE API KEY
# ============================================================

if not NASA_API_KEY:
    raise ValueError(
        "NASA_API_KEY not found. Add it to your .env file."
    )


# ============================================================
# FETCH ONE WINDOW
# ============================================================

def fetch_window(start_date, end_date):

    params = {
        "start_date": start_date.strftime("%Y-%m-%d"),
        "end_date": end_date.strftime("%Y-%m-%d"),
        "api_key": NASA_API_KEY
    }

    response = requests.get(
        NASA_API_URL,
        params=params,
        timeout=30
    )

    # Rate limit
    if response.status_code == 429:
        print("Rate limit reached. Waiting 60 seconds...")
        time.sleep(60)

        return fetch_window(start_date, end_date)

    response.raise_for_status()


    return response.json()


# ============================================================
# EXTRACT ASTEROID RECORDS
# ============================================================

def extract_records(data):

    records = []

    near_earth_objects = data.get(
        "near_earth_objects",
        {}
    )

    for date, asteroids in near_earth_objects.items():

        for asteroid in asteroids:

            diameter = asteroid.get(
                "estimated_diameter",
                {}
            )

            diameter_km = diameter.get(
                "kilometers",
                {}
            )

            min_diameter = diameter_km.get(
                "estimated_diameter_min"
            )

            max_diameter = diameter_km.get(
                "estimated_diameter_max"
            )

            # Average diameter
            if min_diameter is not None and max_diameter is not None:
                avg_diameter = (
                    min_diameter + max_diameter
                ) / 2
            else:
                avg_diameter = None


            # --------------------------------------------
            # Close approach information
            # --------------------------------------------

            close_approach = asteroid.get(
                "close_approach_data",
                []
            )

            if not close_approach:
                continue

            approach = close_approach[0]

            velocity = approach.get(
                "relative_velocity",
                {}
            )

            miss_distance = approach.get(
                "miss_distance",
                {}
            )


            velocity_kmph = velocity.get(
                "kilometers_per_hour"
            )

            miss_distance_km = miss_distance.get(
                "kilometers"
            )


            # --------------------------------------------
            # Create row
            # --------------------------------------------

            record = {

                "asteroid_id":
                    asteroid.get("id"),

                "name":
                    asteroid.get("name"),

                "diameter":
                    avg_diameter,

                "diameter_min":
                    min_diameter,

                "diameter_max":
                    max_diameter,

                "velocity":
                    velocity_kmph,

                "miss_distance":
                    miss_distance_km,

                "magnitude":
                    asteroid.get(
                        "absolute_magnitude_h"
                    ),


                "hazardous":
                    int(
                        asteroid.get(
                            "is_potentially_hazardous_asteroid",
                            False
                        )
                    ),

                "close_approach_date":
                    date
            }

            records.append(record)

    return records


# ============================================================
# LOAD EXISTING DATA
# ============================================================

if os.path.exists(OUTPUT_FILE):

    existing_df = pd.read_csv(
        OUTPUT_FILE
    )

    all_records = existing_df.to_dict(
        orient="records"
    )

    print(
        f"Existing dataset found: "
        f"{len(all_records):,} rows"
    )

else:

    all_records = []

    print("Starting new dataset...")


# ============================================================
# DOWNLOAD DATA
# ============================================================

current_date = START_DATE

while current_date <= END_DATE:

    if len(all_records) >= TARGET_ROWS:
        break

    window_end = min(
        current_date + timedelta(days=WINDOW_DAYS - 1),
        END_DATE
    )

    print(
        f"\nFetching: "
        f"{current_date.strftime('%Y-%m-%d')} "
        f"→ "
        f"{window_end.strftime('%Y-%m-%d')}"
    )

    try:

        data = fetch_window(
            current_date,
            window_end
        )

        records = extract_records(data)

        all_records.extend(records)

        print(
            f"Records received: {len(records):,}"
        )

        print(
            f"Total dataset: "
            f"{len(all_records):,} / {TARGET_ROWS:,}"
        )

        # Save after every request
        df = pd.DataFrame(all_records)

        df.to_csv(
            OUTPUT_FILE,
            index=False
        )

    except requests.exceptions.RequestException as e:

        print(
            f"Request failed: {e}"
        )

        print("Waiting 10 seconds...")
        time.sleep(10)

        continue

    current_date = (
        window_end +
        timedelta(days=1)
    )

    # Small delay to be polite to the API
    time.sleep(0.5)


# ============================================================
# FINAL DATASET
# ============================================================

df = pd.DataFrame(all_records)

# Remove incomplete records
df = df.dropna(
    subset=[
        "diameter",
        "velocity",
        "miss_distance",
        "magnitude"
    ]
)

# Remove exact duplicate records
df = df.drop_duplicates()

# Keep first TARGET_ROWS
df = df.head(TARGET_ROWS)

# Save
df.to_csv(
    OUTPUT_FILE,
    index=False
)

print("\n========================================")
print("ARGUS DATASET COMPLETE")
print("========================================")

print(
    f"Total rows: {len(df):,}"
)

print(
    f"Saved to: {OUTPUT_FILE}"
)

print("\nColumns:")

print(
    list(df.columns)
)