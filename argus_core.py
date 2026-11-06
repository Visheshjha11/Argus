"""
argus_core.py

Single source of truth for everything train.py and predict.py must agree on:
feature engineering, model-bundle I/O, and per-asteroid explanations.

The previous code duplicated feature extraction in three files and used
different diameter units in each (km in collect_dataset.py, m in train.py and
predict.py). Anything that touches raw NASA fields now goes through here.
"""

from __future__ import annotations

import itertools
import math
import warnings

import joblib
import numpy as np
import pandas as pd
import sklearn

# ============================================================
# CONSTANTS
# ============================================================

# Canonical raw schema. Units are fixed here and nowhere else.
RAW_COLUMNS = ["diameter_m", "velocity_kmh", "miss_km", "magnitude"]

# Model inputs. Every one is a deterministic function of RAW_COLUMNS.
FEATURES = [
    "magnitude",
    "log_diameter_m",
    "velocity_kms",
    "log_miss_ld",
    "log_energy_mt",
    "log_miss_over_pha",
    "pha_single_pass",
]

LUNAR_DISTANCE_KM = 384_400.0
AU_KM = 149_597_870.7

# NASA/JPL Potentially Hazardous Asteroid definition:
# absolute magnitude H <= 22.0 AND Earth MOID <= 0.05 AU.
PHA_H_MAX = 22.0
PHA_MOID_AU = 0.05
PHA_MOID_KM = PHA_MOID_AU * AU_KM
PHA_MOID_LD = PHA_MOID_KM / LUNAR_DISTANCE_KM
PHA_DIAMETER_BENCHMARK_M = 140.0  # rough diameter equivalent of H = 22

ASSUMED_DENSITY_KG_M3 = 2600.0
MEGATON_TNT_J = 4.184e15

# (label, megatons TNT). Order matters: ascending.
ENERGY_BENCHMARKS = [
    ("the Hiroshima bomb", 0.015),
    ("the 2013 Chelyabinsk airburst", 0.5),
    ("the 1908 Tunguska airburst (~10-15 Mt)", 12.0),
]

# Shapley "players". Diameter is derived from H by NASA, so the two are one
# physical quantity and are perturbed together.
PLAYERS = {
    "Size": ["diameter_m", "magnitude"],
    "Speed": ["velocity_kmh"],
    "Distance": ["miss_km"],
}

BUNDLE_VERSION = 2


# ============================================================
# FEATURE ENGINEERING
# ============================================================

def impact_energy_mt(diameter_m, velocity_kmh):
    """Hypothetical kinetic energy in megatons TNT (sphere, fixed density)."""
    diameter_m = np.asarray(diameter_m, dtype=float)
    mass_kg = (math.pi / 6.0) * diameter_m ** 3 * ASSUMED_DENSITY_KG_M3
    v_ms = np.asarray(velocity_kmh, dtype=float) / 3.6
    return 0.5 * mass_kg * v_ms ** 2 / MEGATON_TNT_J


def featurize(raw: pd.DataFrame) -> pd.DataFrame:
    """RAW_COLUMNS -> FEATURES. Stateless and vectorised."""
    d = raw["diameter_m"].astype(float).clip(lower=1e-3)
    v = raw["velocity_kmh"].astype(float).clip(lower=1e-3)
    miss = raw["miss_km"].astype(float).clip(lower=1.0)

    out = pd.DataFrame(index=raw.index)
    out["magnitude"] = raw["magnitude"].astype(float)
    out["log_diameter_m"] = np.log10(d)
    out["velocity_kms"] = v / 3600.0
    out["log_miss_ld"] = np.log10(miss / LUNAR_DISTANCE_KM)
    out["log_energy_mt"] = np.log10(impact_energy_mt(d, v))

    # Encodes the published PHA definition as seen through ONE observation.
    # miss distance >= MOID always, so (H <= 22 and miss <= 0.05 AU) proves
    # the object is a PHA. This is domain knowledge, not the label: the label
    # also covers objects whose MOID is small but whose pass was distant.
    out["log_miss_over_pha"] = np.log10(miss / PHA_MOID_KM)
    out["pha_single_pass"] = (
        (out["magnitude"] <= PHA_H_MAX) & (miss <= PHA_MOID_KM)
    ).astype(float)
    return out[FEATURES]


# ============================================================
# DATASET LOADING
# ============================================================

def load_dataset(path: str):
    """
    Load the CSV produced by collect_dataset.py.

    Returns (frame, report). frame has RAW_COLUMNS + hazardous + asteroid_id.
    Cleaning performed:
      * rows without asteroid_id are legacy train.py output (diameter in
        metres, no identity) and are dropped; mixing them in corrupts units
      * collect_dataset.py stores diameter in km -> converted to metres
      * non-finite and non-physical rows are dropped
      * repeated (asteroid, date) rows from the collector's resume bug are
        dropped
    """
    df = pd.read_csv(path)
    report = {"rows_raw": int(len(df))}

    required = {"diameter", "velocity", "miss_distance", "magnitude",
                "hazardous", "asteroid_id"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")

    before = len(df)
    df = df[df["asteroid_id"].notna()].copy()
    report["dropped_legacy_rows"] = int(before - len(df))

    df["diameter_m"] = df["diameter"].astype(float) * 1000.0
    df = df.rename(columns={"velocity": "velocity_kmh",
                            "miss_distance": "miss_km"})

    before = len(df)
    df = df.replace([np.inf, -np.inf], np.nan).dropna(
        subset=RAW_COLUMNS + ["hazardous"])
    df = df[(df["diameter_m"] > 0) & (df["velocity_kmh"] > 0)
            & (df["miss_km"] > 0)]
    report["dropped_invalid_rows"] = int(before - len(df))

    before = len(df)
    subset = ["asteroid_id"]
    if "close_approach_date" in df.columns:
        subset.append("close_approach_date")
    df = df.drop_duplicates(subset=subset)
    report["dropped_duplicate_rows"] = int(before - len(df))

    df["hazardous"] = df["hazardous"].astype(int)
    df["asteroid_id"] = df["asteroid_id"].astype("int64")

    # Label must be constant per asteroid or grouped splitting is unsound.
    inconsistent = (df.groupby("asteroid_id")["hazardous"].nunique() > 1).sum()
    report["asteroids_with_inconsistent_label"] = int(inconsistent)

    df = df.reset_index(drop=True)
    report["rows_clean"] = int(len(df))
    report["unique_asteroids"] = int(df["asteroid_id"].nunique())
    report["hazardous_rate"] = float(df["hazardous"].mean())
    if "close_approach_date" in df.columns:
        report["date_min"] = str(df["close_approach_date"].min())
        report["date_max"] = str(df["close_approach_date"].max())

    keep = RAW_COLUMNS + ["hazardous", "asteroid_id"]
    return df[keep + [c for c in ("close_approach_date",) if c in df.columns]], report


# ============================================================
# BUNDLE I/O
# ============================================================

def _minor(version: str) -> str:
    return ".".join(version.split(".")[:2])


def save_bundle(path, model, background, reference, metrics, meta):
    joblib.dump(
        {
            "bundle_version": BUNDLE_VERSION,
            "sklearn_version": sklearn.__version__,
            "features": FEATURES,
            "model": model,
            "background": background,
            "reference": reference,
            "metrics": metrics,
            "meta": meta,
        },
        path,
        compress=3,
    )


def load_bundle(path):
    """
    Pickles execute code on load: only load files you trained yourself.
    Refuses artifacts built by a different scikit-learn minor version rather
    than silently producing wrong probabilities.
    """
    bundle = joblib.load(path)

    if not isinstance(bundle, dict) or bundle.get("bundle_version") != BUNDLE_VERSION:
        raise RuntimeError(
            f"{path} is not an Argus v{BUNDLE_VERSION} bundle. "
            "Run `python train.py` to rebuild it."
        )
    if _minor(bundle["sklearn_version"]) != _minor(sklearn.__version__):
        raise RuntimeError(
            f"{path} was trained with scikit-learn {bundle['sklearn_version']}, "
            f"installed is {sklearn.__version__}. Run `python train.py`."
        )
    if bundle["features"] != FEATURES:
        raise RuntimeError("Feature list changed since training. Retrain.")
    return bundle


def build_reference(raw: pd.DataFrame, n_background: int, seed: int):
    """Population statistics stored with the model for explanations."""
    grid = np.linspace(0, 100, 101)
    reference = {
        "percentile_grid": grid,
        "velocity_kmh": np.percentile(raw["velocity_kmh"], grid),
        "diameter_m": np.percentile(raw["diameter_m"], grid),
    }
    background = (
        raw[RAW_COLUMNS]
        .sample(n=min(n_background, len(raw)), random_state=seed)
        .reset_index(drop=True)
    )
    return background, reference


def percentile_of(value, reference, key) -> float:
    return float(np.interp(value, reference[key], reference["percentile_grid"]))


# ============================================================
# EXPLANATION: EXACT SHAPLEY OVER 3 PHYSICAL INPUT GROUPS
# ============================================================

def shapley_attribution(model, raw_row: dict, background: pd.DataFrame) -> dict:
    """
    Exact Shapley values of the model's hazardous probability, with players
    {Size, Speed, Distance} and marginal imputation from a background sample
    of the training population.

    Values are in probability units and satisfy
        baseline + sum(contributions) == model probability for this object.
    With 3 players that is 8 coalitions, so it is exact, not sampled.
    """
    names = list(PLAYERS)
    n = len(names)
    masks = list(itertools.product([0, 1], repeat=n))

    frames = []
    for mask in masks:
        mixed = background[RAW_COLUMNS].copy()
        for flag, name in zip(mask, names):
            if flag:
                for col in PLAYERS[name]:
                    mixed[col] = raw_row[col]
        frames.append(mixed)

    stacked = pd.concat(frames, ignore_index=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        proba = model.predict_proba(featurize(stacked))[:, 1]
    value = {
        mask: float(proba[i * len(background):(i + 1) * len(background)].mean())
        for i, mask in enumerate(masks)
    }

    contributions = {}
    for i, name in enumerate(names):
        total = 0.0
        for mask in masks:
            if mask[i] == 0:
                k = sum(mask)
                weight = (math.factorial(k) * math.factorial(n - k - 1)
                          / math.factorial(n))
                with_i = list(mask)
                with_i[i] = 1
                total += weight * (value[tuple(with_i)] - value[mask])
        contributions[name] = total

    return {
        "baseline": value[(0,) * n],
        "prediction": value[(1,) * n],
        "contributions": contributions,
    }


# ============================================================
# EXPLANATION: HUMAN-READABLE REASONING
# ============================================================

def _energy_comparison(energy_mt: float) -> str:
    if energy_mt < ENERGY_BENCHMARKS[0][1]:
        frac = energy_mt / ENERGY_BENCHMARKS[0][1]
        return f"{frac:.1%} of {ENERGY_BENCHMARKS[0][0]}"
    label, megatons = ENERGY_BENCHMARKS[0]
    for name, mt in ENERGY_BENCHMARKS:
        if energy_mt >= mt:
            label, megatons = name, mt
    ratio = energy_mt / megatons
    if ratio < 1.5:
        return f"comparable to {label}"
    if ratio >= 100:
        ratio = float(f"{ratio:.2g}")  # two significant figures, no fake precision
        return f"~{ratio:,.0f}x {label}"
    return f"~{ratio:.0f}x {label}" if ratio >= 10 else f"~{ratio:.1f}x {label}"


def build_reasoning(raw: dict, attribution: dict, reference: dict,
                    nasa_hazardous: bool, ml_risk_score: float,
                    diameter_range_m=None) -> dict:
    """
    Turn one asteroid's numbers into sectioned bullet points.

    Every claim is either (a) arithmetic on the observation, (b) a published
    NASA PHA threshold, or (c) the model's own attribution. Nothing is
    narrated that the model did not compute.
    """
    h = raw["magnitude"]
    d = raw["diameter_m"]
    v_kms = raw["velocity_kmh"] / 3600.0
    miss_km = raw["miss_km"]
    miss_ld = miss_km / LUNAR_DISTANCE_KM
    miss_au = miss_km / AU_KM

    size_ok = h <= PHA_H_MAX
    close_ok = miss_km <= PHA_MOID_KM

    # ---- NASA criteria ------------------------------------------------
    criteria = []

    ratio = d / PHA_DIAMETER_BENCHMARK_M
    size_rel = f"{ratio:.1f}x" if ratio >= 1 else f"{ratio:.0%} of"
    spread = ""
    if diameter_range_m:
        spread = f" (albedo-uncertainty range {diameter_range_m[0]:,.0f}-{diameter_range_m[1]:,.0f} m)"
    criteria.append(
        f"Size: H = {h:.2f} is {'at or below' if size_ok else 'above'} NASA's "
        f"PHA cut of H <= {PHA_H_MAX:.0f}; estimated diameter {d:,.0f} m{spread}, "
        f"{size_rel} the ~{PHA_DIAMETER_BENCHMARK_M:.0f} m benchmark."
    )

    if close_ok:
        criteria.append(
            f"Distance: passes at {miss_ld:,.1f} lunar distances ({miss_au:.4f} AU), "
            f"inside the {PHA_MOID_LD:.1f} LD ({PHA_MOID_AU} AU) PHA limit. "
            "A pass this close proves the orbit's MOID is within that limit."
        )
    else:
        criteria.append(
            f"Distance: passes at {miss_ld:,.1f} lunar distances ({miss_au:.4f} AU), "
            f"outside the {PHA_MOID_LD:.1f} LD ({PHA_MOID_AU} AU) PHA limit. "
            "One distant pass does not bound the orbit's MOID, which this model never sees."
        )

    if size_ok and close_ok:
        criteria.append("Result: this single observation satisfies both PHA conditions.")
    elif size_ok:
        criteria.append(
            "Result: large enough to qualify, but this pass is too distant to "
            "confirm the orbital condition. The score reflects how often objects "
            "with this size/speed/distance profile are NASA-flagged."
        )
    elif close_ok:
        criteria.append("Result: close pass, but below the size threshold. Not a PHA by size.")
    else:
        criteria.append("Result: neither PHA condition is met.")

    # ---- Severity context (independent of the PHA label) --------------
    energy_mt = float(impact_energy_mt(d, raw["velocity_kmh"]))
    v_pct = percentile_of(raw["velocity_kmh"], reference, "velocity_kmh")
    d_pct = percentile_of(d, reference, "diameter_m")

    severity = [
        f"Velocity: {v_kms:,.1f} km/s, faster than {v_pct:.0f}% of catalogued "
        "approaches. Speed is not part of NASA's PHA criterion; it scales consequence.",
        f"Size rank: larger than {d_pct:.0f}% of catalogued approaches.",
        f"Hypothetical impact energy: ~{energy_mt:,.3g} Mt TNT, {_energy_comparison(energy_mt)}. "
        f"Assumes {ASSUMED_DENSITY_KG_M3:,.0f} kg/m^3 and no atmospheric breakup.",
    ]

    # ---- Model drivers -------------------------------------------------
    baseline = attribution["baseline"] * 100
    drivers = [
        f"Population baseline risk: {baseline:.1f}%. This object: "
        f"{attribution['prediction'] * 100:.1f}%."
    ]
    ranked = sorted(attribution["contributions"].items(),
                    key=lambda kv: abs(kv[1]), reverse=True)
    for name, contribution in ranked:
        pts = contribution * 100
        if abs(pts) < 0.5:
            drivers.append(f"{name}: negligible effect ({pts:+.1f} pts).")
        else:
            direction = "raises" if pts > 0 else "lowers"
            drivers.append(f"{name} {direction} risk by {abs(pts):.1f} pts.")

    # ---- Verdict --------------------------------------------------------
    if nasa_hazardous and ml_risk_score >= 75:
        verdict = ["NASA and the model agree: high-confidence hazardous."]
    elif nasa_hazardous and ml_risk_score >= 40:
        verdict = ["NASA flags it; the model is moderately supportive. "
                   "NASA's MOID-based call takes precedence."]
    elif nasa_hazardous:
        verdict = ["NASA flags it; the model does not. The size/speed/distance "
                   "profile is atypical for flagged objects, most likely because "
                   "the flag rests on MOID, which this model cannot observe. "
                   "NASA's call takes precedence."]
    elif ml_risk_score >= 75:
        verdict = ["NASA does not flag it, but the model sees a strongly "
                   "PHA-like profile. Likely cause: MOID just outside 0.05 AU or "
                   "recent orbit refinement. Flag for manual review against "
                   "current JPL SBDB orbital elements."]
    elif ml_risk_score >= 40:
        verdict = ["NASA does not flag it; the model sees a partial match. "
                   "Low priority, monitor only."]
    else:
        verdict = ["NASA and the model agree: not hazardous."]

    return {
        "criteria": criteria,
        "severity": severity,
        "drivers": drivers,
        "verdict": verdict,
        "energy_mt": energy_mt,
    }
