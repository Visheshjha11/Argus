# train.py
"""
Train the Argus hazard-probability model from the dataset built by
collect_dataset.py. Does NOT call the NASA API and does NOT overwrite the
dataset (the previous version did both).

Method
------
* Model: RandomForestClassifier with the README hyperparameters
  (200 trees, depth 8, min_samples_split 4, balanced class weights).
* Split: 80/20, stratified AND grouped by asteroid_id. The CSV holds many
  close approaches per asteroid and the label is constant per asteroid, so a
  plain random split puts the same object in train and test and inflates
  every metric.
* Probabilities: isotonic calibration on grouped inner folds. class_weight=
  "balanced" otherwise skews predict_proba upward, which would corrupt the
  0-100 risk score and its 40/75 thresholds.
* Reported against two baselines: majority class and the literal PHA rule
  applied to this single observation.
"""

import argparse
import json
import time
import warnings
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import sklearn
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedGroupKFold

from argus_core import (
    FEATURES,
    PHA_H_MAX,
    PHA_MOID_KM,
    build_reference,
    featurize,
    load_dataset,
    save_bundle,
)

DATASET_FILE = "nasa_neo_dataset.csv"
MODEL_FILE = "argus_model.pkl"
METRICS_FILE = "argus_metrics.json"

RF_PARAMS = dict(
    n_estimators=200,
    max_depth=8,
    min_samples_split=4,
    class_weight="balanced",
    n_jobs=-1,
)


# ============================================================
# METRICS
# ============================================================

def expected_calibration_error(y_true, proba, bins=10):
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(proba, edges[1:-1]), 0, bins - 1)
    ece = 0.0
    for b in range(bins):
        mask = idx == b
        if mask.any():
            ece += mask.mean() * abs(y_true[mask].mean() - proba[mask].mean())
    return float(ece)


def evaluate(y_true, proba, threshold=0.5):
    y_true = np.asarray(y_true)
    pred = (proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    return {
        "accuracy": float(accuracy_score(y_true, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, pred)),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "mcc": float(matthews_corrcoef(y_true, pred)),
        "roc_auc": float(roc_auc_score(y_true, proba)),
        "pr_auc": float(average_precision_score(y_true, proba)),
        "brier": float(brier_score_loss(y_true, proba)),
        "ece": expected_calibration_error(y_true, proba),
        "confusion": [[int(tn), int(fp)], [int(fn), int(tp)]],
    }


def new_rf(seed):
    return RandomForestClassifier(random_state=seed, **RF_PARAMS)


# ============================================================
# TRAINING
# ============================================================

def fit_calibrated(X, y, groups, seed, folds=5):
    inner = list(
        StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=seed)
        .split(X, y, groups)
    )
    model = CalibratedClassifierCV(new_rf(seed), method="isotonic", cv=inner)
    model.fit(X, y)
    return model


def feature_importance(model):
    parts = [c.estimator.feature_importances_ for c in model.calibrated_classifiers_]
    mean = np.mean(parts, axis=0)
    return dict(sorted(zip(FEATURES, map(float, mean)), key=lambda kv: -kv[1]))


def banner(title):
    print("\n" + "=" * 62)
    print(title)
    print("=" * 62)


def main():
    parser = argparse.ArgumentParser(description="Train the Argus model.")
    parser.add_argument("--data", default=DATASET_FILE)
    parser.add_argument("--out", default=MODEL_FILE)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cv-folds", type=int, default=5)
    args = parser.parse_args()

    started = time.time()
    warnings.filterwarnings("ignore", category=UserWarning)

    # ---------------- data ----------------
    raw, report = load_dataset(args.data)
    if report["rows_clean"] < 1000 or raw["hazardous"].sum() < 50:
        raise SystemExit(
            "Dataset too small to train or evaluate honestly. "
            "Run collect_dataset.py first."
        )

    banner("ARGUS AI DATASET")
    print(f"Rows in CSV            : {report['rows_raw']:,}")
    print(f"Dropped legacy rows    : {report['dropped_legacy_rows']:,}")
    print(f"Dropped invalid rows   : {report['dropped_invalid_rows']:,}")
    print(f"Dropped duplicate rows : {report['dropped_duplicate_rows']:,}")
    print(f"Rows used              : {report['rows_clean']:,}")
    print(f"Unique asteroids       : {report['unique_asteroids']:,}")
    if "date_min" in report:
        print(f"Date range             : {report['date_min']} -> {report['date_max']}")
    print(f"Hazardous rate         : {report['hazardous_rate']:.2%}")
    if report["asteroids_with_inconsistent_label"]:
        print(f"WARNING: {report['asteroids_with_inconsistent_label']} asteroids "
              "have inconsistent labels across rows.")

    X = featurize(raw)
    y = raw["hazardous"].to_numpy()
    groups = raw["asteroid_id"].to_numpy()

    # ---------------- 80/20 grouped + stratified split ----------------
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=args.seed)
    train_idx, test_idx = next(splitter.split(X, y, groups))
    X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
    y_tr, y_te = y[train_idx], y[test_idx]
    g_tr = groups[train_idx]
    assert not set(groups[train_idx]) & set(groups[test_idx]), "group leakage"

    print(f"\nTrain: {len(train_idx):,} rows / {len(set(g_tr)):,} asteroids")
    print(f"Test : {len(test_idx):,} rows / {len(set(groups[test_idx])):,} asteroids "
          "(no asteroid appears in both)")

    # ---------------- grouped CV on the training portion ----------------
    cv_scores = {"roc_auc": [], "pr_auc": [], "f1": [], "recall": []}
    cv = StratifiedGroupKFold(n_splits=args.cv_folds, shuffle=True,
                              random_state=args.seed)
    for fold, (a, b) in enumerate(cv.split(X_tr, y_tr, g_tr), start=1):
        rf = new_rf(args.seed).fit(X_tr.iloc[a], y_tr[a])
        p = rf.predict_proba(X_tr.iloc[b])[:, 1]
        m = evaluate(y_tr[b], p)
        for k in cv_scores:
            cv_scores[k].append(m[k])
        print(f"  CV fold {fold}: AUC {m['roc_auc']:.4f}  F1 {m['f1']:.4f}  "
              f"recall {m['recall']:.4f}")
    cv_summary = {k: {"mean": float(np.mean(v)), "std": float(np.std(v))}
                  for k, v in cv_scores.items()}

    # ---------------- holdout evaluation ----------------
    holdout_model = fit_calibrated(X_tr, y_tr, g_tr, args.seed)
    proba = holdout_model.predict_proba(X_te)[:, 1]
    test_metrics = evaluate(y_te, proba)

    # ---------------- baselines ----------------
    majority = np.zeros_like(y_te)
    raw_te = raw.iloc[test_idx]
    rule = ((raw_te["magnitude"] <= PHA_H_MAX)
            & (raw_te["miss_km"] <= PHA_MOID_KM)).astype(int).to_numpy()
    baselines = {
        "majority_class": {
            "accuracy": float(accuracy_score(y_te, majority)),
            "recall": 0.0,
        },
        "single_pass_pha_rule": {
            "accuracy": float(accuracy_score(y_te, rule)),
            "precision": float(precision_score(y_te, rule, zero_division=0)),
            "recall": float(recall_score(y_te, rule, zero_division=0)),
            "f1": float(f1_score(y_te, rule, zero_division=0)),
        },
    }

    # ---------------- operating-point sweep ----------------
    # Reported, not tuned: nothing is selected on the test set.
    sweep = []
    for thr in (0.10, 0.20, 0.30, 0.40, 0.50):
        pred = (proba >= thr).astype(int)
        sweep.append({
            "threshold": thr,
            "flagged": int(pred.sum()),
            "precision": float(precision_score(y_te, pred, zero_division=0)),
            "recall": float(recall_score(y_te, pred, zero_division=0)),
            "f1": float(f1_score(y_te, pred, zero_division=0)),
        })

    # ---------------- risk-band diagnostics ----------------
    score = proba * 100
    bands = {}
    for name, lo, hi in [("LOW (<40)", 0, 40), ("MEDIUM (40-74.99)", 40, 75),
                         ("HIGH (>=75)", 75, 100.01)]:
        mask = (score >= lo) & (score < hi)
        bands[name] = {
            "count": int(mask.sum()),
            "observed_hazardous_rate": float(y_te[mask].mean()) if mask.any() else None,
            "mean_predicted": float(score[mask].mean() / 100) if mask.any() else None,
        }

    # ---------------- print report ----------------
    banner("ARGUS AI MODEL EVALUATION (held-out asteroids)")
    t = test_metrics
    print(f"\nAccuracy          : {t['accuracy'] * 100:.2f}%")
    print(f"Balanced accuracy : {t['balanced_accuracy'] * 100:.2f}%")
    print(f"Precision         : {t['precision']:.4f}")
    print(f"Recall            : {t['recall']:.4f}")
    print(f"F1                : {t['f1']:.4f}")
    print(f"MCC               : {t['mcc']:.4f}")
    print(f"ROC-AUC           : {t['roc_auc']:.4f}")
    print(f"PR-AUC            : {t['pr_auc']:.4f}   (chance = {y_te.mean():.4f})")
    print(f"Brier score       : {t['brier']:.4f}")
    print(f"Calibration error : {t['ece']:.4f}")

    print("\nClassification Report:")
    print(classification_report(
        y_te, (proba >= 0.5).astype(int),
        target_names=["Safe", "Hazardous"], zero_division=0, digits=4))

    print("Confusion Matrix:  [[TN FP]\n                    [FN TP]]")
    print(np.array(t["confusion"]))

    print("\nGrouped 5-fold CV on training portion (uncalibrated RF):")
    for k, v in cv_summary.items():
        print(f"  {k:8s}: {v['mean']:.4f} +/- {v['std']:.4f}")

    print("\nBaselines on the same test set:")
    print(f"  Majority class        : accuracy {baselines['majority_class']['accuracy'] * 100:.2f}%, recall 0.00")
    b = baselines["single_pass_pha_rule"]
    print(f"  Single-pass PHA rule  : accuracy {b['accuracy'] * 100:.2f}%, "
          f"precision {b['precision']:.4f}, recall {b['recall']:.4f}")

    print("\nOperating points (calibrated probability threshold):")
    print("  threshold  flagged  precision  recall      F1")
    for row in sweep:
        print(f"  {row['threshold']:9.2f}  {row['flagged']:7,d}  "
              f"{row['precision']:9.4f}  {row['recall']:6.4f}  {row['f1']:6.4f}")

    print("\nRisk bands (does the score mean what it says?):")
    for name, s in bands.items():
        if s["count"]:
            print(f"  {name:18s} n={s['count']:6,d}  "
                  f"mean predicted {s['mean_predicted']:.3f}  "
                  f"observed hazardous {s['observed_hazardous_rate']:.3f}")
        else:
            print(f"  {name:18s} n=0")

    # ---------------- final model: refit on everything ----------------
    final_model = fit_calibrated(X, y, groups, args.seed)
    importance = feature_importance(final_model)

    print("\nFeature importance (impurity, final model):")
    for name, value in importance.items():
        print(f"  {name:16s} {value:.4f}")

    background, reference = build_reference(raw, n_background=100, seed=args.seed)

    metrics = {
        "holdout": test_metrics,
        "grouped_cv": cv_summary,
        "baselines": baselines,
        "operating_points": sweep,
        "risk_bands": bands,
        "feature_importance": importance,
    }
    meta = {
        "trained_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "sklearn_version": sklearn.__version__,
        "rows": report["rows_clean"],
        "asteroids": report["unique_asteroids"],
        "hazardous_rate": report["hazardous_rate"],
        "rf_params": {k: v for k, v in RF_PARAMS.items() if k != "n_jobs"},
        "seed": args.seed,
        "note": ("Metrics come from the 80% model evaluated on 20% unseen "
                 "asteroids; the shipped model is refit on 100% of the data."),
    }
    save_bundle(args.out, final_model, background, reference, metrics, meta)
    with open(METRICS_FILE, "w") as handle:
        json.dump({"meta": meta, "data_report": report, **metrics}, handle, indent=2)

    print(f"\nModel saved   : {args.out}")
    print(f"Metrics saved : {METRICS_FILE}")
    print(f"Elapsed       : {time.time() - started:.1f}s")
    print("\nTraining complete.")


if __name__ == "__main__":
    main()