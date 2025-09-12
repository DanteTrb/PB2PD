#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Train RandomForest (balanced) per Triade + salva modello, scaler e soglie.
Usage:
  python scripts/train_model.py \
      --train data/processed/train_balanced_ctgan.csv \
      --test  data/processed/test_original.csv
"""

import os, json, argparse
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score, average_precision_score, brier_score_loss,
    precision_score, recall_score, f1_score
)
import joblib

# ----- Config -----
FEATURES = [
    "MSE ML","iHR V","MSE V","MSE AP","Weigth","Age",
    "Sex (M=1, F=2)","H-Y","Gait Speed","Duration (years)"
]
TARGET = "target_bin"
NUM_COLS = ["MSE ML","iHR V","MSE V","MSE AP","Weigth","Age","Gait Speed","Duration (years)"]
THRESHOLDS = {"thr_green": 0.26, "thr_yellow": 0.40, "thr_orange": 0.50}

MODELS_DIR  = "models"
RESULTS_DIR = "results"

# ------------------

def metrics_at_threshold(y_true, proba, thr):
    pred = (proba >= thr).astype(int)
    return dict(
        precision=precision_score(y_true, pred, zero_division=0),
        recall=recall_score(y_true, pred, zero_division=0),
        f1=f1_score(y_true, pred, zero_division=0),
        brier=brier_score_loss(y_true, proba),
    )

def main(args):
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # --- load data
    train = pd.read_csv(args.train)
    test  = pd.read_csv(args.test)

    # --- split
    X_train = train[FEATURES].copy()
    y_train = train[TARGET].astype(int).values

    X_test  = test[FEATURES].copy()
    y_test  = test[TARGET].astype(int).values

    # --- scaler (solo colonne numeriche continue)
    scaler = StandardScaler().fit(X_train[NUM_COLS])

    X_train_s = X_train.copy()
    X_test_s  = X_test.copy()
    X_train_s[NUM_COLS] = scaler.transform(X_train[NUM_COLS])
    X_test_s[NUM_COLS]  = scaler.transform(X_test[NUM_COLS])

    # --- model
    rf = RandomForestClassifier(
        n_estimators=800,
        min_samples_leaf=3,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )
    rf.fit(X_train_s, y_train)

    # --- eval su test
    proba_test = rf.predict_proba(X_test_s)[:, 1]
    roc = float(roc_auc_score(y_test, proba_test))
    pr  = float(average_precision_score(y_test, proba_test))
    brier = float(brier_score_loss(y_test, proba_test))

    print("\n=== Test metrics ===")
    print(f"ROC_AUC : {roc:.3f}")
    print(f"PR_AUC  : {pr:.3f}")
    print(f"Brier   : {brier:.3f}")

    # --- metrics at key thresholds
    thr_metrics = {}
    for name, thr in THRESHOLDS.items():
        thr_metrics[name] = {**{"thr": thr}, **metrics_at_threshold(y_test, proba_test, thr)}

    # --- save artifacts
    model_path  = os.path.join(MODELS_DIR, "rf_model.pkl")
    scaler_path = os.path.join(MODELS_DIR, "scaler.pkl")
    thr_path    = os.path.join(MODELS_DIR, "thresholds.json")
    metrics_path= os.path.join(RESULTS_DIR, "train_metrics.json")

    joblib.dump(rf, model_path)
    joblib.dump(scaler, scaler_path)
    with open(thr_path, "w") as f:
        json.dump(THRESHOLDS, f, indent=2)

    with open(metrics_path, "w") as f:
        json.dump({
            "ROC_AUC": roc,
            "PR_AUC": pr,
            "Brier": brier,
            "threshold_metrics": thr_metrics
        }, f, indent=2)

    print(f"\n✅ Salvato modello in '{model_path}'")
    print(f"✅ Salvato scaler in '{scaler_path}'")
    print(f"✅ Salvate soglie in '{thr_path}'")
    print(f"✅ Metriche in '{metrics_path}'\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", required=True, help="Path CSV train")
    parser.add_argument("--test",  required=True, help="Path CSV test")
    args = parser.parse_args()
    main(args)