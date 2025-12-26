"""
Train dropout-risk models using a leakage-safe, deployment-ready sklearn Pipeline.

Key fixes vs your current version:
- No SMOTE on categorical-coded features (avoids synthetic fractional categories)
- OneHotEncode categorical features (prevents fake ordinal meaning)
- Hyperparameter tuning happens inside a Pipeline (no CV leakage)
- Saves the full pipeline (preprocessing + model) for deployment
- Optionally selects a probability threshold on a validation split
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import json

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    precision_recall_curve,
)
from sklearn.ensemble import RandomForestClassifier

try:
    from xgboost import XGBClassifier
except Exception as e:
    raise RuntimeError(
        "xgboost is required. Install with: pip install xgboost"
    ) from e

import joblib


DEFAULT_CONTINUOUS = {
    "Previous qualification (grade)",
    "Admission grade",
    "Age at enrollment",
    "Curricular units 1st sem (credited)",
    "Curricular units 1st sem (enrolled)",
    "Curricular units 1st sem (evaluations)",
    "Curricular units 1st sem (approved)",
    "Curricular units 1st sem (grade)",
    "Curricular units 1st sem (without evaluations)",
    "Unemployment rate",
    "Inflation rate",
    "GDP",
}

# If you're predicting earlier than end-of-year, 2nd semester features are "future info"
LEAKAGE_PATTERN = "2nd sem"


@dataclass(frozen=True)
class FeatureGroups:
    categorical: list[str]
    numeric: list[str]
    dropped_leakage: list[str]
    
    # Adding a __str__ method for cleaner printing if needed
    def __repr__(self):
        return f"FeatureGroups(cat={len(self.categorical)}, num={len(self.numeric)}, dropped_leakage={len(self.dropped_leakage)})"


def load_data(csv_path: str | Path, sep: str = ";") -> pd.DataFrame:
    df = pd.read_csv(csv_path, sep=sep)

    # Clean column names
    df.columns = (
        df.columns.astype(str)
        .str.replace("\t", "", regex=False)
        .str.strip()
    )

    # Fix known typo
    df = df.rename(columns={"Nacionality": "Nationality"})

    return df


def make_binary_target(df: pd.DataFrame, target_col: str = "Target") -> pd.Series:
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in data.")
    return df[target_col].apply(lambda x: 1 if x == "Dropout" else 0).astype(int)


def drop_leakage_columns(df: pd.DataFrame, leakage_pattern: str = LEAKAGE_PATTERN) -> tuple[pd.DataFrame, list[str]]:
    leakage_cols = [c for c in df.columns if leakage_pattern in c]
    kept = df.drop(columns=leakage_cols)
    return kept, leakage_cols


def infer_feature_groups(X: pd.DataFrame) -> FeatureGroups:
    # Identify numeric/categorical columns
    # Strategy:
    # - Treat known continuous vars as numeric
    # - Treat low/medium-cardinality integer-coded columns as categorical (codes)
    # - Everything else numeric
    categorical_cols: list[str] = []
    numeric_cols: list[str] = []

    for c in X.columns:
        if c in DEFAULT_CONTINUOUS:
            numeric_cols.append(c)
            continue

        s = X[c]
        if pd.api.types.is_numeric_dtype(s):
            nunique = s.nunique(dropna=True)

            # Many columns are integer-coded categories (course, nationality, etc.)
            # If it looks like a code and not a continuous measure, one-hot it.
            # Using a simplified heuristic: if nunique <= 60 and integer-like
            looks_integer = pd.api.types.is_integer_dtype(s) or np.all(np.isclose(s.dropna().values, np.round(s.dropna().values)))
            if looks_integer and nunique <= 60 and c not in {"Application order"}:
                categorical_cols.append(c)
            else:
                numeric_cols.append(c)
        else:
            categorical_cols.append(c)

    return FeatureGroups(categorical=categorical_cols, numeric=numeric_cols, dropped_leakage=[])


def build_preprocessor(groups: FeatureGroups) -> ColumnTransformer:
    cat_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    num_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("cat", cat_pipe, groups.categorical),
            ("num", num_pipe, groups.numeric),
        ],
        remainder="drop",
    )


def choose_threshold_on_validation(y_true: np.ndarray, y_prob: np.ndarray, objective: str = "f1") -> tuple[float, dict]:
    """
    Choose a decision threshold using validation data ONLY (not test).
    objective:
      - "f1": maximize F1
      - "recall>=0.80": maximize precision subject to recall>=0.80
    """
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    # thresholds has length n-1, precision/recall length n
    thresholds = np.append(thresholds, 1.0)

    eps = 1e-12
    f1 = 2 * precision * recall / (precision + recall + eps)

    if objective == "f1":
        i = int(np.nanargmax(f1))
        return float(thresholds[i]), {"precision": float(precision[i]), "recall": float(recall[i]), "f1": float(f1[i])}

    if objective.startswith("recall>="):
        target_recall = float(objective.split(">=")[1])
        ok = recall >= target_recall
        if not np.any(ok):
            # fall back to max recall
            i = int(np.nanargmax(recall))
            return float(thresholds[i]), {"precision": float(precision[i]), "recall": float(recall[i]), "f1": float(f1[i])}
        # among ok, maximize precision
        i = int(np.nanargmax(np.where(ok, precision, -1)))
        return float(thresholds[i]), {"precision": float(precision[i]), "recall": float(recall[i]), "f1": float(f1[i])}

    raise ValueError(f"Unknown threshold objective: {objective}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="data/data.csv", help="Path to CSV file (semicolon-separated).")
    parser.add_argument("--outdir", type=str, default="models", help="Directory to save trained artifacts.")
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--random_state", type=int, default=42)
    # Changed default: NOW DROPS LEAKAGE BY DEFAULT. Use --keep_leakage to keep it.
    parser.add_argument("--keep_leakage", action="store_true", help="Keep '2nd sem' columns (NOT recommended for early warning).")
    parser.add_argument("--threshold_objective", type=str, default="f1", help='e.g. "f1" or "recall>=0.80"')
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Resolve data path relative to script or current dir
    data_path = Path(args.data)
    if not data_path.exists():
        # Try finding it relative to project root if running from src/
        possible_path = Path("..") / args.data
        if possible_path.exists():
           data_path = possible_path

    if not data_path.exists():
        raise FileNotFoundError(f"Could not find data file at {args.data} or {data_path.resolve()}")

    print(f"Loading data from: {data_path}")
    df = load_data(data_path)

    # Target
    y = make_binary_target(df, target_col="Target")

    # Drop leakage columns if requested (DEFAULT IS TO DROP)
    df_X = df.drop(columns=["Target"])
    dropped_leakage: list[str] = []
    
    if not args.keep_leakage:
        print("Dropping 2nd Semester columns (Leakage Prevention)...")
        df_X, dropped_leakage = drop_leakage_columns(df_X, leakage_pattern=LEAKAGE_PATTERN)
    else:
        print("WARNING: Keeping 2nd Semester columns. This may cause data leakage!")

    # Feature groups
    groups = infer_feature_groups(df_X)
    groups = FeatureGroups(categorical=groups.categorical, numeric=groups.numeric, dropped_leakage=dropped_leakage)
    print(f"Feature Groups: {groups}")

    X_train, X_test, y_train, y_test = train_test_split(
        df_X, y, test_size=args.test_size, random_state=args.random_state, stratify=y
    )

    # Validation split for threshold selection (avoid using test set for threshold choice)
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=args.random_state, stratify=y_train
    )

    preprocessor = build_preprocessor(groups)

    # ---- Random Forest pipeline + tuning ----
    # EXPLICITLY SET n_jobs=1 to avoid issues in this env
    rf = RandomForestClassifier(
        random_state=args.random_state,
        class_weight="balanced",
        n_jobs=1,
    )

    rf_pipe = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("model", rf),
        ]
    )

    rf_param_dist = {
        "model__n_estimators": [100, 200, 300], # reduced range for speed/safety
        "model__max_depth": [None, 10, 20],
        "model__min_samples_split": [2, 5],
        "model__min_samples_leaf": [1, 2],
        "model__bootstrap": [True, False],
    }

    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=args.random_state)
    
    # EXPLICITLY SET n_jobs=1
    rf_search = RandomizedSearchCV(
        rf_pipe,
        param_distributions=rf_param_dist,
        n_iter=10, # Reduced iter to 10 for safety
        scoring="f1",
        cv=cv,
        verbose=1,
        n_jobs=1,
        random_state=args.random_state,
    )

    print("Tuning Random Forest...")
    rf_search.fit(X_tr, y_tr)
    rf_best = rf_search.best_estimator_
    print("RF best params:", rf_search.best_params_)

    # ---- XGBoost pipeline ----
    # Compute scale_pos_weight on the *original* training labels (no SMOTE here)
    pos = int(y_tr.sum())
    neg = int(len(y_tr) - pos)
    spw = (neg / max(pos, 1))

    # EXPLICITLY SET n_jobs=1
    xgb = XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=args.random_state,
        n_jobs=1,
        eval_metric="logloss",
        scale_pos_weight=spw,
    )

    xgb_pipe = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("model", xgb),
        ]
    )

    print("Training XGBoost...")
    xgb_pipe.fit(X_tr, y_tr)

    # ---- Threshold selection on validation (using XGBoost probs) ----
    xgb_val_prob = xgb_pipe.predict_proba(X_val)[:, 1]
    best_thresh, thresh_metrics = choose_threshold_on_validation(
        y_val.to_numpy(), xgb_val_prob, objective=args.threshold_objective
    )
    print(f"Chosen threshold (XGBoost, objective={args.threshold_objective}): {best_thresh:.3f} -> {thresh_metrics}")

    # ---- Evaluate on test (no threshold tuning here) ----
    def eval_model(name: str, model: Pipeline):
        prob = model.predict_proba(X_test)[:, 1]
        pred_default = (prob >= 0.5).astype(int)
        pred_thresh = (prob >= best_thresh).astype(int)

        print(f"\n=== {name} (threshold=0.50) ===")
        print(classification_report(y_test, pred_default, target_names=["Non-Dropout", "Dropout"]))
        print("ROC-AUC:", roc_auc_score(y_test, prob))

        print(f"\n=== {name} (threshold={best_thresh:.3f}) ===")
        print(classification_report(y_test, pred_thresh, target_names=["Non-Dropout", "Dropout"]))

    eval_model("RandomForest", rf_best)
    eval_model("XGBoost", xgb_pipe)

    # ---- Refit on full training set (train+val) for final artifacts ----
    print("Refitting on full training set...")
    rf_best.fit(X_train, y_train)
    xgb_pipe.fit(X_train, y_train)

    # Save artifacts
    joblib.dump(rf_best, outdir / "rf_pipeline.joblib")
    joblib.dump(xgb_pipe, outdir / "xgb_pipeline.joblib")

    meta = {
        "target": {"positive_class": "Dropout", "mapping": {"Dropout": 1, "Graduate": 0, "Enrolled": 0}},
        "drop_2nd_sem": not args.keep_leakage,
        "dropped_leakage_cols": groups.dropped_leakage,
        "categorical_cols": groups.categorical,
        "numeric_cols": groups.numeric,
        "xgb_scale_pos_weight": float(spw),
        "chosen_threshold": float(best_thresh),
        "threshold_objective": args.threshold_objective,
        "threshold_metrics_on_val": thresh_metrics,
        "random_state": args.random_state,
        "test_size": float(args.test_size),
    }
    (outdir / "metadata.json").write_text(json.dumps(meta, indent=2))
    print(f"\nSaved pipelines + metadata to: {outdir.resolve()}")


if __name__ == "__main__":
    main()
