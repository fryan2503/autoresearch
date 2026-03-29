"""
train_xgb.py
============
LendingClub loan-default classification pipeline with XGBoost.
Loads raw CSVs -> cleans -> preprocesses -> runs random search -> generates a Kaggle submission.

Run:
    python train_xgb.py
"""

import random
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import xgboost as xgb
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder

from utils import emp_title_cleaning

# Reproducibility
np.random.seed(1809)
random.seed(1809)

# Device
# XGBoost uses CUDA on NVIDIA GPUs and CPU otherwise. MPS is not supported.
if torch.cuda.is_available():
    xgb_device = "cuda"
    print("✅  CUDA — NVIDIA GPU")
else:
    xgb_device = "cpu"
    print("⚠️   CPU only")

# Paths
ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
SUB_DIR = DATA_DIR / "kaggle_submissions"
RESULTS_DIR = ROOT / "model_results"
SUB_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
TODAY = datetime.now().strftime("%m-%d")

# =============================================================================
# 1. LOAD
# =============================================================================
print("\n[1/5] Loading data ...")

df_train = pd.read_csv(DATA_DIR / "train.csv")
df_holdout = pd.read_csv(DATA_DIR / "holdout.csv")
df_score = pd.read_csv(DATA_DIR / "score.csv")
print(f"  train={df_train.shape}  holdout={df_holdout.shape}  score={df_score.shape}")

# =============================================================================
# 2. CLEAN & ENGINEER FEATURES
# =============================================================================
print("\n[2/5] Cleaning ...")

df_train["mort_acc"] = pd.to_numeric(df_train["mort_acc"], errors="coerce")
df_train["mths_since_last_delinq"] = pd.to_numeric(df_train["mths_since_last_delinq"], errors="coerce")
df_train["revol_util"] = pd.to_numeric(df_train["revol_util"], errors="coerce")

mean_mort_acc_by_purpose = df_train.groupby("purpose", observed=True)["mort_acc"].mean()
median_mths_delinq = df_train["mths_since_last_delinq"].median()
earliest_cr_line_ref = pd.to_datetime(df_train["earliest_cr_line"], format="%b-%Y").min()

frames = {
    "train": df_train,
    "holdout": df_holdout,
    "score": df_score,
}

for split_name, df in frames.items():
    df = df.copy()

    for col in ("mort_acc", "mths_since_last_delinq", "revol_util"):
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["emp_length"] = df["emp_length"].astype("object")

    if split_name != "score":
        df["loan_default"] = df["loan_default"].map({"Yes": 1, "No": 0}).astype(np.int64)

    df["missing_mort_acc"] = df["mort_acc"].isna().astype(int)
    df["missing_mths_since_last_delinq"] = df["mths_since_last_delinq"].isna().astype(int)
    df["emp_title_missing"] = df["emp_title"].isna().astype(int)

    df["mort_acc"] = np.where(
        df["mort_acc"].isna(),
        df["purpose"].map(mean_mort_acc_by_purpose),
        df["mort_acc"],
    )
    df["mths_since_last_delinq"] = df["mths_since_last_delinq"].fillna(median_mths_delinq)
    df["pub_rec_bankruptcies"] = df["pub_rec_bankruptcies"].fillna(0)
    df["revol_util"] = df["revol_util"].fillna(0)

    df["earliest_cr_line"] = pd.to_datetime(df["earliest_cr_line"], format="%b-%Y")
    df["issue_d"] = pd.to_datetime(df["issue_d"], format="%b-%Y")

    df["credit_age"] = (df["issue_d"] - df["earliest_cr_line"]).dt.days / 365
    df["earliest_cr_line_diff"] = (df["earliest_cr_line"] - earliest_cr_line_ref).dt.days / 365

    df["emp_title_caps_ratio"] = df["emp_title"].apply(
        lambda text: 0.0
        if not isinstance(text, str) or not text.strip()
        else sum(word[0].isupper() for word in text.split()) / len(text.split())
    )

    df["collapse_emp_title"] = (
        df["emp_title"]
        .apply(emp_title_cleaning.clean_emp_title)
        .apply(emp_title_cleaning.collapse_emp_title)
    )

    frames[split_name] = df

df_train = frames["train"]
df_holdout = frames["holdout"]
df_score = frames["score"]

# =============================================================================
# 3. PREPROCESSING PIPELINE
# =============================================================================
print("\n[3/5] Building preprocessor ...")

EXCLUDED_CATS = {"address", "grade", "sub_grade", "title", "earliest_cr_line", "emp_title", "loan_default"}

categorical_features = (
    df_train.select_dtypes(include=["object", "str", "bool"])
    .columns.difference(EXCLUDED_CATS)
    .tolist()
)
numerical_features = (
    df_train.select_dtypes(include=["int64", "float64"])
    .columns.drop(["loan_default"], errors="ignore")
    .tolist()
)
ordinal_features = ["grade", "sub_grade"]

ALL_FEATURES = numerical_features + categorical_features + ordinal_features

numeric_pipe = Pipeline([
    ("impute", SimpleImputer(strategy="median")),
])

cat_pipe = Pipeline([
    ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=True)),
])

ord_pipe = Pipeline([
    ("ord", OrdinalEncoder(
        categories=[
            ["A", "B", "C", "D", "E", "F", "G"],
            sorted(df_train["sub_grade"].dropna().unique()),
        ],
        handle_unknown="use_encoded_value",
        unknown_value=-1,
    )),
])

preprocessor = ColumnTransformer([
    ("num", numeric_pipe, numerical_features),
    ("cat", cat_pipe, categorical_features),
    ("ord", ord_pipe, ordinal_features),
])

preprocessor.fit(df_train[ALL_FEATURES])

X_train = preprocessor.transform(df_train[ALL_FEATURES])
X_val = preprocessor.transform(df_holdout[ALL_FEATURES])
X_score = preprocessor.transform(df_score[ALL_FEATURES])
y_train = df_train["loan_default"].to_numpy()
y_val = df_holdout["loan_default"].to_numpy()

print(f"  Feature dims: train={X_train.shape}, val={X_val.shape}, score={X_score.shape}")

# =============================================================================
# 4. RANDOM SEARCH
# =============================================================================
print("\n[4/5] Training ...")

N_RANDOM_SAMPLES = 10
results = []
best_model = None
best_params = None
best_auc = -np.inf
best_f1 = -1.0
best_threshold = 0.5

for i in range(N_RANDOM_SAMPLES):
    params = {
        "n_estimators": random.choice([500, 1000, 1573]),
        "max_depth": random.choice([3, 4, 5, 6, 7, 8, 9]),
        "learning_rate": float(np.exp(np.random.uniform(np.log(1e-3), np.log(0.3)))),
        "subsample": float(np.random.uniform(0.6, 1.0)),
        "gamma": float(np.random.uniform(0.0, 5.0)),
        "reg_alpha": float(np.exp(np.random.uniform(np.log(1e-6), np.log(1.0)))),
        "reg_lambda": float(np.exp(np.random.uniform(np.log(1e-3), np.log(10.0)))),
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "tree_method": "hist",
        "device": xgb_device,
        "random_state": 1809 + i,
        "n_jobs": -1,
    }

    model = xgb.XGBClassifier(**params)
    model.fit(X_train, y_train)

    y_prob = model.predict_proba(X_val)[:, 1]

    thresholds = np.linspace(0.01, 0.99, 99)
    model_best_threshold = 0.5
    model_best_f1 = -1.0

    for threshold in thresholds:
        y_pred = (y_prob >= threshold).astype(int)
        f1 = f1_score(y_val, y_pred)
        if f1 > model_best_f1:
            model_best_f1 = float(f1)
            model_best_threshold = float(threshold)

    auc = float(roc_auc_score(y_val, y_prob))

    results.append({
        "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "params": params,
        "auc": auc,
        "f1": model_best_f1,
        "best_threshold": model_best_threshold
    })

    print(
        f"  [{i + 1}/{N_RANDOM_SAMPLES}] {params} -> "
        f"AUC={auc:.4f} F1={model_best_f1:.4f} Best Thres={model_best_threshold:.3f}"
    )

    if model_best_f1 > best_f1:
        best_model = model
        best_params = params.copy()
        best_auc = auc
        best_f1 = model_best_f1
        best_threshold = model_best_threshold

results_df = pd.DataFrame(results)
results_path = RESULTS_DIR / f"{TODAY}_xgb_random_search_results.csv"
results_df.to_csv(results_path, index=False)
print(f"\n  Results -> {results_path}")
print(results_df.sort_values("f1", ascending=False).head(5).to_string(index=False))

# =============================================================================
# 5. SUBMISSION
# =============================================================================
print("\n[5/5] Generating submission ...")

print(
    f"  Best params={best_params}  "
    f"AUC={best_auc:.4f}  F1={best_f1:.4f}  thr={best_threshold:.3f}"
)

score_prob = best_model.predict_proba(X_score)[:, 1]
preds = (score_prob >= best_threshold).astype(int)

sub = pd.DataFrame({"ID": df_score["ID"], "loan_default": preds})
sub_path = SUB_DIR / f"{TODAY}_xgb_best_f1{best_f1:.4f}_thr{best_threshold:.3f}.csv"
sub.to_csv(sub_path, index=False)
print(f"    Saved -> {sub_path}")

print("\nDone.")
