"""
pipelines.py
Modular scikit-learn pipelines for the Feature Engineering Capstone.
"""

import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import (
    RobustScaler, StandardScaler, OneHotEncoder,
    PowerTransformer, FunctionTransformer
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression


# ─────────────────────────────────────────────────────────────────────────────
# Sub-transformers
# ─────────────────────────────────────────────────────────────────────────────

def log_numeric_pipe():
    """Impute → log1p → RobustScaler for right-skewed columns."""
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("log",     FunctionTransformer(np.log1p, validate=True)),
        ("scaler",  RobustScaler()),
    ])


def power_numeric_pipe():
    """Impute → Yeo-Johnson → StandardScaler for near-normal columns."""
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("power",   PowerTransformer(method="yeo-johnson")),
        ("scaler",  StandardScaler()),
    ])


def categorical_pipe():
    """Impute → OneHotEncoder for categorical columns."""
    return Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])


# ─────────────────────────────────────────────────────────────────────────────
# Full preprocessing + model pipeline
# ─────────────────────────────────────────────────────────────────────────────

def build_rf_pipeline(log_cols, scale_cols, cat_cols, n_estimators=200, seed=42):
    """
    Full end-to-end RandomForest pipeline.

    Parameters
    ----------
    log_cols   : list of column names to log-transform
    scale_cols : list of column names to power-transform + standardize
    cat_cols   : list of categorical column names
    """
    preprocessor = ColumnTransformer([
        ("log_feat",   log_numeric_pipe(),  log_cols),
        ("scale_feat", power_numeric_pipe(), scale_cols),
        ("cat_feat",   categorical_pipe(),  cat_cols),
    ])
    return Pipeline([
        ("preprocess", preprocessor),
        ("model", RandomForestClassifier(
            n_estimators=n_estimators, random_state=seed, n_jobs=-1
        )),
    ])


def build_logreg_pipeline(log_cols, scale_cols, cat_cols, seed=42):
    """
    Full end-to-end Logistic Regression pipeline (used for baseline comparison).
    """
    preprocessor = ColumnTransformer([
        ("log_feat",   log_numeric_pipe(),  log_cols),
        ("scale_feat", power_numeric_pipe(), scale_cols),
        ("cat_feat",   categorical_pipe(),  cat_cols),
    ])
    return Pipeline([
        ("preprocess", preprocessor),
        ("model", LogisticRegression(max_iter=1000, random_state=seed)),
    ])
