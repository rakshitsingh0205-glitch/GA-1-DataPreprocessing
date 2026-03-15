"""
feature_helpers.py
Reusable helper functions for Feature Engineering Capstone.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, roc_auc_score, f1_score,
    confusion_matrix, ConfusionMatrixDisplay
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler

SEED = 42


# ─────────────────────────────────────────────────────────────────────────────
# 1.  Evaluation helpers
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_binary(y_true, y_pred, y_proba, label="Model"):
    """Return a dict of accuracy, ROC-AUC, and F1."""
    return {
        "label"   : label,
        "accuracy": round(accuracy_score(y_true, y_pred), 4),
        "roc_auc" : round(roc_auc_score(y_true, y_proba), 4),
        "f1"      : round(f1_score(y_true, y_pred), 4),
    }


def plot_confusion(y_true, y_pred, class_names=None, title="Confusion Matrix"):
    """Plot a confusion matrix."""
    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay(
        confusion_matrix(y_true, y_pred),
        display_labels=class_names or ["0", "1"],
    ).plot(ax=ax, colorbar=False)
    ax.set_title(title)
    plt.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 2.  Preprocessing helpers
# ─────────────────────────────────────────────────────────────────────────────

def compare_scalers(series: pd.Series):
    """
    Apply MinMax, Standard, and Robust scalers to a Series.
    Returns a DataFrame with original + scaled columns.
    """
    vals = series.values.reshape(-1, 1)
    result = pd.DataFrame({"original": series.values})
    for name, scaler in [
        ("minmax",   MinMaxScaler()),
        ("standard", StandardScaler()),
        ("robust",   RobustScaler()),
    ]:
        result[name] = scaler.fit_transform(vals).ravel()
    return result


def plot_scaling_comparison(series: pd.Series, feature_name: str):
    """Histogram comparison of original vs three scalers."""
    df_scaled = compare_scalers(series)
    fig, axes = plt.subplots(1, 4, figsize=(18, 4), sharey=False)
    colors = ["steelblue", "#DD8452", "#55A868", "#C44E52"]
    for ax, col, c in zip(axes, df_scaled.columns, colors):
        ax.hist(df_scaled[col], bins=50, color=c, alpha=0.85, edgecolor="white")
        ax.set_title(col, fontweight="bold")
    fig.suptitle(f"Scaling comparison — {feature_name}", fontsize=13)
    plt.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 3.  Feature construction helpers
# ─────────────────────────────────────────────────────────────────────────────

def make_hotel_features(df: pd.DataFrame, train_idx=None) -> pd.DataFrame:
    """
    Add engineered features to the hotel bookings dataframe.
    Group aggregates are fit on train_idx only (leakage prevention).
    """
    df = df.copy()

    # -- Ratio features
    df["total_guests"]          = df["adults"] + df["children"].fillna(0) + df["babies"].fillna(0)
    df["price_per_person"]      = df["adr"] / (df["total_guests"] + 1)
    df["total_nights"]          = df["stays_in_week_nights"] + df["stays_in_weekend_nights"]
    df["special_requests_rate"] = df["total_of_special_requests"] / (df["total_nights"] + 1)

    # -- Interaction features
    df["adr_x_lead_time"] = df["adr"] * df["lead_time"]
    df["nights_x_guests"] = df["total_nights"] * df["total_guests"]

    # -- Binary flags
    df["is_family"]             = ((df["children"].fillna(0) + df["babies"].fillna(0)) > 0).astype(int)
    df["is_repeated_guest_flag"]= df["is_repeated_guest"].fillna(0).astype(int)

    # -- Group aggregates (train only)
    if train_idx is None:
        train_idx = df.index
    country_adr_map  = df.loc[train_idx].groupby("country")["adr"].mean()
    hotel_cancel_map = df.loc[train_idx].groupby("hotel")["is_canceled"].mean()
    global_adr_mean  = df.loc[train_idx, "adr"].mean()

    df["country_avg_adr"]   = df["country"].map(country_adr_map).fillna(global_adr_mean)
    df["hotel_cancel_rate"] = df["hotel"].map(hotel_cancel_map).fillna(0.5)

    return df


def make_date_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extract date/time features from hotel booking date columns."""
    df = df.copy()
    month_map = {
        "January":1,"February":2,"March":3,"April":4,"May":5,"June":6,
        "July":7,"August":8,"September":9,"October":10,"November":11,"December":12
    }
    df["arrival_month_num"] = df["arrival_date_month"].map(month_map)
    df["arrival_date_parsed"] = pd.to_datetime(
        df["arrival_date_year"].astype(str) + "-" +
        df["arrival_month_num"].astype(str).str.zfill(2) + "-" +
        df["arrival_date_day_of_month"].astype(str).str.zfill(2),
        errors="coerce",
    )
    df["arrival_weekday"]    = df["arrival_date_parsed"].dt.dayofweek
    df["is_weekend_arrival"] = (df["arrival_weekday"] >= 5).astype(int)
    df["arrival_quarter"]    = df["arrival_date_parsed"].dt.quarter

    def _season(m):
        if m in [12, 1, 2]:  return "winter"
        elif m in [3, 4, 5]: return "spring"
        elif m in [6, 7, 8]: return "summer"
        else:                return "autumn"

    df["arrival_season"] = df["arrival_month_num"].apply(_season)
    df["lead_time_bucket"] = pd.cut(
        df["lead_time"],
        bins=[-1, 7, 30, 90, 365, 10_000],
        labels=["same_week", "1mo", "3mo", "1yr", "far_future"],
    )
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 4.  Leakage-safe target encoding
# ─────────────────────────────────────────────────────────────────────────────

def target_encode(df: pd.DataFrame, col: str, target: str,
                  train_idx, smoothing: float = 10.0) -> pd.Series:
    """
    Smoothed target encoding computed on train_idx only.
    Smoothing formula: (n * cat_mean + k * global_mean) / (n + k)
    """
    global_mean = df.loc[train_idx, target].mean()
    stats = df.loc[train_idx].groupby(col)[target].agg(["mean", "count"])
    smooth = (stats["count"] * stats["mean"] + smoothing * global_mean) / (
        stats["count"] + smoothing
    )
    return df[col].map(smooth).fillna(global_mean)


# ─────────────────────────────────────────────────────────────────────────────
# 5.  Curse of dimensionality demo
# ─────────────────────────────────────────────────────────────────────────────

def distance_concentration_demo(dimensions=(2, 10, 50, 200),
                                 n_samples=500, n_pairs=5000, seed=SEED):
    """
    Generate random data in each dimensionality,
    return dict of pairwise distance arrays.
    """
    from sklearn.datasets import make_classification
    np.random.seed(seed)
    out = {}
    for d in dimensions:
        X, _ = make_classification(
            n_samples=n_samples, n_features=d,
            n_informative=max(2, d // 5), n_redundant=0, random_state=seed,
        )
        idx1 = np.random.randint(0, n_samples, n_pairs)
        idx2 = np.random.randint(0, n_samples, n_pairs)
        dists = np.sqrt(((X[idx1] - X[idx2]) ** 2).sum(axis=1))
        out[d] = dists
    return out
