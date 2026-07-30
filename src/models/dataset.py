"""Dataset assembly, chronological splitting and preprocessing.

A single source of truth for what a "feature" is, so that EDA, selection,
training, explanation, the API and the dashboard can never disagree.
"""
from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import CFG, MODELS_DIR, PROCESSED, TARGET
from src.features.build_features import FEATURE_GROUPS, ID_COLS

CATEGORICAL = [
    "time_block", "season", "dep_rain_intensity", "dep_wind_category",
    "dep_visibility_level", "dep_wx_condition", "arr_wx_condition",
    "distance_band", "carrier_code", "carrier_type", "aircraft_model",
    "category", "hub_pair_type", "origin_region", "dest_region",
    "origin", "destination",
]

ALL_FEATURES = sorted({c for cols in FEATURE_GROUPS.values() for c in cols})
NUMERIC = [c for c in ALL_FEATURES if c not in CATEGORICAL]

LEAKY = {"departure_delay_min", "arrival_delay_min", "is_delayed", "cancelled",
         "diverted", "flight_id", "tail_number", "route", "date",
         "local_departure_date", "scheduled_departure_utc",
         "scheduled_departure_local", "carrier_name", "holiday_name",
         "flight_number", "days_from_major_holiday"}
assert not (set(ALL_FEATURES) & LEAKY), set(ALL_FEATURES) & LEAKY


@dataclass
class Splits:
    X_train: pd.DataFrame
    y_train: pd.Series
    X_valid: pd.DataFrame
    y_valid: pd.Series
    X_test: pd.DataFrame
    y_test: pd.Series
    meta_train: pd.DataFrame
    meta_valid: pd.DataFrame
    meta_test: pd.DataFrame
    features: list[str]

    def summary(self) -> str:
        return (f"train {len(self.y_train):,} ({self.y_train.mean():.1%} pos) | "
                f"valid {len(self.y_valid):,} ({self.y_valid.mean():.1%}) | "
                f"test {len(self.y_test):,} ({self.y_test.mean():.1%})")


def load_features(path=None) -> pd.DataFrame:
    df = pd.read_parquet(path or PROCESSED / "features.parquet")
    df["date"] = pd.to_datetime(df["date"])
    return df


def selected_features() -> list[str] | None:
    p = MODELS_DIR / "selected_features.json"
    if p.exists():
        return json.loads(p.read_text())["selected"]
    return None


def coerce_types(X: pd.DataFrame, categories: dict | None = None) -> pd.DataFrame:
    """Cast categoricals to pandas `category` dtype (native support in LGBM/XGB)."""
    X = X.copy()
    for c in X.columns:
        if c in CATEGORICAL:
            if categories is not None and c in categories:
                X[c] = pd.Categorical(X[c].astype("object"), categories=categories[c])
            else:
                X[c] = X[c].astype("object").astype("category")
        else:
            X[c] = pd.to_numeric(X[c], errors="coerce").astype("float32")
    return X


def category_levels(X: pd.DataFrame) -> dict[str, list]:
    return {c: list(X[c].cat.categories) for c in X.columns if str(X[c].dtype) == "category"}


def make_splits(df: pd.DataFrame | None = None,
                features: list[str] | None = None) -> Splits:
    df = load_features() if df is None else df
    feats = features or selected_features() or ALL_FEATURES
    feats = [f for f in feats if f in df.columns]

    train_end = pd.Timestamp(CFG["split"]["train_end"])
    valid_end = pd.Timestamp(CFG["split"]["valid_end"])

    m_tr = df["date"] <= train_end
    m_va = (df["date"] > train_end) & (df["date"] <= valid_end)
    m_te = df["date"] > valid_end

    meta_cols = [c for c in ID_COLS if c in df.columns]
    cats = category_levels(coerce_types(df.loc[m_tr, feats]))

    return Splits(
        X_train=coerce_types(df.loc[m_tr, feats], cats),
        y_train=df.loc[m_tr, TARGET].astype(int),
        X_valid=coerce_types(df.loc[m_va, feats], cats),
        y_valid=df.loc[m_va, TARGET].astype(int),
        X_test=coerce_types(df.loc[m_te, feats], cats),
        y_test=df.loc[m_te, TARGET].astype(int),
        meta_train=df.loc[m_tr, meta_cols],
        meta_valid=df.loc[m_va, meta_cols],
        meta_test=df.loc[m_te, meta_cols],
        features=feats,
    )


def sklearn_preprocessor(features: list[str]) -> ColumnTransformer:
    """Impute + scale numerics, one-hot the categoricals (for linear models)."""
    num = [f for f in features if f not in CATEGORICAL]
    cat = [f for f in features if f in CATEGORICAL]
    return ColumnTransformer([
        ("num", Pipeline([("impute", SimpleImputer(strategy="median")),
                          ("scale", StandardScaler())]), num),
        ("cat", Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                          ("ohe", OneHotEncoder(handle_unknown="ignore",
                                                min_frequency=30,
                                                sparse_output=False))]), cat),
    ], remainder="drop", verbose_feature_names_out=False)


def to_ordinal(X: pd.DataFrame) -> pd.DataFrame:
    """Integer-code categoricals (for estimators without native support)."""
    X = X.copy()
    for c in X.columns:
        if str(X[c].dtype) == "category":
            X[c] = X[c].cat.codes.replace(-1, np.nan).astype("float32")
    return X
