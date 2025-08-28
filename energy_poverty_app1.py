# energy_poverty_app.py — robust split + friendlier diagnostics (v2)
# Streamlit app to explore energy poverty risk by LSOA

from __future__ import annotations

import io
import textwrap
from typing import List, Optional, Tuple, Dict, Any

import numpy as np
import pandas as pd
import requests
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# -------------------------
# Page config
# -------------------------

st.set_page_config(
    page_title="Energy Poverty Risk Predictor",
    page_icon="⚡",
    layout="wide",
)

# -------------------------
# Helpers for fuzzy column mapping
# -------------------------

def _normalize_col_name(s: str) -> str:
    return (
        str(s)
        .strip()
        .replace(" ", " ")
        .replace("  ", " ")
        .lower()
        .replace("(", "[")
        .replace(")", "]")
        .replace("/", "_")
        .replace("-", "_")
    )


def find_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    if df is None or df.empty:
        return None
    norm_map = {col: _normalize_col_name(col) for col in df.columns.astype(str)}
    inv = {v: k for k, v in norm_map.items()}
    for c in candidates:
        key = _normalize_col_name(c)
        if key in inv:
            return inv[key]
    for c in candidates:
        key = _normalize_col_name(c)
        for norm, orig in inv.items():
            if key in norm:
                return orig
    return None

# Candidate names for key columns
PCT_FP_CANDS = [
    "proportion_of_households_fuel_poor_[%]",
    "proportion of households fuel poor [%]",
    "proportion of households fuel poor (%)",
    "fuel poverty (%)",
    "% fuel poor",
]
HH_CANDS = [
    "estimated_number_of_households",
    "estimated number of households",
    "total households",
]
HH_FP_CANDS = [
    "estimated_number_of_households_in_fuel_poverty",
    "estimated number of households in fuel poverty",
    "households in fuel poverty",
]
LSOA_CANDS = [
    "lsoa_code",
    "lsoa code",
    "lsoa code (2011)",
    "lsoa",
]
IMD_DECILE_CANDS = [
    "imd_decile",
    "imd decile",
    "index of multiple deprivation [imd] decile [where 1 is most deprived 10 is least deprived]",
]

# -------------------------
# Data loading (with tiny embedded samples so the app always renders)
# -------------------------

SAMPLE_FUEL_CSV = """LSOA_Code,Proportion_of_households_fuel_poor_(%)
E01000001,17.2
E01000002,12.4
E01000003,23.1
E01000005,14.7
E01000006,9.9
"""
SAMPLE_IMD_CSV = """LSOA_Code,IMD_decile
E01000001,2
E01000002,6
E01000003,1
E01000005,4
E01000006,7
"""


@st.cache_data(show_spinner=False)
def _download_csv(url: str) -> pd.DataFrame:
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    return pd.read_csv(io.BytesIO(r.content))


def _canonicalize_fuel(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c) for c in df.columns]
    lsoa_col = find_col(df, LSOA_CANDS)
    pct_col = find_col(df, PCT_FP_CANDS)
    hh_col = find_col(df, HH_CANDS)
    hh_fp_col = find_col(df, HH_FP_CANDS)

    if lsoa_col is None:
        available = ", ".join(df.columns[:40])
        msg = textwrap.dedent(
            f"""
            Dataset is missing an LSOA column.
            Looked for any of: {LSOA_CANDS}
            Available columns include: {available}
            """
        ).strip()
        raise KeyError(msg)

    if pct_col is None:
        if hh_col is not None and hh_fp_col is not None:
            with np.errstate(divide="ignore", invalid="ignore"):
                pct = 100.0 * pd.to_numeric(df[hh_fp_col], errors="coerce") / pd.to_numeric(df[hh_col], errors="coerce")
            df["pct_fuel_poor"] = pct.astype(float)
        else:
            available = ", ".join(df.columns[:40])
            msg = textwrap.dedent(
                f"""
                Dataset missing required columns. Need either a percent column OR both household counts.

                Looked for:
                  pct candidates: {PCT_FP_CANDS}
                  households candidates: {HH_CANDS}
                  fuel-poor households candidates: {HH_FP_CANDS}

                Available columns include: {available}
                """
            ).strip()
            raise KeyError(msg)
    else:
        df["pct_fuel_poor"] = pd.to_numeric(df[pct_col], errors="coerce")

    df.rename(columns={lsoa_col: "LSOA_Code"}, inplace=True)
    return df[["LSOA_Code", "pct_fuel_poor"]].dropna()


@st.cache_data(show_spinner=False)
def load_fuel_poverty(url: Optional[str] = None) -> pd.DataFrame:
    try:
        if url:
            df = _download_csv(url)
        else:
            raise RuntimeError("No URL provided; using embedded sample.")
    except Exception:
        df = pd.read_csv(io.StringIO(SAMPLE_FUEL_CSV))
    return _canonicalize_fuel(df)


def _canonicalize_imd(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    lsoa_col = find_col(df, LSOA_CANDS)
    decile_col = find_col(df, IMD_DECILE_CANDS)
    if lsoa_col is None or decile_col is None:
        # Provide minimal frame with NaNs so the model still runs using median imputation
        return pd.DataFrame({
            "LSOA_Code": df.get(lsoa_col, pd.Series(dtype=str)),
            "IMD_decile": pd.to_numeric(df.get(decile_col, pd.Series(dtype=float)), errors="coerce"),
        })
    out = df[[lsoa_col, decile_col]].copy()
    out.columns = ["LSOA_Code", "IMD_decile"]
    out["IMD_decile"] = pd.to_numeric(out["IMD_decile"], errors="coerce")
    return out.dropna(subset=["LSOA_Code"]).drop_duplicates("LSOA_Code")


@st.cache_data(show_spinner=False)
def load_imd(url: Optional[str] = None) -> pd.DataFrame:
    try:
        if url:
            df = _download_csv(url)
        else:
            raise RuntimeError("No URL provided; using embedded sample.")
    except Exception:
        df = pd.read_csv(io.StringIO(SAMPLE_IMD_CSV))
    return _canonicalize_imd(df)

# -------------------------
# Feature build + label definition
# -------------------------

def build_dataset(
    threshold_pct: float = 15.0,
    top_quantile: float = 0.2,
    fuel_url: Optional[str] = None,
    imd_url: Optional[str] = None,
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    fuel = load_fuel_poverty(fuel_url)
    imd = load_imd(imd_url)

    df = fuel.merge(imd, on="LSOA_Code", how="left")

    # Label: above absolute threshold OR in top-q tail
    q_cut = df["pct_fuel_poor"].quantile(1 - top_quantile)
    high_risk = (df["pct_fuel_poor"] >= threshold_pct) | (df["pct_fuel_poor"] >= q_cut)

    y = high_risk.astype(int)

    # Features (keep MVP tiny)
    X = pd.DataFrame(index=df.index)
    X["IMD_decile"] = df["IMD_decile"].fillna(df["IMD_decile"].median())

    meta = df[["LSOA_Code", "pct_fuel_poor", "IMD_decile"]]
    return X, y, meta

# -------------------------
# Safer train/test split that avoids scikit-learn's stratify errors
# -------------------------

def robust_train_test_split(
    X: pd.DataFrame,
    y: pd.Series,
    meta: pd.DataFrame,
    test_size: float,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    info: Dict[str, Any] = {"warning": None}

    counts = y.value_counts()
    can_stratify = (y.nunique() >= 2) and (counts.min() >= 2)

    if can_stratify:
        X_train, X_test, y_train, y_test, meta_train, meta_test = train_test_split(
            X, y, meta, test_size=test_size, random_state=random_state, stratify=y
        )
        return X_train, X_test, y_train, y_test, meta_train, meta_test, info

    # If not stratifiable, try unstratified splits until the training set has both classes
    info["warning"] = (
        "Stratified split not possible (need ≥2 samples per class). Using an unstratified split;"
        " consider lowering the threshold or increasing sample size for better balance."
    )
    # If y has only a single class, manufacture a temporary label using the median to at least let the app run
    if y.nunique() < 2:
        if "pct_fuel_poor" in meta.columns:
            y = (meta["pct_fuel_poor"] >= meta["pct_fuel_poor"].median()).astype(int)
            counts = y.value_counts()
            info["warning"] += f" Auto-relabeled using median split (class counts: {counts.to_dict()})."
        else:
            info["warning"] += " Could not auto-relabeled due to missing pct_fuel_poor; continuing anyway."

    for seed in range(random_state, random_state + 200):
        X_train, X_test, y_train, y_test, meta_train, meta_test = train_test_split(
            X, y, meta, test_size=test_size, random_state=seed, stratify=None
        )
        if y_train.nunique() >= 2:
            return X_train, X_test, y_train, y_test, meta_train, meta_test, info

    # Last resort: return the last split and warn loudly
    info["warning"] += " Unable to ensure both classes in train split; model training may fail."
    return X_train, X_test, y_train, y_test, meta_train, meta_test, info

# -------------------------
# UI
# -------------------------

st.title("⚡ Energy Poverty Risk Predictor — MVP")
with st.sidebar:
    st.header("Settings")
    threshold = st.slider("Fuel poverty threshold (%)", 5.0, 40.0, 15.0, 0.5)
    top_q = st.slider("Top-quantile tail (q)", 0.05, 0.5, 0.2, 0.05, help="Label as high-risk if in the top (1 - q) tail of % fuel poor.")
    test_size = st.slider("Test size", 0.1, 0.4, 0.2, 0.05)

    st.divider()
    st.caption("Data sources (optional). Leave blank to use embedded samples while wiring things up.")
    fuel_url = st.text_input("Fuel poverty CSV URL", value="")
    imd_url = st.text_input("IMD CSV URL", value="")

# Build dataset
try:
    X, y, meta = build_dataset(threshold_pct=threshold, top_quantile=top_q, fuel_url=fuel_url or None, imd_url=imd_url or None)
except Exception as e:
    st.error("Data loading / preparation failed. See details below and adjust your URLs or file schemas.")
    st.code(textwrap.dedent(f"""
    {type(e).__name__}: {e}

    Hint: the loader auto-maps common column names; open the sidebar and try the built-in sample first to verify.
    """))
    st.stop()

# Diagnostics: label counts
lc = y.value_counts().to_dict()
colA, colB = st.columns(2)
with colA:
    st.subheader("Label definition")
    st.write(
        textwrap.dedent(
            f"""
            Expected label: **1** if `pct_fuel_poor ≥ {threshold}%` **or** in the top **(1 - {top_q})** tail; else **0**.
            Current label counts: {lc}
            """
        )
    )
with colB:
    st.subheader("Feature columns used")
    st.write(["IMD_decile"])

# Split (robust)
X_train, X_test, y_train, y_test, meta_train, meta_test, split_info = robust_train_test_split(
    X, y, meta, test_size=test_size, random_state=42
)
if split_info["warning"]:
    st.warning(split_info["warning"])

# Model choice
with st.sidebar:
    model_choice = st.selectbox("Model", ["Logistic Regression", "Random Forest"])

if model_choice == "Logistic Regression":
    pre = ColumnTransformer([("num", StandardScaler(), ["IMD_decile"])])
    clf = LogisticRegression(max_iter=200)
else:
    pre = "passthrough"
    clf = RandomForestClassifier(n_estimators=200, random_state=42)

pipe = Pipeline([("pre", pre), ("clf", clf)])

# Guard: ensure at least two classes in y_train for training
if y_train.nunique() < 2:
    st.error("Training set contains only one class after splitting. Lower the threshold or increase top-quantile to create both classes.")
    st.stop()

pipe.fit(X_train, y_train)

y_pred = pipe.predict(X_test)

c1, c2 = st.columns(2)
with c1:
    st.subheader("Confusion Matrix")
    cm = confusion_matrix(y_test, y_pred)
    st.dataframe(pd.DataFrame(cm, index=["Actual 0", "Actual 1"], columns=["Pred 0", "Pred 1"]))
with c2:
    st.subheader("Classification Report")
    report = classification_report(y_test, y_pred, output_dict=True)
    st.dataframe(pd.DataFrame(report).T)

st.subheader("Peek test set")
preview = meta_test.copy()
preview["pred_high_risk"] = y_pred
st.dataframe(preview.head(25))

st.caption("MVP note: the app gracefully handles sparse labels and schema variants; for best results, tune the threshold/top-q to ensure both classes are present.")
