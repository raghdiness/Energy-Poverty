# Energy Poverty Risk Predictor — Streamlit App (Day‑1 MVP)

import io
import textwrap
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# ----------------------------
# Utilities
# ----------------------------

def _norm(s: str) -> str:
    """Normalize a column name: lowercase + only letters/numbers/underscore."""
    return "".join(ch if ch.isalnum() else "_" for ch in s.strip().lower())


def _rename_like(df: pd.DataFrame, mapping: Dict[str, List[str]]) -> Tuple[pd.DataFrame, Dict[str, str]]:
    """Return a copy of df with standardized column names based on candidate lists.

    Parameters
    ----------
    df: the original dataframe
    mapping: dict where keys are canonical names (e.g., 'lsoa_code') and values are
             lists of candidate source header strings (case/spacing-insensitive)

    Returns
    -------
    (new_df, used) where used maps canonical->original header actually used
    """
    norm_cols = {col: _norm(col) for col in df.columns}
    inv = {v: k for k, v in norm_cols.items()}  # normalized -> original

    used: Dict[str, str] = {}
    for canon, candidates in mapping.items():
        # normalize candidates and try to find the first that exists
        for cand in candidates:
            nc = _norm(cand)
            if nc in inv:
                used[canon] = inv[nc]
                break

    # Build rename dict for those found
    rename_dict = {used[k]: k for k in used}
    new_df = df.rename(columns=rename_dict).copy()
    return new_df, used


def _coerce_numeric(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


# ----------------------------
# Data loading
# ----------------------------

SAMPLE_CSV = """
LSOA Code,LSOA Name,Estimated number of households,Estimated number of households in fuel poverty,Proportion of households fuel poor (%)
E01000001,City of London 001A,850,95,11.18
E01000002,City of London 001B,1000,60,6.00
E01000003,City of London 001C,750,120,16.00
E01000005,City of London 001E,600,48,8.00
""".strip()


@st.cache_data(show_spinner=False)
def load_sample() -> pd.DataFrame:
    return pd.read_csv(io.StringIO(SAMPLE_CSV))


def load_from_upload(upload) -> Optional[pd.DataFrame]:
    if upload is None:
        return None
    try:
        return pd.read_csv(upload)
    except Exception:
        upload.seek(0)
        try:
            return pd.read_excel(upload)
        except Exception as e:  # pragma: no cover
            st.error(f"Could not read file: {e}")
            return None


# ----------------------------
# Feature engineering & label
# ----------------------------

CANDIDATES = {
    "lsoa_code": [
        "lsoa_code", "LSOA Code", "LSOA_Code", "LSOA code", "LSOA2021CD", "LSOA11CD",
    ],
    "lsoa_name": [
        "lsoa_name", "LSOA Name", "LSOA_Name", "LSOA name", "LSOA2021NM", "LSOA11NM",
    ],
    "households": [
        "Estimated number of households", "Estimated_number_of_households", "households",
        "Total households",
    ],
    "households_in_fp": [
        "Estimated number of households in fuel poverty",
        "Estimated_number_of_households_in_fuel_poverty",
        "households_in_fuel_poverty",
    ],
    "pct_fuel_poor": [
        "Proportion of households fuel poor (%)",
        "Proportion_of_households_fuel_poor_(%)",
        "fuel_poverty_%", "fuel_poverty_percent", "pct_fuel_poor",
    ],
    # optional extras that we may one-hot encode if present
    "region": ["Region", "region_name", "region"],
    "imd_decile": ["IMD_Decile", "imd_decile", "IMD decile", "Index of Multiple Deprivation Decile"],
}


def derive_missing_fields(df: pd.DataFrame) -> pd.DataFrame:
    # Ensure core numeric columns are numeric
    df = _coerce_numeric(df, ["households", "households_in_fp", "pct_fuel_poor"])  # type: ignore

    # Derive pct if missing
    if "pct_fuel_poor" not in df.columns and {"households", "households_in_fp"}.issubset(df.columns):
        df["pct_fuel_poor"] = (df["households_in_fp"] / df["households"]) * 100.0

    # Derive households_in_fp if missing
    if "households_in_fp" not in df.columns and {"households", "pct_fuel_poor"}.issubset(df.columns):
        df["households_in_fp"] = (df["households"] * df["pct_fuel_poor"] / 100.0).round()

    # Derive households if missing
    if "households" not in df.columns and {"households_in_fp", "pct_fuel_poor"}.issubset(df.columns):
        with np.errstate(divide="ignore", invalid="ignore"):
            denom = df["pct_fuel_poor"] / 100.0
            df["households"] = np.where(denom > 0, df["households_in_fp"] / denom, np.nan)

    return df


@dataclass
class Dataset:
    X: pd.DataFrame
    y: pd.Series
    meta: pd.DataFrame


def build_dataset(raw_df: pd.DataFrame, threshold_pct: float) -> Dataset:
    # Standardize names
    std_df, used = _rename_like(raw_df, CANDIDATES)

    required_any = [["pct_fuel_poor"], ["households", "households_in_fp"]]
    if not any(all(c in std_df.columns for c in group) for group in required_any):
        missing = "pct_fuel_poor" if "pct_fuel_poor" not in std_df.columns else "households/households_in_fp"
        msg = textwrap.dedent(
            f"""Dataset missing required columns. Need either 'pct_fuel_poor' OR both 'households' and 'households_in_fp'.
Detected columns: {list(raw_df.columns)}
Mapped columns: {used}"""
        ).strip()
        raise ValueError(msg)
            f"Mapped columns: {used}"
        )

    std_df = derive_missing_fields(std_df)

    # Drop rows without an LSOA code if present
    if "lsoa_code" in std_df.columns:
        std_df = std_df.dropna(subset=["lsoa_code"]).drop_duplicates("lsoa_code")

    # Target
    std_df["is_fuel_poor"] = (std_df["pct_fuel_poor"] >= threshold_pct).astype(int)

    # Features: numeric + optional categoricals
    feature_cols_num = [c for c in ["pct_fuel_poor", "households", "households_in_fp"] if c in std_df.columns]
    cat_cols = [c for c in ["region", "imd_decile"] if c in std_df.columns]

    X = std_df[feature_cols_num + cat_cols].copy()
    y = std_df["is_fuel_poor"].copy()
    meta_cols = [c for c in ["lsoa_code", "lsoa_name"] if c in std_df.columns]
    meta = std_df[meta_cols + ["pct_fuel_poor"]].copy()

    return Dataset(X=X, y=y, meta=meta)


# ----------------------------
# Modeling
# ----------------------------

def make_pipeline(numeric_cols: List[str], categorical_cols: List[str]) -> Pipeline:
    transformers = []
    if numeric_cols:
        transformers.append(("num", StandardScaler(), numeric_cols))
    if categorical_cols:
        transformers.append(("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols))

    pre = ColumnTransformer(transformers=transformers) if transformers else "passthrough"
    clf = LogisticRegression(max_iter=1000)
    return Pipeline(steps=[("pre", pre), ("clf", clf)])


# ----------------------------
# App UI
# ----------------------------

st.set_page_config(page_title="Energy Poverty Risk Predictor", layout="wide")

st.title("Energy Poverty Risk Predictor ")
st.caption("Fast MVP: robust to column name variations and friendly diagnostics.")

with st.sidebar:
    st.header("Settings")
    source = st.radio("Data source", ["Sample", "Upload CSV"], index=0)
    threshold = st.slider("Fuel-poverty threshold (%)", min_value=5.0, max_value=25.0, value=10.0, step=0.5)
    test_size = st.slider("Test size", min_value=0.1, max_value=0.4, value=0.2, step=0.05)

    st.markdown("**Expected label:** 1 if `pct_fuel_poor ≥ threshold`, else 0.")

# Load data
if source == "Sample":
    df_raw = load_sample()
else:
    uploaded = st.file_uploader("Upload a CSV or Excel with LSOA-level fuel poverty data", type=["csv", "xls", "xlsx"])
    df_raw = load_from_upload(uploaded) if uploaded is not None else None

if df_raw is None:
    st.info("Upload a file or switch to the sample to get started.")
    st.stop()

st.subheader("1) Data preview & column mapping")
st.dataframe(df_raw.head(10))

# Try to map columns and build dataset
try:
    data = build_dataset(df_raw, threshold_pct=threshold)
    st.success("Columns detected & engineered successfully.")
except Exception as e:
    st.error("Could not build dataset. See details below.")
    st.exception(e)
    st.stop()

# Show mapping summary
std_df, used_map = _rename_like(df_raw, CANDIDATES)
col1, col2 = st.columns(2)
with col1:
    st.write("**Canonical → Source mapping**")
    if used_map:
        st.json(used_map)
    else:
        st.write("No canonical columns found; using raw column names.")
with col2:
    st.write("**Engineered features present**")
    present = [c for c in ["pct_fuel_poor", "households", "households_in_fp", "region", "imd_decile"] if c in std_df.columns]
    st.write(present)

# Train/test split and model
X, y, meta = data.X, data.y, data.meta
num_cols = [c for c in ["pct_fuel_poor", "households", "households_in_fp"] if c in X.columns and pd.api.types.is_numeric_dtype(X[c])]
cat_cols = [c for c in ["region", "imd_decile"] if c in X.columns and not pd.api.types.is_numeric_dtype(X[c])]

if y.nunique() < 2:
    st.warning("Your threshold produced only one class — adjust the slider until both classes exist.")
    st.stop()

X_train, X_test, y_train, y_test, meta_train, meta_test = train_test_split(
    X, y, meta, test_size=test_size, random_state=42, stratify=y
)

pipe = make_pipeline(num_cols, cat_cols)
pipe.fit(X_train, y_train)

proba = pipe.predict_proba(X_test)[:, 1]
preds = (proba >= 0.5).astype(int)

st.subheader("2) Model performance (Logistic Regression)")
colA, colB, colC = st.columns(3)
with colA:
    st.metric("Accuracy", f"{accuracy_score(y_test, preds):.3f}")
with colB:
    try:
        auc = roc_auc_score(y_test, proba)
        st.metric("ROC AUC", f"{auc:.3f}")
    except Exception:
        st.metric("ROC AUC", "n/a")
with colC:
    cm = confusion_matrix(y_test, preds)
    st.write("Confusion matrix:")
    st.write(pd.DataFrame(cm, index=["Actual 0", "Actual 1"], columns=["Pred 0", "Pred 1"]))

with st.expander("Classification report"):
    st.text(classification_report(y_test, preds))

# Rankings export
st.subheader("3) High-risk areas & export")
rank_df = meta_test.copy()
rank_df["proba_fuel_poor"] = proba
rank_df = rank_df.sort_values("proba_fuel_poor", ascending=False)

st.dataframe(rank_df.head(20))

csv_bytes = rank_df.to_csv(index=False).encode("utf-8")
st.download_button(
    label="Download predictions as CSV",
    data=csv_bytes,
    file_name="energy_poverty_risk_rankings.csv",
    mime="text/csv",
)

# Helpful how-to (make sure the string closes properly!)
st.subheader("How to run locally (optional)")
st.code(textwrap.dedent('''
# 1) Create & activate a virtual environment (Windows PowerShell example)
python -m venv .venv
.venv\Scripts\Activate.ps1

# 2) Install dependencies
pip install streamlit pandas numpy scikit-learn

# 3) Run the app
streamlit run energy_poverty_app.py

# (Optional) Run a quick CLI self-test without Streamlit
python energy_poverty_app.py --cli
''').strip())

# ----------------------------
# Minimal self-tests (do not change app behavior)
# ----------------------------

def _self_tests() -> None:
    # Variant headers case 1
    df1 = pd.DataFrame({
        "LSOA_Code": ["A", "B"],
        "Estimated_number_of_households": [100, 200],
        "Estimated_number_of_households_in_fuel_poverty": [10, 30],
    })
    d1 = build_dataset(df1, threshold_pct=10)
    assert "pct_fuel_poor" in _rename_like(derive_missing_fields(_rename_like(df1, CANDIDATES)[0]), CANDIDATES)[0].columns
    assert d1.y.iloc[0] == 1  # 10/100 = 10% → label 1

    # Variant headers case 2 (percent provided, counts missing)
    df2 = pd.DataFrame({
        "LSOA code": ["C", "D"],
        "fuel_poverty_%": [5.0, 12.0],
    })
    d2 = build_dataset(df2, threshold_pct=10)
    assert d2.y.tolist() == [0, 1]

# CLI mode for quick smoke tests
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--cli", action="store_true", help="Run a quick CLI smoke test & write CSV")
    args = parser.parse_args()
    if args.cli:
        print("Running CLI self-tests...")
        _self_tests()
        out = pd.read_csv(io.StringIO(SAMPLE_CSV))
        d = build_dataset(out, threshold_pct=10)
        X_train, X_test, y_train, y_test = train_test_split(d.X, d.y, test_size=0.2, random_state=42, stratify=d.y)
        pipe = make_pipeline([c for c in ["pct_fuel_poor", "households", "households_in_fp"] if c in d.X.columns],
                             [c for c in ["region", "imd_decile"] if c in d.X.columns and not pd.api.types.is_numeric_dtype(d.X[c])])
        pipe.fit(X_train, y_train)
        proba = pipe.predict_proba(X_test)[:, 1]
        meta = d.meta.loc[X_test.index].copy()
        meta["proba_fuel_poor"] = proba
        meta.sort_values("proba_fuel_poor", ascending=False).to_csv("energy_poverty_risk_rankings.csv", index=False)
        print("Wrote energy_poverty_risk_rankings.csv")



