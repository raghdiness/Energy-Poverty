"""
Energy Poverty Risk Predictor - Day 1 MVP
Author: Raghda (Rosaline)

This file now supports two modes so it does not crash when Streamlit is missing.

A) Streamlit UI (preferred)
   - Run:  pip install -U pandas numpy scikit-learn streamlit matplotlib requests
   - Then: streamlit run app.py

B) CLI fallback (no Streamlit needed)
   - Run:  python app.py --cli
   - Produces: metrics printed to console and a CSV file `energy_poverty_risk_rankings.csv`

C) Built-in smoke tests
   - Run:  python app.py --test
   - Uses tiny synthetic data to verify the pipeline builds, trains, and predicts.

Notes
- England-only MVP because the LSOA fuel poverty CSV covers England.
- "High risk" label is a proxy using a quantile or absolute threshold. For policy, prefer official LILEE definitions and microdata.
"""

from __future__ import annotations

import argparse
import io
from typing import Tuple, Optional

import numpy as np
import pandas as pd
import requests
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    RocCurveDisplay,
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt

# --------------------------------------------------------------
# Optional Streamlit import with graceful fallback
# --------------------------------------------------------------
try:
    import streamlit as st  # type: ignore
    HAS_STREAMLIT = True
    cache_data = st.cache_data
except Exception:  # ModuleNotFoundError or anything else
    HAS_STREAMLIT = False

    def cache_data(*args, **kwargs):  # no-op decorator
        def deco(fn):
            return fn
        return deco

    class _Dummy:
        session_state = {}

    st = _Dummy()  # minimal object so code that reads session_state still works

# --------------------------------------------------------------
# Data sources
# --------------------------------------------------------------
FUEL_POVERTY_CSV_URL = (
    "https://datamillnorth.org/download/2j70l/5dl/Fuel%20poverty%20by%20LSOA.csv"
)
IMD_FILE7_CSV_URL = (
    "https://assets.publishing.service.gov.uk/media/5d8d7d68e5274a0d7d66b3bc/"
    "File_7_-_All_ranks__deciles_and_scores_for_the_indices_of_deprivation__and_population_denominators.csv"
)

# --------------------------------------------------------------
# Loaders (with caching where supported)
# --------------------------------------------------------------
@cache_data(show_spinner=True)
def load_fuel_poverty() -> pd.DataFrame:
    r = requests.get(FUEL_POVERTY_CSV_URL, timeout=60)
    r.raise_for_status()
    df = pd.read_csv(io.BytesIO(r.content))
    df.columns = [c.strip().replace(" ", "_") for c in df.columns]
    keep = [
        "Region",
        "LA_Code",
        "LA_Name",
        "LSOA_Code",
        "LSOA_Name",
        "Estimated_number_of_households",
        "Estimated_number_of_households_in_fuel_poverty",
        "Proportion_of_households_fuel_poor_(%)",
    ]
    df = df[keep].dropna(subset=["LSOA_Code"]).drop_duplicates("LSOA_Code")
    for col in [
        "Estimated_number_of_households",
        "Estimated_number_of_households_in_fuel_poverty",
        "Proportion_of_households_fuel_poor_(%)",
    ]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df

@cache_data(show_spinner=True)
def load_imd() -> pd.DataFrame:
    r = requests.get(IMD_FILE7_CSV_URL, timeout=60)
    r.raise_for_status()
    df = pd.read_csv(io.BytesIO(r.content))
    df.columns = [c.strip().replace(" ", "_") for c in df.columns]

    # Determine LSOA code column
    if "LSOA_code_(2011)" in df.columns:
        code_col = "LSOA_code_(2011)"
    elif "LSOA11CD" in df.columns:
        code_col = "LSOA11CD"
    else:
        candidates = [c for c in df.columns if "LSOA" in c and "code" in c.lower()]
        code_col = candidates[0] if candidates else df.columns[0]

    df = df[df[code_col].astype(str).str.startswith("E010")].copy()

    wanted_cols = [code_col]
    labels = [
        "Index_of_Multiple_Deprivation_(IMD)_Decile_(where_1_is_most_deprived_10%_of_LSOAs)",
        "Income_Decile_(where_1_is_most_deprived_10%_of_LSOAs)",
        "Employment_Decile_(where_1_is_most_deprived_10%_of_LSOAs)",
        "Education,_Skills_and_Training_Decile_(where_1_is_most_deprived_10%_of_LSOAs)",
        "Health_Deprivation_and_Disability_Decile_(where_1_is_most_deprived_10%_of_LSOAs)",
        "Crime_Decile_(where_1_is_most_deprived_10%_of_LSOAs)",
        "Barriers_to_Housing_and_Services_Decile_(where_1_is_most_deprived_10%_of_LSOAs)",
        "Living_Environment_Decile_(where_1_is_most_deprived_10%_of_LSOAs)",
    ]
    for lbl in labels:
        for c in df.columns:
            if c.replace(" ", "_") == lbl:
                wanted_cols.append(c)
                break

    if len(wanted_cols) < 5:
        decile_cols = [
            c for c in df.columns if c.endswith("Decile_(where_1_is_most_deprived_10%_of_LSOAs)")
        ]
        wanted_cols = list({code_col} | set(decile_cols))

    feat = df[wanted_cols].copy()
    feat.columns = ["LSOA_Code"] + [f"{i:02d}_" + c.split("_Decile")[0] for i, c in enumerate(feat.columns[1:], start=1)]
    for c in feat.columns:
        if c != "LSOA_Code":
            feat[c] = pd.to_numeric(feat[c], errors="coerce")
    return feat

# --------------------------------------------------------------
# Data building (supports dependency injection for tests)
# --------------------------------------------------------------
def build_dataset(
    threshold_pct: float,
    top_quantile: float,
    fuel: Optional[pd.DataFrame] = None,
    imd: Optional[pd.DataFrame] = None,
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    if fuel is None:
        fuel = load_fuel_poverty()
    if imd is None:
        imd = load_imd()

    df = fuel.merge(imd, on="LSOA_Code", how="inner")

    y_threshold = df["Proportion_of_households_fuel_poor_(%)"] >= threshold_pct
    quant_cut = df["Proportion_of_households_fuel_poor_(%)"].quantile(top_quantile)
    y_quant = df["Proportion_of_households_fuel_poor_(%)"] >= quant_cut

    mode = st.session_state.get("label_mode", "quantile")
    y = y_quant if mode == "quantile" else y_threshold
    y = y.astype(int)

    X = imd.drop(columns=["LSOA_Code"]).reindex(df.index)

    meta = df[[
        "Region",
        "LA_Code",
        "LA_Name",
        "LSOA_Code",
        "LSOA_Name",
        "Estimated_number_of_households",
        "Estimated_number_of_households_in_fuel_poverty",
        "Proportion_of_households_fuel_poor_(%)",
    ]].copy()

    return X, y, meta

# --------------------------------------------------------------
# Tiny synthetic sample for tests and offline runs
# --------------------------------------------------------------
def make_tiny_sample(n: int = 12) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(7)
    lsoas = [f"E010{str(10000+i)}" for i in range(n)]
    regions = ["North East", "North West", "Yorkshire and The Humber", "East Midlands"]
    las = ["LA01", "LA02", "LA03"]

    fuel = pd.DataFrame({
        "Region": [regions[i % len(regions)] for i in range(n)],
        "LA_Code": [las[i % len(las)] for i in range(n)],
        "LA_Name": [f"LA-{i%3+1}" for i in range(n)],
        "LSOA_Code": lsoas,
        "LSOA_Name": [f"LSOA {i+1}" for i in range(n)],
        "Estimated_number_of_households": rng.integers(600, 2200, size=n),
        "Estimated_number_of_households_in_fuel_poverty": rng.integers(50, 400, size=n),
        "Proportion_of_households_fuel_poor_(%)": rng.uniform(5, 35, size=n).round(2),
    })

    imd = pd.DataFrame({
        "LSOA_Code": lsoas,
        "01_IMD": rng.integers(1, 10, size=n),
        "02_Income": rng.integers(1, 10, size=n),
        "03_Employment": rng.integers(1, 10, size=n),
        "04_Education": rng.integers(1, 10, size=n),
        "05_Health": rng.integers(1, 10, size=n),
        "06_Crime": rng.integers(1, 10, size=n),
        "07_Barriers": rng.integers(1, 10, size=n),
        "08_LivingEnv": rng.integers(1, 10, size=n),
    })
    return fuel, imd

# --------------------------------------------------------------
# Modeling helpers
# --------------------------------------------------------------
def build_model(name: str) -> Pipeline:
    name = name.lower()
    if name.startswith("logistic"):
        return Pipeline([
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=400))
        ])
    elif name.startswith("random"):
        return Pipeline([
            ("model", RandomForestClassifier(n_estimators=400, random_state=42, class_weight="balanced"))
        ])
    else:
        raise ValueError("Unknown model. Use 'Logistic Regression' or 'Random Forest'.")

# --------------------------------------------------------------
# Streamlit UI
# --------------------------------------------------------------
if HAS_STREAMLIT:
    st.set_page_config(page_title="Energy Poverty Risk Predictor", layout="wide")
    st.title("Energy Poverty Risk Predictor - England (LSOA)")
    st.caption("Merge fuel poverty (2022) with IMD 2019 deciles, build a simple classifier, and surface at-risk areas.")

    with st.sidebar:
        st.header("Labelling and Model Settings")
        label_mode = st.radio("Label mode", ["quantile", "absolute %"], index=0)
        st.session_state["label_mode"] = "quantile" if label_mode == "quantile" else "absolute"

        if label_mode == "quantile":
            q = st.slider("Top quantile as high risk", 0.50, 0.95, 0.75, 0.05)
            threshold = st.number_input("Absolute % threshold (unused in quantile mode)", value=15.0)
        else:
            q = 0.75
            threshold = st.number_input("Absolute % threshold", value=15.0)

        model_choice = st.selectbox("Model", ["Logistic Regression", "Random Forest"], index=1)
        test_size = st.slider("Test size", 0.1, 0.4, 0.2, 0.05)

    X, y, meta = build_dataset(threshold_pct=threshold, top_quantile=q)

    X_train, X_test, y_train, y_test, meta_train, meta_test = train_test_split(
        X, y, meta, test_size=test_size, random_state=42, stratify=y
    )

    clf = build_model(model_choice)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    try:
        y_proba = clf.predict_proba(X_test)[:, 1]
    except Exception:
        y_proba = None

    col1, col2 = st.columns([1, 1])
    with col1:
        st.subheader("Evaluation")
        st.write(pd.DataFrame(confusion_matrix(y_test, y_pred),
                              index=["Actual 0", "Actual 1"],
                              columns=["Pred 0", "Pred 1"]))
        st.text("Classification report:\n" + classification_report(y_test, y_pred, zero_division=0))

    with col2:
        if y_proba is not None:
            auc = roc_auc_score(y_test, y_proba)
            st.metric("ROC AUC", f"{auc:.3f}")
            fig, ax = plt.subplots()
            RocCurveDisplay.from_predictions(y_test, y_proba, ax=ax)
            st.pyplot(fig)

    st.subheader("Feature importance or coefficients")
    if hasattr(clf.named_steps.get("model"), "feature_importances_"):
        importances = clf.named_steps["model"].feature_importances_
        imp_df = pd.DataFrame({"feature": list(X.columns), "importance": importances}).sort_values("importance", ascending=False)
    else:
        coefs = getattr(clf.named_steps.get("model"), "coef_", None)
        if coefs is not None:
            importances = np.abs(coefs[0])
            imp_df = pd.DataFrame({"feature": list(X.columns), "importance": importances}).sort_values("importance", ascending=False)
        else:
            imp_df = pd.DataFrame({"feature": list(X.columns), "importance": 0})
    st.write(imp_df.head(15))

    st.subheader("Ranked areas by predicted risk (test set)")
    rank_df = meta_test.copy()
    rank_df["y_true"] = y_test.values
    rank_df["y_pred"] = y_pred
    if y_proba is not None:
        rank_df["risk_score"] = y_proba
        rank_df = rank_df.sort_values("risk_score", ascending=False)
    else:
        rank_df = rank_df.sort_values("y_pred", ascending=False)

    st.dataframe(rank_df.head(50), use_container_width=True)

    with st.expander("Filter by Local Authority or Region"):
        la = st.selectbox("Local Authority", ["(all)"] + sorted(meta_test.LA_Name.unique().tolist()))
        region = st.selectbox("Region", ["(all)"] + sorted(meta_test.Region.unique().tolist()))
        filtered = rank_df.copy()
        if la != "(all)":
            filtered = filtered[filtered["LA_Name"] == la]
        if region != "(all)":
            filtered = filtered[filtered["Region"] == region]
        st.write(filtered.head(100))

    csv = rank_df.to_csv(index=False).encode("utf-8")
    st.download_button("Download ranked predictions (CSV)", csv, file_name="energy_poverty_risk_rankings.csv")

    st.markdown(
        """
**Data sources**
- Fuel poverty by LSOA (England, 2022), Data Mill North: https://datamillnorth.org/dataset/fuel-poverty-by-lsoa-england-2j70l
- English Indices of Deprivation 2019, File 7 (all ranks, deciles, scores): https://www.gov.uk/government/statistics/english-indices-of-deprivation-2019

**Caveats**
- Labels here are proxy thresholds, not official designations.
- IMD 2019 predates 2022 fuel poverty estimates; treat relationships as correlational.
- LSOA estimates are modelled and volatile at small area level.
"""
    )

# --------------------------------------------------------------
# CLI fallback and tests
# --------------------------------------------------------------
def cli_run(model_choice: str = "Random Forest", label_mode: str = "quantile", q: float = 0.75, threshold: float = 15.0, test_size: float = 0.2):
    # Emulate Streamlit session state for label mode
    st.session_state["label_mode"] = "quantile" if label_mode == "quantile" else "absolute"

    # Try online data first; if it fails, fall back to synthetic sample
    try:
        X, y, meta = build_dataset(threshold_pct=threshold, top_quantile=q)
    except Exception as e:
        print("[CLI] Online data failed (", e, ") -> falling back to tiny synthetic sample.")
        fuel, imd = make_tiny_sample()
        X, y, meta = build_dataset(threshold_pct=threshold, top_quantile=q, fuel=fuel, imd=imd)

    X_train, X_test, y_train, y_test, meta_train, meta_test = train_test_split(
        X, y, meta, test_size=test_size, random_state=42, stratify=y
    )

    clf = build_model(model_choice)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    try:
        y_proba = clf.predict_proba(X_test)[:, 1]
    except Exception:
        y_proba = None

    print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))
    print("\nClassification report:\n", classification_report(y_test, y_pred, zero_division=0))
    if y_proba is not None:
        auc = roc_auc_score(y_test, y_proba)
        print(f"ROC AUC: {auc:.3f}")

    rank_df = meta_test.copy()
    rank_df["y_true"] = y_test.values
    rank_df["y_pred"] = y_pred
    if y_proba is not None:
        rank_df["risk_score"] = y_proba
        rank_df = rank_df.sort_values("risk_score", ascending=False)
    else:
        rank_df = rank_df.sort_values("y_pred", ascending=False)

    rank_df.to_csv("energy_poverty_risk_rankings.csv", index=False)
    print("Saved rankings to energy_poverty_risk_rankings.csv")


def run_smoke_tests():
    print("Running smoke tests on tiny synthetic sample...")
    fuel, imd = make_tiny_sample(n=20)

    # Test 1: dataset build returns aligned shapes
    st.session_state["label_mode"] = "quantile"
    X, y, meta = build_dataset(threshold_pct=15.0, top_quantile=0.75, fuel=fuel, imd=imd)
    assert len(X) == len(y) == len(meta) > 0, "Dataset sizes must align and be nonempty"

    # Test 2: both models train and predict
    for model_choice in ["Logistic Regression", "Random Forest"]:
        clf = build_model(model_choice)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0, stratify=y)
        clf.fit(X_train, y_train)
        preds = clf.predict(X_test)
        assert preds.shape[0] == y_test.shape[0], "Predictions must match test labels length"

    # Test 3: threshold mode also works
    st.session_state["label_mode"] = "absolute"
    X2, y2, meta2 = build_dataset(threshold_pct=20.0, top_quantile=0.75, fuel=fuel, imd=imd)
    assert (y2.isin([0, 1])).all(), "Labels must be binary"

    print("All smoke tests passed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cli", action="store_true", help="Run in CLI mode instead of Streamlit UI")
    parser.add_argument("--test", action="store_true", help="Run built-in smoke tests")
    parser.add_argument("--model", default="Random Forest", choices=["Random Forest", "Logistic Regression"], help="Model choice for CLI run")
    parser.add_argument("--label", default="quantile", choices=["quantile", "absolute"], help="Labeling mode for CLI run")
    parser.add_argument("--q", type=float, default=0.75, help="Top quantile for high risk (quantile mode)")
    parser.add_argument("--threshold", type=float, default=15.0, help="Absolute percent threshold for high risk")
    parser.add_argument("--test_size", type=float, default=0.2, help="Test size for train-test split")
    args = parser.parse_args()

    if args.test:
        run_smoke_tests()
    if args.cli:
        cli_run(model_choice=args.model, label_mode=args.label, q=args.q, threshold=args.threshold, test_size=args.test_size)
    if not args.cli and not args.test and not HAS_STREAMLIT:
        # No Streamlit available and no flags passed -> fall back to CLI
        print("Streamlit not available. Falling back to CLI run. Use --cli explicitly to silence this message.")
        cli_run(model_choice=args.model, label_mode=args.label, q=args.q, threshold=args.threshold, test_size=args.test_size)
