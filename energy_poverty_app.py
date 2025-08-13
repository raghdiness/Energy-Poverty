# energy_poverty_app.py
# Energy Poverty Risk Predictor — hardened for multiple CSV variants
# - Handles LSOA 2011/2021 code headers and Local Authority code variants
# - Normalizes fuel-poverty column names across official CSVs
# - Derives missing columns from the others (proportion, counts, totals)
# - Graceful diagnostics instead of hard crashes
# - CLI: `--cli` to export CSV, `--test` to run normalization tests

from __future__ import annotations

import argparse
import io
import math
import sys
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

# --- Optional: scikit-learn for the model UI (installed via requirements.txt) ---
try:
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import roc_auc_score, classification_report
    from sklearn.linear_model import LogisticRegression
    SKLEARN_AVAILABLE = True
except Exception:
    SKLEARN_AVAILABLE = False

# --- Streamlit (app will still provide CLI/tests without it) ---
try:
    import streamlit as st
    STREAMLIT_AVAILABLE = True
except Exception:
    STREAMLIT_AVAILABLE = False

# --------------------------------------------------------------------------------------
# Column normalization utilities
# --------------------------------------------------------------------------------------

def _norm(s: str) -> str:
    return ''.join(ch for ch in s.lower() if ch.isalnum())

# Known header variants across official releases (examples; not exhaustive)
LSOA_CANDIDATES = [
    'LSOA_Code', 'LSOA code', 'LSOA Code', 'LSOA11CD', 'LSOA code (2011)', 'LSOA21CD', 'LSOA code (2021)',
    'LSOA_CODE', 'lsoa_code'
]
LA_CANDIDATES = [
    'Local_Authority_Code', 'Local Authority Code', 'LAD code', 'LAD22CD', 'LAD23CD', 'LAD21CD', 'LAD20CD', 'LAD19CD',
    'LAD17CD', 'LA_Code', 'LADCD'
]
# Fuel poverty measures
TOT_HH_CANDIDATES = [
    'Estimated_number_of_households', 'Estimated number of households', 'Total households', 'Households', 'TotHH'
]
FP_HH_CANDIDATES = [
    'Estimated_number_of_households_in_fuel_poverty', 'Estimated number of households in fuel poverty',
    'Households in fuel poverty', 'Fuel poor households', 'FP households', 'Num fuel poor'
]
PROP_FP_CANDIDATES = [
    'Proportion_of_households_fuel_poor_(%)', 'Proportion of households fuel poor (%)',
    'Fuel poverty rate (%)', 'Fuel poverty proportion (%)', 'Fuel poverty (%)', 'FP rate (%)'
]

@dataclass
class FuelColumns:
    lsoa: str
    la: Optional[str]
    total_hh: Optional[str]
    fp_hh: Optional[str]
    prop_pct: Optional[str]


def find_first_present(df: pd.DataFrame, candidates: Iterable[str]) -> Optional[str]:
    cols = {_norm(c): c for c in df.columns}
    for cand in candidates:
        key = _norm(cand)
        if key in cols:
            return cols[key]
    return None


def identify_fuel_columns(df: pd.DataFrame) -> FuelColumns:
    lsoa = find_first_present(df, LSOA_CANDIDATES)
    la = find_first_present(df, LA_CANDIDATES)
    total_hh = find_first_present(df, TOT_HH_CANDIDATES)
    fp_hh = find_first_present(df, FP_HH_CANDIDATES)
    prop_pct = find_first_present(df, PROP_FP_CANDIDATES)
    if not lsoa:
        raise KeyError("Could not find an LSOA code column. Saw: " + ', '.join(map(str, df.columns)))
    return FuelColumns(lsoa, la, total_hh, fp_hh, prop_pct)


def coerce_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors='coerce')


# --------------------------------------------------------------------------------------
# Data loaders — keep your existing URLs; these functions just normalize after read_csv
# --------------------------------------------------------------------------------------
DEFAULT_FUEL_URL = (
    # Tip: set st.secrets["FUEL_URL"] to override in Cloud without code changes
    None
)
DEFAULT_IMD_URL = None  # Your IMD loader can stay as-is if you already have a URL


@st.cache_data(show_spinner=True) if STREAMLIT_AVAILABLE else (lambda f: f)
def load_fuel_poverty(url: Optional[str] = DEFAULT_FUEL_URL, *, uploaded: Optional[io.BytesIO] = None) -> pd.DataFrame:
    """Load fuel poverty data from a URL or uploaded CSV and normalize headers/derived fields.
    Expected outputs after normalization:
    - 'LSOA_Code'
    - 'Local_Authority_Code' (optional)
    - 'Estimated_number_of_households'
    - 'Estimated_number_of_households_in_fuel_poverty'
    - 'Proportion_of_households_fuel_poor_(%)'
    """
    if uploaded is not None:
        df = pd.read_csv(uploaded)
    else:
        if url is None:
            # Fall back: ask user to supply file/URL in the UI. We still return an empty frame here.
            return pd.DataFrame()
        df = pd.read_csv(url)

    cols = identify_fuel_columns(df)

    # Start by copying through the minimally required columns
    out = pd.DataFrame()
    out['LSOA_Code'] = df[cols.lsoa].astype(str).str.strip()
    if cols.la:
        out['Local_Authority_Code'] = df[cols.la].astype(str).str.strip()

    # Bring over whatever exists
    if cols.total_hh:
        out['Estimated_number_of_households'] = coerce_numeric(df[cols.total_hh])
    if cols.fp_hh:
        out['Estimated_number_of_households_in_fuel_poverty'] = coerce_numeric(df[cols.fp_hh])
    if cols.prop_pct:
        out['Proportion_of_households_fuel_poor_(%)'] = coerce_numeric(df[cols.prop_pct])

    # Derive missing pieces from the others
    tot = out.get('Estimated_number_of_households')
    fp = out.get('Estimated_number_of_households_in_fuel_poverty')
    prop = out.get('Proportion_of_households_fuel_poor_(%)')

    if tot is None and fp is not None and prop is not None:
        denom = prop / 100.0
        denom = denom.replace(0, np.nan)
        out['Estimated_number_of_households'] = fp / denom
    if fp is None and tot is not None and prop is not None:
        out['Estimated_number_of_households_in_fuel_poverty'] = (tot * (prop / 100.0))
    if prop is None and tot is not None and fp is not None:
        denom = tot.replace(0, np.nan)
        out['Proportion_of_households_fuel_poor_(%)'] = (fp / denom) * 100.0

    # Final sanity: ensure numeric columns present
    for c in [
        'Estimated_number_of_households',
        'Estimated_number_of_households_in_fuel_poverty',
        'Proportion_of_households_fuel_poor_(%)',
    ]:
        if c not in out.columns:
            out[c] = np.nan
        out[c] = coerce_numeric(out[c])

    # Drop rows without LSOA codes
    out = out.dropna(subset=['LSOA_Code']).drop_duplicates('LSOA_Code')

    return out


@st.cache_data(show_spinner=True) if STREAMLIT_AVAILABLE else (lambda f: f)
def load_imd(url: Optional[str] = DEFAULT_IMD_URL, *, uploaded: Optional[io.BytesIO] = None) -> pd.DataFrame:
    """Load IMD data and normalize an LSOA code column. We keep only LSOA code + IMD rank/decile if present."""
    if uploaded is not None:
        df = pd.read_csv(uploaded)
    else:
        if url is None:
            return pd.DataFrame()
        df = pd.read_csv(url)

    # Try to find an LSOA column
    lsoa_col = find_first_present(df, LSOA_CANDIDATES)
    if not lsoa_col:
        raise KeyError("Could not find an LSOA column in IMD dataset.")

    out = pd.DataFrame({'LSOA_Code': df[lsoa_col].astype(str).str.strip()})

    # Optional IMD fields (many names exist)
    imd_rank = find_first_present(df, ['IMD Rank', 'Index of Multiple Deprivation (Rank)', 'IMD_Rank', 'imd_rank'])
    imd_dec = find_first_present(df, ['IMD Decile', 'IMD_Decile', 'imd_decile', 'Index of Multiple Deprivation (Decile)'])
    if imd_rank:
        out['IMD_Rank'] = coerce_numeric(df[imd_rank])
    if imd_dec:
        out['IMD_Decile'] = coerce_numeric(df[imd_dec])
    return out.dropna(subset=['LSOA_Code']).drop_duplicates('LSOA_Code')


# --------------------------------------------------------------------------------------
# Feature building
# --------------------------------------------------------------------------------------

def build_dataset(
    *,
    threshold_pct: float = 10.0,
    top_quantile: float = 0.2,
    fuel: Optional[pd.DataFrame] = None,
    imd: Optional[pd.DataFrame] = None,
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """Return (X, y, meta) for classification.
    y = 1 if fuel-poverty proportion >= threshold_pct, else 0.
    Features: proportion, IMD features (if present), simple ratios.
    """
    if fuel is None:
        fuel = load_fuel_poverty()
    if imd is None:
        imd = load_imd()

    if fuel.empty:
        raise ValueError("Fuel dataset is empty — provide a URL or upload a CSV via the sidebar.")

    df = fuel.copy()

    # Label
    df['label'] = (df['Proportion_of_households_fuel_poor_(%)'] >= threshold_pct).astype(int)

    # Merge IMD
    if imd is not None and not imd.empty:
        df = df.merge(imd, on='LSOA_Code', how='left')

    # Simple engineered features
    df['fp_rate'] = df['Proportion_of_households_fuel_poor_(%)'] / 100.0
    # Densities unavailable without area; for now keep a minimal robust set

    # X / y / meta
    feature_candidates = [
        'fp_rate',
        'IMD_Rank',
        'IMD_Decile',
        'Estimated_number_of_households',
    ]
    X_cols = [c for c in feature_candidates if c in df.columns]
    X = df[X_cols].fillna(df[X_cols].median(numeric_only=True))
    y = df['label']

    meta_cols = ['LSOA_Code']
    if 'Local_Authority_Code' in df.columns:
        meta_cols.append('Local_Authority_Code')
    meta = df[meta_cols]

    return X, y, meta


# --------------------------------------------------------------------------------------
# Streamlit UI
# --------------------------------------------------------------------------------------

def app():
    st.set_page_config(page_title="Energy Poverty Risk Predictor", layout='wide')
    st.title("Energy Poverty Risk Predictor")
    st.caption("Hardened data loader that accepts multiple official CSV header variants.")

    with st.sidebar:
        st.subheader("Data input")
        use_custom = st.toggle("Provide data files/URLs", value=False, help="If off, the app expects repo/default URLs inside the code or secrets.")
        fuel_upload = None
        imd_upload = None
        fuel_url = None
        imd_url = None
        if use_custom:
            fuel_upload = st.file_uploader("Fuel-poverty CSV", type=['csv'])
            fuel_url = st.text_input("...or Fuel CSV URL (optional)") or None
            imd_upload = st.file_uploader("IMD CSV (optional)", type=['csv'])
            imd_url = st.text_input("...or IMD CSV URL (optional)") or None

        st.subheader("Labeling")
        threshold = st.slider("Fuel-poverty threshold (%)", min_value=5.0, max_value=30.0, value=10.0, step=0.5)
        test_size = st.slider("Test size", 0.1, 0.4, 0.2, 0.05)
        model_choice = st.selectbox("Model", ["Logistic Regression"])  # Keep simple for MVP

    # Load datasets with resilience
    try:
        fuel = load_fuel_poverty(url=fuel_url, uploaded=fuel_upload)
    except Exception as e:
        st.error(f"Fuel dataset load error: {e}")
        st.stop()

    try:
        imd = load_imd(url=imd_url, uploaded=imd_upload) if (imd_upload or imd_url) else pd.DataFrame()
    except Exception as e:
        st.warning(f"IMD dataset load problem (continuing without IMD): {e}")
        imd = pd.DataFrame()

    if fuel.empty:
        with st.expander("Diagnostics: fuel dataset is empty", expanded=True):
            st.write("No data loaded. Provide a CSV or set a URL in the sidebar or code.")
        st.stop()

    # Show diagnostics
    with st.expander("Diagnostics: fuel dataset preview", expanded=False):
        st.write(f"Rows: {len(fuel):,}")
        st.dataframe(fuel.head(20))
        st.code("
".join(map(str, fuel.columns.tolist())), language='text')

    # Build features
    try:
        X, y, meta = build_dataset(threshold_pct=threshold, fuel=fuel, imd=imd)
    except Exception as e:
        st.error(f"Dataset build error: {e}")
        st.stop()

    st.subheader("Training & Evaluation")
    if not SKLEARN_AVAILABLE:
        st.warning("scikit-learn is not available. The dashboard will show data only.")
        st.dataframe(pd.concat([meta.reset_index(drop=True), y.rename('label')], axis=1).head(25))
        st.stop()

    # Train/test split
    X_train, X_test, y_train, y_test, meta_train, meta_test = train_test_split(
        X, y, meta, test_size=test_size, random_state=42, stratify=y if y.nunique() > 1 else None
    )

    # Simple model
    model = LogisticRegression(max_iter=1000, n_jobs=1)
    model.fit(X_train, y_train)
    proba = model.predict_proba(X_test)[:, 1]
    preds = (proba >= 0.5).astype(int)

    col1, col2 = st.columns(2)
    with col1:
        try:
            auc = roc_auc_score(y_test, proba) if y_test.nunique() > 1 else float('nan')
        except ValueError:
            auc = float('nan')
        st.metric("ROC AUC", f"{auc:.3f}" if not math.isnan(auc) else "n/a")
    with col2:
        st.metric("Positives in test", int(y_test.sum()))

    st.code(classification_report(y_test, preds, zero_division=0), language='text')

    # Export
    export_df = meta_test.copy()
    export_df['pred_proba'] = proba
    export_df['pred_label'] = preds
    st.download_button("Download predictions (CSV)", export_df.to_csv(index=False).encode('utf-8'), file_name='energy_poverty_predictions.csv', mime='text/csv')


# --------------------------------------------------------------------------------------
# Tests (run with: python energy_poverty_app.py --test)
# --------------------------------------------------------------------------------------

def _df_from(headers: Dict[str, Iterable], rows: List[List]) -> pd.DataFrame:
    cols = list(headers.keys())
    df = pd.DataFrame(rows, columns=cols)
    # Rename columns to the headers (some inputs will be odd spellings)
    df.columns = cols
    return df


def run_tests() -> None:
    # 1) Exact expected headers
    df1 = pd.DataFrame({
        'LSOA_Code': ['A', 'B'],
        'Local_Authority_Code': ['L1', 'L2'],
        'Estimated_number_of_households': [1000, 1200],
        'Estimated_number_of_households_in_fuel_poverty': [100, 240],
        'Proportion_of_households_fuel_poor_(%)': [10.0, 20.0],
    })
    n1 = load_fuel_poverty(uploaded=io.BytesIO(df1.to_csv(index=False).encode('utf-8')))
    assert set(['LSOA_Code', 'Estimated_number_of_households', 'Estimated_number_of_households_in_fuel_poverty', 'Proportion_of_households_fuel_poor_(%)']).issubset(n1.columns)

    # 2) Variant headers (common official spellings) without totals — derive totals
    df2 = pd.DataFrame({
        'LSOA11CD': ['X'],
        'LAD22CD': ['LAD'],
        'Estimated number of households in fuel poverty': [50],
        'Proportion of households fuel poor (%)': [25.0],
    })
    n2 = load_fuel_poverty(uploaded=io.BytesIO(df2.to_csv(index=False).encode('utf-8')))
    assert 'Estimated_number_of_households' in n2.columns
    assert math.isclose(float(n2['Estimated_number_of_households'].iloc[0]), 200.0, rel_tol=1e-6)

    # 3) Variant headers with totals + proportion — derive fp count
    df3 = pd.DataFrame({
        'LSOA21CD': ['Y'],
        'Local Authority Code': ['LA'],
        'Total households': [800],
        'Fuel poverty rate (%)': [12.5],
    })
    n3 = load_fuel_poverty(uploaded=io.BytesIO(df3.to_csv(index=False).encode('utf-8')))
    assert 'Estimated_number_of_households_in_fuel_poverty' in n3.columns
    assert math.isclose(float(n3['Estimated_number_of_households_in_fuel_poverty'].iloc[0]), 100.0, rel_tol=1e-6)

    # 4) IMD loader LSOA variants
    imd = pd.DataFrame({'LSOA code (2011)': ['A', 'B'], 'IMD Rank': [1000, 2000]})
    imd_n = load_imd(uploaded=io.BytesIO(imd.to_csv(index=False).encode('utf-8')))
    assert 'LSOA_Code' in imd_n.columns and 'IMD_Rank' in imd_n.columns

    print("All tests passed ✔")


# --------------------------------------------------------------------------------------
# Entrypoint
# --------------------------------------------------------------------------------------
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cli', action='store_true', help='Export a CSV of predictions without a UI')
    parser.add_argument('--test', action='store_true', help='Run quick tests and exit')
    parser.add_argument('--fuel', type=str, default=None, help='Fuel CSV path/URL')
    parser.add_argument('--imd', type=str, default=None, help='IMD CSV path/URL')
    args = parser.parse_args()

    if args.test:
        run_tests()
        sys.exit(0)

    if STREAMLIT_AVAILABLE and not args.cli:
        app()
    else:
        # Minimal CLI: require files, train a tiny model, write CSV
        if not SKLEARN_AVAILABLE:
            print("scikit-learn not available; cannot run CLI model.")
            sys.exit(2)
        # Load via pandas directly (URLs or local paths)
        fuel = load_fuel_poverty(url=args.fuel)
        imd = load_imd(url=args.imd) if args.imd else pd.DataFrame()
        X, y, meta = build_dataset(fuel=fuel, imd=imd)
        X_train, X_test, y_train, y_test, meta_train, meta_test = train_test_split(
            X, y, meta, test_size=0.2, random_state=42, stratify=y if y.nunique() > 1 else None
        )
        model = LogisticRegression(max_iter=1000, n_jobs=1)
        model.fit(X_train, y_train)
        proba = model.predict_proba(X_test)[:, 1]
        out = meta_test.copy()
        out['pred_proba'] = proba
        out['pred_label'] = (proba >= 0.5).astype(int)
        out.to_csv('energy_poverty_predictions.csv', index=False)
        print('Wrote energy_poverty_predictions.csv with', len(out), 'rows')
