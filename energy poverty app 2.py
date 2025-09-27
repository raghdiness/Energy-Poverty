import pandas as pd
import numpy as np
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# --- Data loading and cleaning helpers ---
def load_and_clean_fuel_data(path_or_url: str) -> pd.DataFrame:
    df = pd.read_csv(path_or_url, header=1)
    rename_map = {
        "Number of households": "Estimated number of households",
        "Number of households in fuel poverty": "Estimated number of households in fuel poverty"
    }
    df = df.rename(columns=rename_map)
    return df

# --- Robust train/test split helper (from earlier patch) ---
def smart_train_test_split(X, y, meta, test_size=0.2, random_state=42, allow_autogrow=True):
    import numpy as np
    from sklearn.model_selection import train_test_split as _tts
    classes, counts = np.unique(y, return_counts=True)
    n = len(y)

    def strat_ok(ts):
        if len(classes) < 2:
            return False
        for c_count in counts:
            if c_count < 2:
                return False
            if int(np.floor(ts * c_count)) < 1:
                return False
            if int(np.floor((1.0 - ts) * c_count)) < 1:
                return False
        return True

    if strat_ok(test_size):
        return _tts(X, y, meta, test_size=test_size, random_state=random_state, stratify=y)
    else:
        return _tts(X, y, meta, test_size=test_size, random_state=random_state, stratify=None)

# --- Streamlit app main ---
def main():
    st.title("Energy Poverty Risk Predictor")

fuel_url = st.text_input("Fuel poverty CSV URL or path")

if fuel_url:
    fuel_df = load_and_clean_fuel_data(fuel_url)
merged = fuel_df.copy()
merged["label"] = (
    merged["Proportion of households fuel poor (%)"] >= 10
).astype(int)

drop_cols = [
    "LSOA Code",
    "LSOA Name",
    "Local Authority Code",
    "Local Authority Name",
    "Region",
    "label"
]

X = merged.drop(columns=drop_cols, errors="ignore").select_dtypes(include=["number"])
y = merged["label"].values
meta = merged[["LSOA Code", "LSOA Name"]]

    st.write("Feature dtypes:", X.dtypes)

        # Train/test split
    X_train, X_test, y_train, y_test, meta_train, meta_test = smart_train_test_split(
            X, y, meta, test_size=0.2, random_state=42, allow_autogrow=True
        )

        # Build pipeline
    pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000, class_weight="balanced"))
        ])

    pipe.fit(X_train, y_train)
    acc = pipe.score(X_test, y_test)
    st.write("Test accuracy:", acc)

if __name__ == "__main__":
    main()
