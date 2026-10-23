from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# -------------------------------------------------------------------
# Page setup
# -------------------------------------------------------------------
st.set_page_config(
    page_title="Mileage Lab | MPG Predictor",
    page_icon="🏎️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    :root { color-scheme: dark; }
    .stApp {
        background: radial-gradient(circle at 10% 0%, #182b3b 0, #0b1118 42%, #080c11 100%);
        color: #e8eef5;
    }
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #101c27 0%, #0b121a 100%);
        border-right: 1px solid #263747;
    }
    .hero {
        padding: 1.4rem 1.6rem;
        border: 1px solid #2a4659;
        border-radius: 18px;
        background: linear-gradient(120deg, rgba(21,48,66,.96), rgba(15,23,34,.96));
        margin-bottom: 1.1rem;
    }
    .eyebrow { color: #46d9c4; letter-spacing: .16em; font-size: .76rem; font-weight: 800; }
    .hero h1 { margin: .25rem 0 .4rem 0; font-size: clamp(2rem, 4vw, 3.1rem); }
    .hero p { color: #b9c9d8; margin: 0; }
    .metric-card {
        background: #111d28; border: 1px solid #263d4e;
        border-radius: 14px; padding: 1rem 1.1rem;
    }
    .small-label { color: #9eb1c1; font-size: .82rem; }
    .big-number { color: #47dfc6; font-size: 1.8rem; font-weight: 800; }
    div[data-testid="stForm"] {
        background: rgba(17, 29, 40, .72); border: 1px solid #263d4e;
        padding: 1rem; border-radius: 16px;
    }
    .stButton button[kind="primary"], .stFormSubmitButton button {
        background: #22cbb2; color: #07131a; border: 0; font-weight: 800;
        border-radius: 10px; min-height: 2.8rem;
    }
    .stButton button[kind="primary"]:hover, .stFormSubmitButton button:hover {
        background: #4ce6d0; color: #07131a;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# -------------------------------------------------------------------
# Data loading and preparation
# -------------------------------------------------------------------
DEFAULT_DATA = Path(__file__).with_name("auto-mpg.csv")


def read_dataset(uploaded_file=None):
    """Load CSV from upload or bundled auto-mpg.csv."""
    if uploaded_file is not None:
        frame = pd.read_csv(uploaded_file)
    elif DEFAULT_DATA.exists():
        frame = pd.read_csv(DEFAULT_DATA)
    else:
        return None

    frame.columns = [str(col).strip() for col in frame.columns]
    # Locate the target regardless of capitalization or surrounding spaces.
    target_matches = [col for col in frame.columns if col.strip().lower() == "mpg"]
    if not target_matches:
        raise ValueError("The dataset must contain a target column named 'MPG' (or 'mpg').")

    target = target_matches[0]
    frame = frame.rename(columns={target: "MPG"})

    # The Auto MPG dataset commonly represents missing horsepower as '?'.
    if "horsepower" in [c.lower() for c in frame.columns]:
        hp_col = next(c for c in frame.columns if c.lower() == "horsepower")
        frame[hp_col] = pd.to_numeric(frame[hp_col].replace("?", np.nan), errors="coerce")

    # Convert columns that look numeric but were read as text, while preserving
    # genuine categorical columns such as car name.
    for col in frame.columns:
        if col != "MPG" and frame[col].dtype == "object":
            cleaned = frame[col].replace("?", np.nan)
            converted = pd.to_numeric(cleaned, errors="coerce")
            if converted.notna().sum() >= max(1, int(cleaned.notna().sum() * 0.85)):
                frame[col] = converted

    frame["MPG"] = pd.to_numeric(frame["MPG"], errors="coerce")
    frame = frame.dropna(subset=["MPG"])
    if frame.empty:
        raise ValueError("No usable rows remain after checking the MPG target.")
    return frame


@st.cache_resource(show_spinner="Building the mileage prediction pipeline...")
def train_model(csv_bytes, random_state=42):
    """Fit a preprocessing + regression pipeline and return model metrics."""
    from io import BytesIO

    data = read_dataset(BytesIO(csv_bytes))
    X = data.drop(columns="MPG")
    y = data["MPG"]

    # Treat low-cardinality text columns as categories; numerical columns stay numeric.
    categorical_cols = [
        col for col in X.columns
        if X[col].dtype == "object" or X[col].dtype.name == "category"
    ]
    numeric_cols = [col for col in X.columns if col not in categorical_cols]

    numeric_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    preprocessing = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipe, numeric_cols),
            ("categorical", categorical_pipe, categorical_cols),
        ],
        remainder="drop",
    )
    pipeline = Pipeline(
        steps=[
            ("preprocessing", preprocessing),
            ("regressor", RandomForestRegressor(
                n_estimators=250,
                min_samples_leaf=2,
                random_state=random_state,
                n_jobs=-1,
            )),
        ]
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state
    )
    pipeline.fit(X_train, y_train)
    predictions = pipeline.predict(X_test)
    metrics = {
        "mae": float(mean_absolute_error(y_test, predictions)),
        "r2": float(r2_score(y_test, predictions)),
        "rows": int(len(data)),
        "features": list(X.columns),
        "numeric_cols": numeric_cols,
        "categorical_cols": categorical_cols,
        "data": data,
        "test_size": int(len(y_test)),
    }
    # Fit a final model on all available rows for the prediction form.
    pipeline.fit(X, y)
    return pipeline, metrics


def feature_input(col, series):
    """Render a suitable Streamlit input for one feature."""
    clean = series.dropna()
    if series.dtype == "object" or series.dtype.name == "category":
        options = sorted(clean.astype(str).unique().tolist())
        if not options:
            return st.text_input(col.replace("_", " ").title(), value="")
        return st.selectbox(
            col.replace("_", " ").title(),
            options=options,
            help=f"Known values in the dataset: {len(options)}",
        )

    if pd.api.types.is_integer_dtype(series.dtype):
        low = int(clean.min()) if not clean.empty else 0
        high = int(clean.max()) if not clean.empty else 100
        default = int(clean.median()) if not clean.empty else 0
        # A little breathing room allows realistic values outside the training range.
        span = max(high - low, 10)
        return st.number_input(
            col.replace("_", " ").title(),
            min_value=int(low - span * 0.2),
            max_value=int(high + span * 0.2),
            value=default,
            step=1,
        )

    low = float(clean.min()) if not clean.empty else 0.0
    high = float(clean.max()) if not clean.empty else 100.0
    default = float(clean.median()) if not clean.empty else 0.0
    span = max(high - low, 1.0)
    return st.number_input(
        col.replace("_", " ").title(),
        min_value=float(low - span * 0.2),
        max_value=float(high + span * 0.2),
        value=default,
        step=0.1,
        format="%.2f",
    )


# -------------------------------------------------------------------
# Interface
# -------------------------------------------------------------------
st.markdown(
    """
    <div class="hero">
      <div class="eyebrow">AUTOMOTIVE ANALYTICS • MACHINE LEARNING</div>
      <h1>🏎️ Mileage Lab</h1>
      <p>Estimate a vehicle's fuel efficiency using an end-to-end machine-learning pipeline.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.markdown("## 🛠️ Control Garage")
    st.caption("Choose the dataset used to train the model.")
    uploaded = st.file_uploader("Upload a CSV dataset", type=["csv"])
    st.divider()
    st.markdown("**Pipeline**")
    st.caption("Missing-value imputation · scaling · categorical encoding · Random Forest")
    st.caption("Target column: `MPG`")

try:
    data = read_dataset(uploaded)
except Exception as exc:
    st.error(f"Could not read the dataset: {exc}")
    st.stop()

if data is None:
    st.warning("Upload your Auto MPG CSV in the sidebar, or place `auto-mpg.csv` beside `app.py`.")
    st.stop()

# Use a stable cache key for uploaded or bundled CSV bytes.
csv_bytes = uploaded.getvalue() if uploaded is not None else DEFAULT_DATA.read_bytes()
try:
    model, info = train_model(csv_bytes)
except Exception as exc:
    st.error(f"Model training failed: {exc}")
    st.stop()

m1, m2, m3, m4 = st.columns(4)
with m1:
    st.markdown(f'<div class="metric-card"><div class="small-label">Vehicles in dataset</div><div class="big-number">{info["rows"]:,}</div></div>', unsafe_allow_html=True)
with m2:
    st.markdown(f'<div class="metric-card"><div class="small-label">Input features</div><div class="big-number">{len(info["features"])}</div></div>', unsafe_allow_html=True)
with m3:
    st.markdown(f'<div class="metric-card"><div class="small-label">Test MAE</div><div class="big-number">{info["mae"]:.2f}</div><div class="small-label">MPG average error</div></div>', unsafe_allow_html=True)
with m4:
    r2_text = f'{info["r2"]:.3f}' if np.isfinite(info["r2"]) else "N/A"
    st.markdown(f'<div class="metric-card"><div class="small-label">Test R² score</div><div class="big-number">{r2_text}</div></div>', unsafe_allow_html=True)

st.write("")
left, right = st.columns([1.2, 0.8], gap="large")

with left:
    st.subheader("🔧 Configure your vehicle")
    st.caption("Set the specifications below, then run the estimator.")
    with st.form("prediction_form"):
        values = {}
        feature_cols = info["features"]
        cols = st.columns(2)
        for idx, col in enumerate(feature_cols):
            with cols[idx % 2]:
                values[col] = feature_input(col, info["data"][col])
        submitted = st.form_submit_button("⚡ Predict mileage", type="primary", use_container_width=True)

    if submitted:
        input_df = pd.DataFrame([values], columns=feature_cols)
        prediction = float(model.predict(input_df)[0])
        st.session_state["last_prediction"] = prediction
        st.session_state["last_vehicle"] = values

    if "last_prediction" in st.session_state:
        pred = st.session_state["last_prediction"]
        st.markdown(
            f"""
            <div class="hero" style="margin-top:1rem;text-align:center">
              <div class="eyebrow">ESTIMATED FUEL EFFICIENCY</div>
              <div style="font-size:3.5rem;font-weight:900;color:#47dfc6;line-height:1.25">{pred:.1f}</div>
              <div style="font-size:1.1rem;color:#c7d5e1">miles per gallon (MPG)</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.caption("This is a model estimate, not a guaranteed real-world result.")

with right:
    st.subheader("📊 Dataset overview")
    st.caption("A quick look at the data currently used for training.")
    st.dataframe(info["data"].head(8), use_container_width=True, hide_index=True)
    st.markdown("**Missing values by column**")
    missing = info["data"].isna().sum()
    missing = missing[missing > 0]
    if missing.empty:
        st.success("No missing values remain in the displayed dataset. Any feature gaps are imputed inside the pipeline.")
    else:
        st.dataframe(
            missing.rename("Missing rows").reset_index().rename(columns={"index": "Feature"}),
            use_container_width=True,
            hide_index=True,
        )
    st.markdown("**How the pipeline works**")
    st.markdown(
        """
        1. Reads the CSV and identifies `MPG` as the target.
        2. Converts placeholder values such as `?` to missing values.
        3. Imputes numeric values with the median and categories with the most frequent value.
        4. Scales numeric features and one-hot encodes categorical features.
        5. Trains a Random Forest regressor and reports holdout-set metrics.
        """
    )

with st.expander("ℹ️ About this model"):
    st.write(
        "The test MAE is the average absolute difference between predicted and actual MPG on a held-out test split. "
        "The R² score describes how much variation in MPG the model explains on that split. "
        "Model performance depends on dataset quality and may differ on vehicles outside the training data."
    )
