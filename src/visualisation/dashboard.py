# src/visualisation/dashboard.py

import sys, os
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import json
import pandas as pd
import streamlit as st

from src.pipeline.run_all import run_all
from src.data.utils.helpers import load_yaml

st.set_page_config(page_title="NL Housing – Portfolio", layout="wide")

@st.cache_data(show_spinner=False)
def load_data(parquet_path: str) -> pd.DataFrame:
    return pd.read_parquet(parquet_path)

def safe_load_metrics(path: str) -> dict:
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

params = load_yaml("config/params.yaml")
parquet_path = params["processed_parquet"]

# --- Control ---
colA, colB, colC = st.columns([1,1,2])
with colA:
    if st.button("Run pipeline"):
        run_all()
        st.cache_data.clear()
        st.success("Pipeline executed.")
with colB:
    st.write("Processed file:")
    st.code(parquet_path)

# --- Data ---
if os.path.exists(parquet_path):
    df = load_data(parquet_path)
    st.metric("Rows", len(df))
    st.dataframe(df.head(20))
    num_cols = df.select_dtypes(include=["float64","float32","int64","int32"]).columns.tolist()
    if num_cols:
        st.subheader("Correlation (numeric)")
        corr = df[num_cols].corr(numeric_only=True)
        st.dataframe(corr.style.background_gradient())
else:
    st.warning("Processed parquet not found. Run the pipeline.")

# --- Model metrics ---
metrics = safe_load_metrics("data/processed/metrics.json")
if metrics:
    st.subheader("Model metrics")
    st.json(metrics)
